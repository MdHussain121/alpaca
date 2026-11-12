//go:build livecheck

package ui

import (
	"bytes"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"

	"github.com/MdHussain121/alpaca/internal/runner"
)

type conptyProc struct {
	pid    uint32
	procH  windows.Handle
	hpc    windows.Handle
	inW    *os.File
	outR   *os.File
	bufMu  sync.Mutex
	buf    bytes.Buffer
	doneCh chan uint32
}

func startInConPTY(t *testing.T, exePath, workDir string, env []string) *conptyProc {
	t.Helper()

	var ptyInR, ptyInW, ptyOutR, ptyOutW windows.Handle
	if err := windows.CreatePipe(&ptyInR, &ptyInW, nil, 0); err != nil {
		t.Fatalf("CreatePipe in: %v", err)
	}
	if err := windows.CreatePipe(&ptyOutR, &ptyOutW, nil, 0); err != nil {
		_ = windows.CloseHandle(ptyInR)
		_ = windows.CloseHandle(ptyInW)
		t.Fatalf("CreatePipe out: %v", err)
	}

	var hpc windows.Handle
	if err := windows.CreatePseudoConsole(windows.Coord{X: 120, Y: 36}, ptyInR, ptyOutW, 0, &hpc); err != nil {
		_ = windows.CloseHandle(ptyInR)
		_ = windows.CloseHandle(ptyInW)
		_ = windows.CloseHandle(ptyOutR)
		_ = windows.CloseHandle(ptyOutW)
		t.Fatalf("CreatePseudoConsole: %v", err)
	}
	// Child uses ptyInR / ptyOutW through ConPTY; parent keeps ptyInW / ptyOutR.
	_ = windows.CloseHandle(ptyInR)
	_ = windows.CloseHandle(ptyOutW)

	attrList, err := windows.NewProcThreadAttributeList(1)
	if err != nil {
		windows.ClosePseudoConsole(hpc)
		t.Fatalf("NewProcThreadAttributeList: %v", err)
	}
	if err := attrList.Update(windows.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, unsafe.Pointer(hpc), unsafe.Sizeof(hpc)); err != nil {
		attrList.Delete()
		windows.ClosePseudoConsole(hpc)
		t.Fatalf("Update PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE: %v", err)
	}

	var siEx windows.StartupInfoEx
	siEx.Cb = uint32(unsafe.Sizeof(siEx))
	siEx.Flags = windows.STARTF_USESTDHANDLES
	siEx.ProcThreadAttributeList = attrList.List()

	cmdPtr, err := windows.UTF16PtrFromString(`"` + exePath + `"`)
	if err != nil {
		t.Fatal(err)
	}
	dirPtr, err := windows.UTF16PtrFromString(workDir)
	if err != nil {
		t.Fatal(err)
	}
	envBlock := createEnvBlock(env)

	var pi windows.ProcessInformation
	flags := uint32(windows.EXTENDED_STARTUPINFO_PRESENT | windows.CREATE_UNICODE_ENVIRONMENT)
	err = windows.CreateProcess(
		nil,
		cmdPtr,
		nil,
		nil,
		false,
		flags,
		envBlock,
		dirPtr,
		&siEx.StartupInfo,
		&pi,
	)
	attrList.Delete()
	if err != nil {
		windows.ClosePseudoConsole(hpc)
		_ = windows.CloseHandle(ptyInW)
		_ = windows.CloseHandle(ptyOutR)
		t.Fatalf("CreateProcess(%s): %v", exePath, err)
	}
	_ = windows.CloseHandle(pi.Thread)

	cp := &conptyProc{
		pid:    pi.ProcessId,
		procH:  pi.Process,
		hpc:    hpc,
		inW:    os.NewFile(uintptr(ptyInW), "pty-in"),
		outR:   os.NewFile(uintptr(ptyOutR), "pty-out"),
		doneCh: make(chan uint32, 1),
	}

	go func() {
		tmp := make([]byte, 4096)
		for {
			n, rerr := cp.outR.Read(tmp)
			if n > 0 {
				cp.bufMu.Lock()
				cp.buf.Write(tmp[:n])
				cp.bufMu.Unlock()
			}
			if rerr != nil {
				return
			}
		}
	}()

	go func() {
		_, _ = windows.WaitForSingleObject(cp.procH, windows.INFINITE)
		var ec uint32
		_ = windows.GetExitCodeProcess(cp.procH, &ec)
		cp.doneCh <- ec
	}()

	return cp
}

func createEnvBlock(env []string) *uint16 {
	var utf16 []uint16
	for _, entry := range env {
		u, err := windows.UTF16FromString(entry)
		if err != nil {
			continue
		}
		utf16 = append(utf16, u...)
	}
	utf16 = append(utf16, 0)
	return &utf16[0]
}

func (cp *conptyProc) output() string {
	cp.bufMu.Lock()
	defer cp.bufMu.Unlock()
	return cp.buf.String()
}

func (cp *conptyProc) send(s string) error {
	_, err := cp.inW.Write([]byte(s))
	return err
}

func (cp *conptyProc) waitContains(substr string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if strings.Contains(cp.output(), substr) {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return false
}

func (cp *conptyProc) waitExit(timeout time.Duration) (uint32, bool) {
	select {
	case ec := <-cp.doneCh:
		return ec, true
	case <-time.After(timeout):
		return 0, false
	}
}

func (cp *conptyProc) close() {
	_ = cp.inW.Close()
	windows.ClosePseudoConsole(cp.hpc)
	_ = cp.outR.Close()
	_ = windows.CloseHandle(cp.procH)
	time.Sleep(150 * time.Millisecond)
}

func listLlamaServerPIDs(t *testing.T) []string {
	t.Helper()
	out, err := exec.Command("tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH").Output()
	if err != nil {
		t.Fatalf("tasklist: %v", err)
	}
	var pids []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "INFO:") {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) >= 2 && strings.Contains(strings.ToLower(fields[0]), "llama-server.exe") {
			pids = append(pids, strings.Trim(fields[1], `"`))
		}
	}
	return pids
}

func waitHealthy(port int, timeout time.Duration) bool {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	cli := &http.Client{Timeout: 500 * time.Millisecond}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := cli.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return true
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}

func setupLiveRoot(t *testing.T, port int, modelPath string) string {
	t.Helper()
	root := t.TempDir()
	distBin, err := filepath.Abs(filepath.Join("..", "..", "dist", "bin"))
	if err != nil {
		t.Fatal(err)
	}
	// Junction to dist\bin so llama-server launches with real DLLs at full speed.
	binLink := filepath.Join(root, "bin")
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", binLink, distBin).CombinedOutput(); err != nil {
		t.Fatalf("mklink bin: %v: %s", err, string(out))
	}
	t.Cleanup(func() {
		_ = exec.Command("cmd", "/c", "rmdir", binLink).Run()
	})
	modelsDir := filepath.Join(root, "models")
	_ = os.MkdirAll(modelsDir, 0o755)
	_ = os.MkdirAll(filepath.Join(root, "presets"), 0o755)
	_ = os.MkdirAll(filepath.Join(root, "logs"), 0o755)

	cfg := fmt.Sprintf(`theme = "mocha"
backend = "cuda"
port = %d
models_dir = %q
extra_models = [%q]

[last_used]
`, port, modelsDir, modelPath)
	if err := os.WriteFile(filepath.Join(root, "config.toml"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestRealOrphanAndCtrlC(t *testing.T) {
	alpacaExe, err := filepath.Abs(filepath.Join("..", "..", "dist", "alpaca.exe"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(alpacaExe); err != nil {
		t.Fatalf("missing %s", alpacaExe)
	}
	modelPath := testModelFile(t, "Spark-X2.6-4B-abliterated-Q4_K_M.gguf")
	if pids := listLlamaServerPIDs(t); len(pids) > 0 {
		t.Fatalf("pre-existing llama-server processes before test: %v", pids)
	}

	startCase := func(t *testing.T, port int) (*conptyProc, string) {
		root := setupLiveRoot(t, port, modelPath)
		env := append(os.Environ(), "ALPACA_ROOT="+root)
		cp := startInConPTY(t, alpacaExe, filepath.Dir(alpacaExe), env)
		if !cp.waitContains("Spark-X2.6-4B", 8*time.Second) {
			out := cp.output()
			cp.close()
			t.Fatalf("TUI did not render Spark model (len=%d); raw=%q", len(out), out)
		}
		return cp, root
	}

	t.Run("ctrlC_launcher", func(t *testing.T) {
		port := runner.NextFreePort(18310)
		cp, _ := startCase(t, port)
		defer cp.close()
		t0 := time.Now()
		_ = cp.send("\x03")
		if _, ok := cp.waitExit(1 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not exit within 1s on launcher Ctrl+C")
		}
		t.Logf("launcher Ctrl+C exit in %v", time.Since(t0).Round(time.Millisecond))
		if pids := listLlamaServerPIDs(t); len(pids) > 0 {
			t.Fatalf("leftover llama-server: %v", pids)
		}
	})

	t.Run("ctrlC_help", func(t *testing.T) {
		port := runner.NextFreePort(18311)
		cp, _ := startCase(t, port)
		defer cp.close()
		_ = cp.send("?")
		if !cp.waitContains("press ? or esc to close", 2*time.Second) {
			t.Fatalf("help screen did not open; raw=%q", cp.output())
		}
		t0 := time.Now()
		_ = cp.send("\x03")
		if _, ok := cp.waitExit(1 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not exit within 1s on help Ctrl+C")
		}
		t.Logf("help Ctrl+C exit in %v", time.Since(t0).Round(time.Millisecond))
	})

	t.Run("ctrlC_presets", func(t *testing.T) {
		port := runner.NextFreePort(18312)
		cp, _ := startCase(t, port)
		defer cp.close()
		_ = cp.send("p")
		if !cp.waitContains("no presets yet", 2*time.Second) {
			t.Fatalf("presets overlay did not open; raw=%q", cp.output())
		}
		t0 := time.Now()
		_ = cp.send("\x03")
		if _, ok := cp.waitExit(1 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not exit within 1s on presets Ctrl+C")
		}
		t.Logf("presets Ctrl+C exit in %v", time.Since(t0).Round(time.Millisecond))
	})

	t.Run("ctrlC_modal_edit", func(t *testing.T) {
		port := runner.NextFreePort(18313)
		cp, _ := startCase(t, port)
		defer cp.close()
		_ = cp.send("\t")
		time.Sleep(100 * time.Millisecond)
		_ = cp.send("\r")
		time.Sleep(150 * time.Millisecond)
		t0 := time.Now()
		_ = cp.send("\x03")
		if _, ok := cp.waitExit(1 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not exit within 1s on edit modal Ctrl+C")
		}
		t.Logf("edit modal Ctrl+C exit in %v", time.Since(t0).Round(time.Millisecond))
	})

	t.Run("ctrlC_during_launch", func(t *testing.T) {
		port := runner.NextFreePort(18314)
		cp, _ := startCase(t, port)
		defer cp.close()
		_ = cp.send("\r")
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) && len(listLlamaServerPIDs(t)) == 0 {
			time.Sleep(50 * time.Millisecond)
		}
		if len(listLlamaServerPIDs(t)) == 0 {
			t.Fatalf("llama-server.exe never spawned on launch")
		}
		t0 := time.Now()
		_ = cp.send("\x03")
		if _, ok := cp.waitExit(1 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not exit within 1s on launch Ctrl+C")
		}
		t.Logf("during-launch Ctrl+C exit in %v", time.Since(t0).Round(time.Millisecond))
		time.Sleep(300 * time.Millisecond)
		if pids := listLlamaServerPIDs(t); len(pids) > 0 {
			t.Fatalf("ORPHAN llama-server after during-launch Ctrl+C: %v", pids)
		}
	})

	t.Run("ctrlC_run_streaming", func(t *testing.T) {
		port := runner.NextFreePort(18315)
		cp, _ := startCase(t, port)
		defer cp.close()
		_ = cp.send("\r")
		if !waitHealthy(port, 45*time.Second) {
			t.Fatalf("llama-server failed to become healthy on port %d", port)
		}
		if !cp.waitContains("READY", 5*time.Second) {
			t.Fatalf("run view did not reach READY")
		}
		_ = cp.send("Write a long detailed story about compilers in 400 words.\r")
		time.Sleep(600 * time.Millisecond)
		t0 := time.Now()
		_ = cp.send("\x03")
		if _, ok := cp.waitExit(1 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not exit within 1s during streaming Ctrl+C")
		}
		t.Logf("run streaming Ctrl+C exit in %v", time.Since(t0).Round(time.Millisecond))
		time.Sleep(300 * time.Millisecond)
		if pids := listLlamaServerPIDs(t); len(pids) > 0 {
			t.Fatalf("ORPHAN llama-server after streaming Ctrl+C: %v", pids)
		}
	})

	t.Run("taskkill_hard_alpaca_job_object", func(t *testing.T) {
		port := runner.NextFreePort(18316)
		cp, _ := startCase(t, port)
		defer cp.close()
		_ = cp.send("\r")
		if !waitHealthy(port, 45*time.Second) {
			t.Fatalf("llama-server failed to become healthy on port %d", port)
		}
		pidsBefore := listLlamaServerPIDs(t)
		if len(pidsBefore) != 1 {
			t.Fatalf("expected 1 llama-server running, got %v", pidsBefore)
		}
		t0 := time.Now()
		if out, err := exec.Command("taskkill", "/F", "/PID", fmt.Sprintf("%d", cp.pid)).CombinedOutput(); err != nil {
			t.Fatalf("taskkill alpaca.exe failed: %v: %s", err, string(out))
		}
		if _, ok := cp.waitExit(2 * time.Second); !ok {
			t.Fatalf("alpaca.exe did not die on taskkill /F")
		}
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) && len(listLlamaServerPIDs(t)) > 0 {
			time.Sleep(50 * time.Millisecond)
		}
		if pids := listLlamaServerPIDs(t); len(pids) > 0 {
			t.Fatalf("ORPHAN llama-server %v survived >3s after hard kill of alpaca.exe (Job Object failed!)", pids)
		}
		t.Logf("hard taskkill /F killed alpaca.exe (PID %d) + llama-server (PID %v) in %v", cp.pid, pidsBefore, time.Since(t0).Round(time.Millisecond))
	})
}
