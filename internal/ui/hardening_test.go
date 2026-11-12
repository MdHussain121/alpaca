package ui

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestStreaming60FPS300LinesWithCodeBlocks(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 160, 40
	m.screen = screenRun
	m.status = "READY"
	m.runTab = runTabChat
	m.resizeRunViews()

	var doc strings.Builder
	doc.WriteString("# Architectural Walkthrough\n\n")
	for sec := 1; sec <= 10; sec++ {
		doc.WriteString(fmt.Sprintf("## Section %d: Worker Pool & Pipeline\n\n", sec))
		doc.WriteString("```go\n")
		for ln := 1; ln <= 24; ln++ {
			doc.WriteString(fmt.Sprintf("func workerStep%d_%d(ctx context.Context, val int) int { return val + %d }\n", sec, ln, ln))
		}
		doc.WriteString("```\n\n")
		doc.WriteString(fmt.Sprintf("- Summary bullet for section %d with **bold** and `inline_code()`.\n\n", sec))
	}
	full := doc.String()
	totalLines := strings.Count(full, "\n")
	if totalLines < 300 {
		t.Fatalf("test document has %d lines, want >= 300", totalLines)
	}

	m.chatStreaming = true
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "Explain the 10-stage pipeline."},
		{Role: "assistant", Content: ""},
	}

	const frames = 50
	step := len(full) / frames
	var frameSum time.Duration
	for f := 1; f <= frames; f++ {
		end := f * step
		if f == frames {
			end = len(full)
		}
		m.chatMsgs[1].Content = full[:end]
		if f == frames {
			m.chatStreaming = false
		}
		t0 := time.Now()
		m.refreshChatView(true)
		v := m.View()
		frameSum += time.Since(t0)
		if len(v) == 0 {
			t.Fatal("empty view during streaming")
		}
	}
	avgFrame := frameSum / frames
	t.Logf("300-line (%d total lines) 50-frame streaming avg=%v", totalLines, avgFrame)
	if !raceDetectorActive && avgFrame > 16600*time.Microsecond {
		t.Fatalf("streaming frame budget exceeded 60fps target: avg=%v > 16.6ms", avgFrame)
	}
}

func TestMemoryAfter40TurnSession(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 160, 40
	m.screen = screenRun
	m.status = "READY"
	m.runTab = runTabChat
	m.resizeRunViews()

	for i := 1; i <= 40; i++ {
		m.chatMsgs = append(m.chatMsgs,
			chatTurn{Role: "user", Content: fmt.Sprintf("Turn %02d: inspect file module_%02d.go and summarize exports.", i, i)},
			chatTurn{
				Role:      "assistant",
				Reasoning: strings.Repeat(fmt.Sprintf("Analyzing module_%02d dependencies. ", i), 6),
				ToolCalls: []toolCard{{
					CallID:     fmt.Sprintf("call-%d", i),
					Name:       "read_file",
					ArgsPretty: fmt.Sprintf(`{"path":"module_%02d.go"}`, i),
					Summary:    "read 80 lines",
					Output:     strings.Repeat("line of file output\n", 40),
					Status:     "ok",
					Duration:   15 * time.Millisecond,
				}},
				Content: fmt.Sprintf("Turn %02d complete: `module_%02d.go` exports `Run%d()`.\n", i, i, i),
				Tokens:  48,
			},
		)
	}
	m.refreshChatView(true)
	_ = m.View()

	runtime.GC()
	debug.FreeOSMemory()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	ws, _, _ := hw.ProcessRAM(uint32(os.Getpid()))
	t.Logf("40-turn session: heap=%.1f MiB, procWS=%.1f MiB", float64(ms.HeapAlloc)/(1<<20), float64(ws)/(1<<20))
	if ms.HeapAlloc > 60<<20 {
		t.Fatalf("heap after 40 turns = %.1f MiB, want < 60 MiB", float64(ms.HeapAlloc)/(1<<20))
	}
}

func TestDataDrivenEvalBadgeComputation(t *testing.T) {
	cases := []struct {
		pct  float64
		want string
	}{
		{90.0, config.EvalBadgeRecommended},
		{80.0, config.EvalBadgeRecommended},
		{79.9, config.EvalBadgeWorksLimits},
		{50.0, config.EvalBadgeWorksLimits},
		{49.9, config.EvalBadgeNotRecommended},
		{0.0, config.EvalBadgeNotRecommended},
	}
	for _, tc := range cases {
		if got := config.ComputeEvalBadge(tc.pct); got != tc.want {
			t.Fatalf("ComputeEvalBadge(%.1f) = %q, want %q", tc.pct, got, tc.want)
		}
	}

	m := testUI(t)
	m.width, m.height = 160, 40
	m.models = []modelEntry{
		{
			cand: gguf.Candidate{RelPath: "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf"},
			meta: &gguf.Metadata{Quant: "Q4_K_M", TotalParams: 4_000_000_000, BlockCount: 36, Arch: "qwen35"},
		},
	}
	m.visible = []int{0}
	out := stripANSI(m.viewModels(56, 12))
	if !strings.Contains(out, "tools:recommended") {
		t.Fatalf("expected data-driven eval badge in launcher model list, got:\n%s", out)
	}
}

func TestRealCtrlCHelper(t *testing.T) {
	if os.Getenv("ALPACA_KILL_HELPER") != "1" {
		return
	}
	ws := os.Getenv("ALPACA_KILL_WS")
	mcpToml := os.Getenv("ALPACA_KILL_MCP_TOML")
	llamaExe := os.Getenv("ALPACA_KILL_LLAMA_EXE")
	modelPath := os.Getenv("ALPACA_KILL_MODEL")
	readyFile := os.Getenv("ALPACA_KILL_READY")

	m := newTestRunModel(ws, nil)
	m.mcpConfigOv = mcpToml
	m.ensureAgentReady()

	port := runner.NextFreePort(18620)
	proc, err := runner.Start(llamaExe, filepath.Dir(llamaExe), []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"-c", "2048",
		"-ngl", "99",
		"--no-webui",
	}, filepath.Join(ws, "llama-kill.log"))
	if err != nil {
		os.Exit(11)
	}
	proc.Port = port
	m.proc = proc
	m.chatBaseURL = fmt.Sprintf("http://127.0.0.1:%d", port)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	for {
		if proc.Exited() {
			cancel()
			os.Exit(12)
		}
		if err := proc.Health(ctx); err == nil {
			break
		}
		time.Sleep(250 * time.Millisecond)
	}
	cancel()

	// Start a long-running local command tool concurrently so Ctrl+C / Kill exercises
	// llama-server.exe + MCP child tree + run_command process tree simultaneously!
	tctx, tcancel := context.WithCancel(context.Background())
	go func() {
		rSpec, _ := m.toolRegistry.Lookup("run_command")
		_ = rSpec.Execute(tctx, agent.ToolContext{WorkspaceRoot: ws}, map[string]any{
			"command":   "ping -n 60 127.0.0.1 >NUL",
			"timeout_s": 60,
		})
	}()
	time.Sleep(300 * time.Millisecond)
	_ = os.WriteFile(readyFile, []byte(fmt.Sprintf("%d", proc.PID())), 0o644)

	if os.Getenv("ALPACA_KILL_MODE") == "ctrlc" {
		// Trigger Bubble Tea Ctrl+C handler + Shutdown()
		_, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
		tcancel()
		m.Shutdown()
		_ = os.WriteFile(readyFile+".done", []byte("ctrlc-clean"), 0o644)
		return
	}
	// Otherwise stay alive until external taskkill /F (Task Manager kill) terminates the host PID!
	time.Sleep(60 * time.Second)
	tcancel()
}

func TestLiveCtrlCAndTaskManagerKillOrphanCheck(t *testing.T) {
	if os.Getenv("ALPACA_LIVE_ORPHAN_HARDENING") != "1" {
		t.Skip("set ALPACA_LIVE_ORPHAN_HARDENING=1 to run live Ctrl+C & taskkill /F orphan verification")
	}
	root := filepath.Clean(filepath.Join("..", ".."))
	llamaExe := filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe")
	modelPath := testModelFile(t, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf")
	if _, err := os.Stat(llamaExe); err != nil {
		t.Fatalf("missing llama-server.exe: %v", err)
	}

	echoBin := buildTestEchoMCP(t)
	tmpDir := t.TempDir()
	hostBin := filepath.Join(tmpDir, "alpaca-kill.test.exe")
	if out, err := exec.Command("go", "test", "-c", "-o", hostBin, ".").CombinedOutput(); err != nil {
		t.Fatalf("compile helper: %v\n%s", err, string(out))
	}

	for _, mode := range []string{"ctrlc", "taskkill"} {
		t.Run(mode, func(t *testing.T) {
			ws := filepath.Join(tmpDir, mode)
			_ = os.MkdirAll(ws, 0o755)
			gcPidFile := filepath.Join(ws, "gc.pid")
			readyFile := filepath.Join(ws, "ready.txt")
			mcpToml := filepath.Join(ws, "mcp.toml")
			_ = os.WriteFile(mcpToml, []byte(fmt.Sprintf(`
[servers.echomcp]
command = %q
args = ["-spawn-child", "-child-pid-file", %q]
enabled = true
`, echoBin, gcPidFile)), 0o644)

			cmd := exec.Command(hostBin, "-test.run", "^TestRealCtrlCHelper$")
			cmd.Env = append(os.Environ(),
				"ALPACA_KILL_HELPER=1",
				"ALPACA_KILL_MODE="+mode,
				"ALPACA_KILL_WS="+ws,
				"ALPACA_KILL_MCP_TOML="+mcpToml,
				"ALPACA_KILL_LLAMA_EXE="+llamaExe,
				"ALPACA_KILL_MODEL="+modelPath,
				"ALPACA_KILL_READY="+readyFile,
			)
			if err := cmd.Start(); err != nil {
				t.Fatalf("start helper: %v", err)
			}
			deadline := time.Now().Add(45 * time.Second)
			for time.Now().Before(deadline) {
				if _, err := os.Stat(readyFile); err == nil {
					break
				}
				time.Sleep(100 * time.Millisecond)
			}
			pidBytes, err := os.ReadFile(readyFile)
			if err != nil {
				_ = cmd.Process.Kill()
				t.Fatalf("helper never became ready: %v", err)
			}
			var llamaPID, gcPID int
			_, _ = fmt.Sscanf(strings.TrimSpace(string(pidBytes)), "%d", &llamaPID)
			if gcBytes, gErr := os.ReadFile(gcPidFile); gErr == nil {
				_, _ = fmt.Sscanf(strings.TrimSpace(string(gcBytes)), "%d", &gcPID)
			}

			if mode == "ctrlc" {
				doneDead := time.Now().Add(15 * time.Second)
				for time.Now().Before(doneDead) {
					if _, err := os.Stat(readyFile + ".done"); err == nil {
						break
					}
					time.Sleep(100 * time.Millisecond)
				}
				_ = cmd.Wait()
			} else {
				_ = exec.Command("taskkill", "/F", "/PID", fmt.Sprintf("%d", cmd.Process.Pid)).Run()
				_ = cmd.Wait()
			}

			// Allow brief kernel cleanup, then assert ZERO orphan llama-server, echomcp, or grandchild PIDs!
			checkDead := time.Now().Add(4 * time.Second)
			for time.Now().Before(checkDead) {
				outL, _ := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", llamaPID), "/NH").Output()
				outE, _ := exec.Command("tasklist", "/FI", "IMAGENAME eq echomcp.exe", "/NH").Output()
				if !strings.Contains(string(outL), fmt.Sprintf("%d", llamaPID)) &&
					!strings.Contains(string(outE), "echomcp.exe") {
					break
				}
				time.Sleep(50 * time.Millisecond)
			}
			outL, _ := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", llamaPID), "/NH").Output()
			if strings.Contains(string(outL), fmt.Sprintf("%d", llamaPID)) {
				t.Fatalf("orphan llama-server PID %d survived %s!", llamaPID, mode)
			}
			outE, _ := exec.Command("tasklist", "/FI", "IMAGENAME eq echomcp.exe", "/NH").Output()
			if strings.Contains(string(outE), "echomcp.exe") {
				t.Fatalf("orphan echomcp.exe survived %s!", mode)
			}
			if gcPID > 0 {
				outGC, _ := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", gcPID), "/NH").Output()
				if strings.Contains(string(outGC), fmt.Sprintf("%d", gcPID)) {
					t.Fatalf("orphan grandchild PID %d survived %s!", gcPID, mode)
				}
			}
		})
	}
}
