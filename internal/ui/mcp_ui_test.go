package ui

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func buildTestEchoMCP(t *testing.T) string {
	t.Helper()
	binDir := t.TempDir()
	binPath := filepath.Join(binDir, "echomcp.exe")
	cmd := exec.Command("go", "build", "-o", binPath, "../mcp/testdata/echomcp")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build echomcp: %v\n%s", err, string(out))
	}
	return binPath
}

func newTestRunModel(ws string, cfg *config.Config) *Model {
	if cfg == nil {
		cfg = config.Load(ws)
	}
	m := New(ws, cfg)
	m.ready = true
	m.width, m.height = 120, 36
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.agentWorkDir = ws
	m.resizeRunViews()
	return m
}

func TestMCPPanelTogglesReloadAndNoSecretLeak(t *testing.T) {
	bin := buildTestEchoMCP(t)
	ws := t.TempDir()
	secret := "TOP_SECRET_MCP_KEY_112233"
	mcpToml := filepath.Join(ws, "mcp.toml")
	err := os.WriteFile(mcpToml, []byte(fmt.Sprintf(`
[servers.echomcp]
command = %q
args = ["-many", "12"]
enabled = true

[servers.echomcp.env]
MCP_SECRET = %q
`, bin, secret)), 0o644)
	if err != nil {
		t.Fatalf("write mcp.toml: %v", err)
	}

	m := newTestRunModel(ws, nil)
	m.mcpConfigOv = mcpToml
	defer m.Shutdown()

	m.ensureAgentReady()
	if _, ok := m.toolRegistry.Lookup("echomcp.echo"); !ok {
		t.Fatalf("expected echomcp.echo tool registered on ready")
	}

	// Open /mcp modal via command input.
	m.chatInput.SetValue("/mcp")
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.mcpModal == nil {
		t.Fatalf("expected mcpModal to open")
	}
	rendered := stripANSI(m.View())
	if !strings.Contains(rendered, "MCP Servers (/mcp)") || !strings.Contains(rendered, "echomcp") {
		t.Fatalf("expected MCP panel content, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "many tools") && !strings.Contains(rendered, "small models degrade") {
		t.Fatalf("expected large tool-list warning in MCP panel, got:\n%s", rendered)
	}
	if strings.Contains(rendered, secret) {
		t.Fatalf("secret env value leaked in /mcp panel!")
	}

	// Expand server row (enter) -> verify tool rows appear -> move down -> toggle first tool (add).
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	expandedView := stripANSI(m.View())
	if !strings.Contains(expandedView, "echomcp.add") {
		t.Fatalf("expected expanded tool rows with echomcp.add, got:\n%s", expandedView)
	}
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown}) // select echomcp.add
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(" ")})
	if _, ok := m.toolRegistry.Lookup("echomcp.add"); ok {
		t.Fatalf("expected echomcp.add disabled after space toggle")
	}

	// Close modal, test /mcp reload after updating mcp.toml.
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if m.mcpModal != nil {
		t.Fatalf("expected mcpModal closed on esc")
	}
	if err := os.WriteFile(mcpToml, []byte(fmt.Sprintf(`
[servers.echomcp]
command = %q
enabled = true
`, bin)), 0o644); err != nil {
		t.Fatalf("rewrite mcp.toml: %v", err)
	}
	m.chatInput.SetValue("/mcp reload")
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if _, ok := m.toolRegistry.Lookup("echomcp.add"); !ok {
		t.Fatalf("expected echomcp.add restored after /mcp reload")
	}
	bd := m.computeContextBreakdown()
	if bd.ToolDefsTokens <= 0 {
		t.Fatalf("expected ToolDefsTokens > 0 with MCP tools active")
	}
}

// TestMCPKillHostHelper is invoked as a subprocess by TestLiveMCPIntegration
// to verify abrupt termination (taskkill /F) leaves zero orphan MCP or grandchild processes.
func TestMCPKillHostHelper(t *testing.T) {
	if os.Getenv("MCP_KILL_HELPER") != "1" {
		return
	}
	ws := os.Getenv("MCP_WS_PATH")
	m := newTestRunModel(ws, nil)
	m.mcpConfigOv = os.Getenv("MCP_TOML_PATH")
	m.ensureAgentReady()
	if _, ok := m.toolRegistry.Lookup("echomcp.echo"); !ok {
		os.Exit(9)
	}
	_ = os.WriteFile(os.Getenv("MCP_READY_FILE"), []byte("ready"), 0o644)
	time.Sleep(60 * time.Second)
}

func TestLiveMCPIntegration(t *testing.T) {
	if os.Getenv("ALPACA_LIVE_MCP") != "1" {
		t.Skip("set ALPACA_LIVE_MCP=1 to run live llama-server + real stdio MCP integration test")
	}
	root := filepath.Clean(filepath.Join("..", ".."))
	exe := filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe")
	modelPath := testModelFile(t, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf")
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("missing llama-server.exe at %s: %v", exe, err)
	}

	bin := buildTestEchoMCP(t)
	ws := t.TempDir()
	mcpToml := filepath.Join(ws, "mcp.toml")
	if err := os.WriteFile(mcpToml, []byte(fmt.Sprintf(`
[servers.echomcp]
command = %q
enabled = true
class = "net"
`, bin)), 0o644); err != nil {
		t.Fatalf("write mcp.toml: %v", err)
	}

	port := runner.NextFreePort(18380)
	proc, err := runner.Start(exe, filepath.Dir(exe), []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"-c", "8192",
		"-ngl", "99",
		"--no-webui",
		"--metrics",
	}, filepath.Join(t.TempDir(), "qwen-mcp.log"))
	if err != nil {
		t.Fatal(err)
	}
	proc.Port = port
	defer func() {
		_ = proc.Stop()
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	for {
		if proc.Exited() {
			t.Fatalf("llama-server exited early: %v", proc.ExitError())
		}
		if err := proc.Health(ctx); err == nil {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("llama-server health timeout")
		case <-time.After(300 * time.Millisecond):
		}
	}

	m := newTestRunModel(ws, nil)
	m.mcpConfigOv = mcpToml
	m.chatBaseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	m.proc = proc
	m.ensureAgentReady()
	m.permEngine.SetMode(agent.ModeWorkspaceAuto) // Even workspace-auto MUST ask for MCP tools!

	m.chatInput.SetValue(`Call the tool echomcp.echo with text "MCP-ECHO-OK" and report its exact output.`)
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})

	sawMCPPerm := false
	deadline := time.Now().Add(75 * time.Second)
	for cmd != nil {
		if time.Now().After(deadline) {
			t.Fatalf("MCP turn timed out (sawMCPPerm=%v permModal=%v)", sawMCPPerm, m.permModal != nil)
		}
		cmd = pumpUntilModalOrDone(t, m, cmd)
		if m.permModal != nil {
			if strings.Contains(m.permModal.prompt.Tool, "echomcp") {
				sawMCPPerm = true
			}
			_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
		} else if m.askModal != nil {
			_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
		}
	}
	if !sawMCPPerm {
		t.Fatalf("expected MCP tool permission prompt even in workspace-auto mode")
	}
	allText := strings.Builder{}
	for _, tr := range m.chatMsgs {
		allText.WriteString(tr.Content)
		allText.WriteByte('\n')
		for _, tc := range tr.ToolCalls {
			allText.WriteString(tc.Output)
			allText.WriteByte('\n')
		}
	}
	if !strings.Contains(allText.String(), "MCP-ECHO-OK") {
		t.Fatalf("expected MCP-ECHO-OK in transcript/tool output, got:\n%s", allText.String())
	}

	m.Shutdown()
	_ = proc.Stop()
	m.proc = nil

	// Sub-test B: Force-kill a compiled host process running Manager + echomcp (-spawn-child)
	// and verify Job Object kills both echomcp AND its grandchild process immediately.
	killDir := t.TempDir()
	childPidFile := filepath.Join(killDir, "grandchild.pid")
	readyFile := filepath.Join(killDir, "ready.flag")
	killToml := filepath.Join(killDir, "mcp.toml")
	_ = os.WriteFile(killToml, []byte(fmt.Sprintf(`
[servers.echomcp]
command = %q
args = ["-spawn-child", "-child-pid-file", %q]
enabled = true
`, bin, childPidFile)), 0o644)

	hostBin := filepath.Join(killDir, "ui-host.test.exe")
	buildOut, err := exec.Command("go", "test", "-c", "-o", hostBin, ".").CombinedOutput()
	if err != nil {
		t.Fatalf("build kill helper binary: %v\n%s", err, string(buildOut))
	}
	hostCmd := exec.Command(hostBin, "-test.run", "^TestMCPKillHostHelper$")
	hostCmd.Env = append(os.Environ(),
		"MCP_KILL_HELPER=1",
		"MCP_TOML_PATH="+killToml,
		"MCP_WS_PATH="+killDir,
		"MCP_READY_FILE="+readyFile,
	)
	if err := hostCmd.Start(); err != nil {
		t.Fatalf("start kill helper: %v", err)
	}
	waitReady := time.Now().Add(20 * time.Second)
	for time.Now().Before(waitReady) {
		if _, err := os.Stat(readyFile); err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if _, err := os.Stat(readyFile); err != nil {
		_ = hostCmd.Process.Kill()
		t.Fatalf("kill helper never reached ready state")
	}
	pidRaw, err := os.ReadFile(childPidFile)
	if err != nil {
		_ = hostCmd.Process.Kill()
		t.Fatalf("missing grandchild pid file: %v", err)
	}
	var gcPID int
	_, _ = fmt.Sscanf(strings.TrimSpace(string(pidRaw)), "%d", &gcPID)
	if gcPID <= 0 {
		_ = hostCmd.Process.Kill()
		t.Fatalf("invalid grandchild PID %q", string(pidRaw))
	}

	_ = exec.Command("taskkill", "/F", "/PID", fmt.Sprintf("%d", hostCmd.Process.Pid)).Run()
	_ = hostCmd.Wait()

	gcDeadline := time.Now().Add(4 * time.Second)
	for time.Now().Before(gcDeadline) {
		out, _ := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", gcPID), "/NH").Output()
		if !strings.Contains(string(out), fmt.Sprintf("%d", gcPID)) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	outGC, _ := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", gcPID), "/NH").Output()
	if strings.Contains(string(outGC), fmt.Sprintf("%d", gcPID)) {
		t.Fatalf("orphan grandchild PID %d survived abrupt host kill!", gcPID)
	}
	outEcho, _ := exec.Command("tasklist", "/FI", "IMAGENAME eq echomcp.exe", "/NH").Output()
	if strings.Contains(string(outEcho), "echomcp.exe") {
		t.Fatalf("orphan echomcp.exe survived abrupt host kill:\n%s", string(outEcho))
	}
}
