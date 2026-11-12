package ui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestLiveChat(t *testing.T) {
	if os.Getenv("ALPACA_LIVE_CHAT") != "1" {
		t.Skip("set ALPACA_LIVE_CHAT=1 to run live model chat test")
	}

	root := filepath.Clean(filepath.Join("..", ".."))
	exe := filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe")
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("missing llama-server.exe at %s: %v", exe, err)
	}
	evDir := t.TempDir()

	startLive := func(modelPath string, port int, logName string) *runner.Process {
		t.Helper()
		logPath := filepath.Join(t.TempDir(), logName)
		args := []string{
			"-m", modelPath,
			"--port", fmt.Sprintf("%d", port),
			"-c", "4096",
			"-ngl", "99",
			"--no-webui",
			"--metrics",
		}
		p, err := runner.Start(exe, filepath.Dir(exe), args, logPath)
		if err != nil {
			t.Fatalf("runner.Start(%s): %v", modelPath, err)
		}
		p.Port = port
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		for {
			if p.Exited() {
				_ = p.Stop()
				t.Fatalf("%s exited early: %v", modelPath, p.ExitError())
			}
			if err := p.Health(ctx); err == nil {
				return p
			}
			select {
			case <-ctx.Done():
				_ = p.Stop()
				t.Fatalf("%s did not become healthy in time", modelPath)
			case <-time.After(300 * time.Millisecond):
			}
		}
	}

	// 1. Live Qwen3.5-4B: /props + live thinking/code streaming + /think off turn 2.
	qwenModel := testModelFile(t, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf")
	pQwen := startLive(qwenModel, 18081, "qwen-live.log")
	defer pQwen.Stop()

	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.proc = pQwen
	m.activePort = 18081
	m.settings.Port = 18081
	m.width, m.height = 120, 32
	m.resizeRunViews()

	pm, ok := fetchPropsCmd(18081, "")().(propsMsg)
	if !ok || pm.err != nil {
		t.Fatalf("FetchProps qwen failed: %#v", pm)
	}
	m.modelCaps = pm.props
	if !m.thinkToggleSupported() {
		t.Fatalf("expected Qwen3.5-4B SupportsEnableThinking=true via /props")
	}

	_, cmd := m.startChatCompletion("Write a 4-line Go function Add(a, b int) int inside a go fenced code block. Be concise.")
	for cmd != nil {
		msg := cmd()
		_, cmd = m.Update(msg)
	}
	if len(m.chatMsgs) < 2 {
		t.Fatalf("expected assistant response turn")
	}
	t1 := m.chatMsgs[len(m.chatMsgs)-1]
	if t1.Err != "" {
		t.Fatalf("qwen turn 1 error: %s", t1.Err)
	}
	if !strings.Contains(t1.Content, "Add") {
		t.Fatalf("expected Go Add function in qwen output, got:\n%s", t1.Content)
	}
	if len(m.allCodeBlocks()) == 0 {
		t.Fatalf("expected parsed fenced code block in qwen response, got:\n%s", t1.Content)
	}
	frame1 := stripANSI(m.View())
	if !strings.Contains(frame1, "[c] copy") {
		t.Logf("note: code block scrolled or wrapped; full chat:\n%s", renderChatAll(m))
	}
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	_ = os.WriteFile(filepath.Join(evDir, "qwen3.5-4b_live_chat.txt"), []byte(renderChatAll(m)+"\n---\n"+frame1+"\n"), 0o644)

	// /think off live test on Qwen3.5-4B
	m.chatInput.SetValue("/think off")
	m.submitChatInput()
	_, cmd2 := m.startChatCompletion("Say just the word OK.")
	for cmd2 != nil {
		msg := cmd2()
		_, cmd2 = m.Update(msg)
	}
	t2 := m.chatMsgs[len(m.chatMsgs)-1]
	if t2.Err != "" {
		t.Fatalf("qwen turn 2 (/think off) error: %s", t2.Err)
	}
	if len(t2.Reasoning) != 0 {
		t.Fatalf("expected 0 reasoning chars with /think off on Qwen3.5-4B, got %d (%q)", len(t2.Reasoning), t2.Reasoning)
	}

	_ = pQwen.Stop()
	select {
	case <-pQwen.Done():
	case <-time.After(10 * time.Second):
		t.Fatalf("qwen process did not exit cleanly")
	}

	// 2. Live LFM2.5-8B: verify /props hides /think on|off automatically.
	lfmModel := testModelFile(t, "LFM2.5-8B-A1B-Q4_0.gguf")
	pLFM := startLive(lfmModel, 18082, "lfm-live.log")
	defer pLFM.Stop()

	m2 := testUI(t)
	m2.screen = screenRun
	m2.runTab = runTabChat
	m2.status = "READY"
	m2.proc = pLFM
	m2.activePort = 18082
	m2.settings.Port = 18082
	m2.width, m2.height = 120, 32
	m2.resizeRunViews()

	pm2, ok := fetchPropsCmd(18082, "")().(propsMsg)
	if !ok || pm2.err != nil {
		t.Fatalf("FetchProps lfm failed: %#v", pm2)
	}
	m2.modelCaps = pm2.props
	if m2.thinkToggleSupported() {
		t.Fatalf("expected LFM2.5-8B SupportsEnableThinking=false via /props")
	}
	m2.chatInput.SetValue("/help")
	m2.submitChatInput()
	lfmHelp := renderChatAll(m2)
	if strings.Contains(lfmHelp, "/think [on|off]") {
		t.Fatalf("/help showed /think [on|off] on LFM2.5-8B:\n%s", lfmHelp)
	}
	summary := fmt.Sprintf("Qwen3.5-4B: SupportsEnableThinking=%v turn1Reason=%d turn1Content=%d blocks=%d turn2ReasonOff=%d\nLFM2.5-8B: SupportsEnableThinking=%v helpHasThinkToggle=%v\n",
		m.modelCaps.SupportsEnableThinking, len(t1.Reasoning), len(t1.Content), len(m.allCodeBlocks()), len(t2.Reasoning),
		m2.modelCaps.SupportsEnableThinking, strings.Contains(lfmHelp, "/think [on|off]"))
	_ = os.WriteFile(filepath.Join(evDir, "live_chat_summary.txt"), []byte(summary), 0o644)

	_ = pLFM.Stop()
	select {
	case <-pLFM.Done():
	case <-time.After(10 * time.Second):
		t.Fatalf("lfm process did not exit cleanly")
	}
}
