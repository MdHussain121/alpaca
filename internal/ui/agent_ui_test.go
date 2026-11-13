package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func pumpUntilModalOrDone(t *testing.T, m *Model, cmd tea.Cmd) tea.Cmd {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for cmd != nil {
		if time.Now().After(deadline) {
			t.Fatalf("timed out pumping UI agent loop (permModal=%v askModal=%v streaming=%v)", m.permModal != nil, m.askModal != nil, m.chatStreaming)
		}
		msg := cmd()
		_, cmd = m.Update(msg)
	}
	return cmd
}

func TestAgentLoopUIPermissionsDiffUndoToolsAndAskUser(t *testing.T) {
	ws := t.TempDir()
	origFile := filepath.Join(ws, "app.txt")
	if err := os.WriteFile(origFile, []byte("hello world\nline two\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var reqBodies []map[string]any
	step := 0

	writeSSE := func(w http.ResponseWriter, delta any, finish string, usage bool) {
		b, _ := json.Marshal(delta)
		if finish != "" {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":%s,\"finish_reason\":%q}]}\n\n", b, finish)
		} else {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":%s,\"finish_reason\":null}]}\n\n", b)
		}
		if usage {
			fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":310,\"completion_tokens\":24,\"prompt_tokens_details\":{\"cached_tokens\":120}},\"timings\":{\"cache_n\":120,\"predicted_per_second\":62.5}}\n\n")
		}
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions/input_tokens":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			w.Header().Set("Content-Type", "application/json")
			if _, hasTools := body["tools"]; hasTools {
				_, _ = w.Write([]byte(`{"input_tokens":620,"object":"response.input_tokens"}`))
			} else {
				_, _ = w.Write([]byte(`{"input_tokens":10,"object":"response.input_tokens"}`))
			}
		case "/v1/chat/completions":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			mu.Lock()
			reqBodies = append(reqBodies, body)
			cur := step
			step++
			mu.Unlock()

			w.Header().Set("Content-Type", "text/event-stream")
			switch cur {
			case 0:
				// Step 1: read_file (auto-allowed in read-auto)
				writeSSE(w, map[string]any{"role": "assistant"}, "", false)
				writeSSE(w, map[string]any{"tool_calls": []map[string]any{{
					"index":    0,
					"id":       "call_read",
					"type":     "function",
					"function": map[string]any{"name": "read_file", "arguments": `{"path":"app.txt"}`},
				}}}, "tool_calls", false)
				fmt.Fprint(w, "data: [DONE]\n\n")
			case 1:
				// Step 2: write_file (will be denied with note!)
				writeSSE(w, map[string]any{"tool_calls": []map[string]any{{
					"index":    0,
					"id":       "call_write_bad",
					"type":     "function",
					"function": map[string]any{"name": "write_file", "arguments": `{"path":"app.txt","content":"completely replaced\n"}`},
				}}}, "tool_calls", false)
				fmt.Fprint(w, "data: [DONE]\n\n")
			case 2:
				// Step 3: after seeing deny note, propose targeted edit_file (will be allowed once)
				writeSSE(w, map[string]any{"tool_calls": []map[string]any{{
					"index":    0,
					"id":       "call_edit_ok",
					"type":     "function",
					"function": map[string]any{"name": "edit_file", "arguments": `{"path":"app.txt","old_text":"hello world","new_text":"hello agent"}`},
				}}}, "tool_calls", false)
				fmt.Fprint(w, "data: [DONE]\n\n")
			case 3:
				// Step 4: ask_user modal
				writeSSE(w, map[string]any{"tool_calls": []map[string]any{{
					"index":    0,
					"id":       "call_ask",
					"type":     "function",
					"function": map[string]any{"name": "ask_user", "arguments": `{"question":"Keep tagline?","options":["Yes","No"]}`},
				}}}, "tool_calls", false)
				fmt.Fprint(w, "data: [DONE]\n\n")
			default:
				// Step 5: final answer
				writeSSE(w, map[string]any{"content": "Updated `app.txt` safely."}, "stop", true)
				fmt.Fprint(w, "data: [DONE]\n\n")
			}
		}
	}))
	defer srv.Close()

	m := testUI(t)
	m.agentWorkDir = ws
	m.checkpoints = agent.NewCheckpointStore(ws, "test-sess")
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 34
	m.chatBaseURL = srv.URL
	m.resizeRunViews()

	// Check status bar badges
	v0 := stripANSI(m.View())
	if !strings.Contains(v0, "perm:read-auto") || !strings.Contains(v0, "ws:"+filepath.Base(ws)) {
		t.Fatalf("status bar missing perm/ws badges:\n%s", v0)
	}

	// Open /tools modal -> measures /input_tokens (620 - 10 = 610) & toggle exec off/on
	m.chatInput.SetValue("/tools")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil {
		msg := cmd()
		m.Update(msg)
	}
	if m.toolsModal == nil {
		t.Fatal("expected /tools to open modal")
	}
	if m.toolSchemaTokens != 610 || !m.toolSchemaLive {
		t.Fatalf("expected live tool schema tokens 610, got %d (live=%v)", m.toolSchemaTokens, m.toolSchemaLive)
	}
	// Toggle files/exec and close modal
	m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if m.toolsModal != nil {
		t.Fatal("expected esc to close tools modal")
	}

	// Submit user prompt
	m.chatInput.SetValue("Update app.txt safely")
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	// Should pause at PermissionModal for write_file with colored unified diff preview
	if m.permModal == nil || m.permModal.prompt.Tool != "write_file" {
		t.Fatalf("expected permission modal for write_file, got %+v", m.permModal)
	}
	permView := stripANSI(m.View())
	if !strings.Contains(permView, "-hello world") || !strings.Contains(permView, "+completely replaced") {
		t.Fatalf("permission dialog missing unified diff lines:\n%s", permView)
	}

	// Deny with note ('d' -> type note -> Enter)
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	for _, r := range "use edit_file instead" {
		_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	// Should pause next at PermissionModal for edit_file
	if m.permModal == nil || m.permModal.prompt.Tool != "edit_file" {
		t.Fatalf("expected permission modal for edit_file after deny note, got %+v", m.permModal)
	}
	diffView := stripANSI(m.View())
	if !strings.Contains(diffView, "+hello agent") || !strings.Contains(diffView, "+1 / -1") {
		t.Fatalf("edit_file diff modal missing expected delta:\n%s", diffView)
	}
	// Allow once ('y')
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	// Should pause next at AskUser modal with options ["Yes", "No"]
	if m.askModal == nil || m.askModal.question != "Keep tagline?" {
		t.Fatalf("expected askModal, got %+v", m.askModal)
	}
	// Pick option 1 ('1' or enter)
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'1'}})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	if m.chatStreaming {
		t.Fatal("expected chatStreaming=false after final answer")
	}

	// Check file updated on disk
	data, _ := os.ReadFile(origFile)
	if !strings.Contains(string(data), "hello agent") {
		t.Fatalf("expected edited file on disk, got %q", string(data))
	}

	// Verify final view renders tool cards and content
	finalView := stripANSI(m.View())
	for _, want := range []string{"read_file", "write_file", "edit_file", "ask_user", "Updated app.txt safely."} {
		if !strings.Contains(finalView, want) {
			t.Fatalf("final view missing %q:\n%s", want, finalView)
		}
	}

	// Verify deny note actually reached the model in step 2's messages
	mu.Lock()
	b, _ := json.Marshal(reqBodies[2]["messages"])
	mu.Unlock()
	if !strings.Contains(string(b), "use edit_file instead") {
		t.Fatalf("deny note did not propagate to LLM history: %s", string(b))
	}

	// Toggle expand tool cards with ctrl+o
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	expView := stripANSI(m.View())
	if !strings.Contains(expView, "Args:") {
		t.Fatalf("ctrl+o did not expand tool cards:\n%s", expView)
	}

	// Verify /undo restores original file and warns about run_command
	if cmdEntry, ok := m.lookupCommand("undo"); !ok {
		t.Fatal("missing /undo command")
	} else if en, _ := cmdEntry.Enabled(m); !en {
		t.Fatal("expected /undo to be enabled before undoing")
	}
	m.chatInput.SetValue("/undo")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	restored, _ := os.ReadFile(origFile)
	if string(restored) != "hello world\nline two\n" {
		t.Fatalf("/undo did not restore original content, got %q", string(restored))
	}
	undoView := stripANSI(m.View())
	if !strings.Contains(undoView, "run_command") || !strings.Contains(undoView, "restored: app.txt") {
		t.Fatalf("/undo info message missing restored file or command warning:\n%s", undoView)
	}
	if cmdEntry, ok := m.lookupCommand("undo"); ok {
		if en, reason := cmdEntry.Enabled(m); en || reason != "nothing to undo" {
			t.Fatalf("expected /undo disabled after undo, got enabled=%v reason=%q", en, reason)
		}
	}

	// Verify /permissions switches mode and updates badge
	m.chatInput.SetValue("/permissions workspace-auto")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if !strings.Contains(stripANSI(m.View()), "perm:workspace-auto") {
		t.Fatalf("badge did not update to perm:workspace-auto")
	}
}

func TestSessionAllowAndEscInterruptsRunningCommand(t *testing.T) {
	ws := t.TempDir()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_sleep\",\"type\":\"function\",\"function\":{\"name\":\"run_command\",\"arguments\":%q}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", `{"command":"ping -n 10 127.0.0.1"}`)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	m := testUI(t)
	m.agentWorkDir = ws
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 32
	m.chatBaseURL = srv.URL
	m.resizeRunViews()

	// Also verify /tools exec on emits OS user-rights warning once
	m.chatInput.SetValue("/tools exec on")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if !strings.Contains(stripANSI(m.View()), "Windows user permissions") {
		t.Fatalf("expected exec honesty notice in chat log")
	}

	m.chatInput.SetValue("run slow ping")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	if m.permModal == nil || m.permModal.prompt.Tool != "run_command" {
		t.Fatalf("expected run_command permission modal")
	}
	// Approve once: run_command starts executing while readAgentEvtCmd waits for EvtToolEnd.
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	if cmd == nil {
		t.Fatal("expected readAgentEvtCmd after approving run_command")
	}

	msgCh := make(chan tea.Msg, 1)
	go func() {
		msgCh <- cmd()
	}()
	time.Sleep(150 * time.Millisecond)

	startAbort := time.Now()
	m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if time.Since(startAbort) > time.Second {
		t.Fatalf("Esc abort blocked too long")
	}
	if m.chatStreaming {
		t.Fatal("expected chatStreaming=false immediately after Esc")
	}
	if len(m.chatMsgs) == 0 || !m.chatMsgs[len(m.chatMsgs)-1].Aborted {
		t.Fatalf("expected last turn marked aborted")
	}
	select {
	case <-msgCh:
	case <-time.After(2 * time.Second):
		t.Fatal("running command + background cmd did not unwind cleanly within 2s after Esc")
	}
}

func TestAgentCardGoldenSnapshots(t *testing.T) {
	update := os.Getenv("UPDATE_GOLDEN") == "1"

	for _, sz := range []struct {
		w, h int
		tag  string
	}{
		{80, 24, "80x24"},
		{120, 32, "120x32"},
		{160, 40, "160x40"},
	} {
		t.Run("toolcards-"+sz.tag, func(t *testing.T) {
			m := testUI(t)
			m.agentWorkDir = "demo-ws"
			m.screen = screenRun
			m.runTab = runTabChat
			m.status = "READY"
			m.width, m.height = sz.w, sz.h
			m.resizeRunViews()

			exp := true
			m.chatMsgs = []chatTurn{
				{Role: "user", Content: "Add Greet to main.go"},
				{
					Role: "assistant",
					ToolCalls: []toolCard{
						{
							CallID:   "c1",
							Name:     "read_file",
							Summary:  "main.go",
							Status:   "ok",
							Duration: 4 * time.Millisecond,
						},
						{
							CallID:      "c2",
							Name:        "edit_file",
							Summary:     "main.go",
							ArgsPretty:  `{"path":"main.go"}`,
							Diff:        "--- a/main.go\n+++ b/main.go\n@@ -1,2 +1,3 @@\n package main\n+func Greet() {}\n",
							DiffAdded:   1,
							DiffRemoved: 0,
							Status:      "ok",
							Duration:    12 * time.Millisecond,
							Expanded:    &exp,
						},
					},
					Content: "Added `Greet()` to `main.go`.",
				},
			}
			m.refreshChatView(false)

			got := strings.TrimSpace(stripANSI(m.View()))
			assertNamedGolden(t, "toolcards-"+sz.tag+".golden", got, update)
		})

		t.Run("permdiff-"+sz.tag, func(t *testing.T) {
			m := testUI(t)
			m.agentWorkDir = "demo-ws"
			m.screen = screenRun
			m.runTab = runTabChat
			m.status = "READY"
			m.width, m.height = sz.w, sz.h
			m.resizeRunViews()
			m.permModal = &permModalState{
				prompt: agent.PermissionPrompt{
					CallID:      "c_perm",
					Tool:        "edit_file",
					Summary:     "src/config.go",
					Reason:      "file modification requires approval",
					DiffPath:    "src/config.go",
					DiffAdded:   1,
					DiffRemoved: 1,
					Diff:        "--- a/src/config.go\n+++ b/src/config.go\n@@ -1,3 +1,3 @@\n package cfg\n-var Port = 8080\n+var Port = 9090\n",
				},
			}
			m.refreshChatView(true)

			got := strings.TrimSpace(stripANSI(m.View()))
			assertNamedGolden(t, "permdiff-"+sz.tag+".golden", got, update)
		})
	}
}

func TestLiveAgentTools(t *testing.T) {
	if os.Getenv("ALPACA_LIVE_AGENT") != "1" {
		t.Skip("set ALPACA_LIVE_AGENT=1 to run live model agent tools test")
	}

	root := filepath.Clean(filepath.Join("..", ".."))
	exe := filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe")
	modelPath := testModelFile(t, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf")
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("missing llama-server.exe: %v", err)
	}

	evDir := t.TempDir()

	port := runner.NextFreePort(18160)
	logPath := filepath.Join(t.TempDir(), "qwen3.5-agent.log")
	proc, err := runner.Start(exe, filepath.Dir(exe), []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"-c", "8192",
		"-ngl", "99",
		"--no-webui",
		"--metrics",
	}, logPath)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	proc.Port = port
	defer func() { _ = proc.Stop() }()

	hctx, hcancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer hcancel()
	for {
		if proc.Exited() {
			t.Fatalf("llama-server exited early: %v", proc.ExitError())
		}
		if err := proc.Health(hctx); err == nil {
			break
		}
		select {
		case <-hctx.Done():
			t.Fatalf("llama-server did not become healthy in time")
		case <-time.After(300 * time.Millisecond):
		}
	}

	// Build a temporary workspace and let Qwen3.5-4B run multi-step agent tools:
	// create subfolder, write two files, and run a shell command to inspect/verify.
	ws := t.TempDir()
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	reg := agent.NewDefaultRegistry()
	perms := agent.NewPermissionEngine(agent.ModeWorkspaceAuto)
	ck := agent.NewCheckpointStore(ws, "live-agent")

	var mu sync.Mutex
	var events []string
	var transcript strings.Builder

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	res := agent.RunTurn(ctx, agent.LoopRequest{
		BaseURL:      baseURL,
		SystemPrompt: "You are a concise coding agent inside an isolated workspace. Always use the provided file and command tools directly instead of guessing.",
		History: []runner.APIMessage{{
			Role:    "user",
			Content: "Create a subfolder named `pkgdemo`, write `pkgdemo/alpha.txt` containing `hello-alpha` and `pkgdemo/beta.txt` containing `hello-beta`, then run a shell command (`dir pkgdemo` or `cmd /c type pkgdemo\\alpha.txt`) via run_command to verify both files exist, and report the result.",
		}},
		MaxSteps:    12,
		MaxTokens:   1024,
		Temperature: 0.2,
		Registry:    reg,
		Permissions: perms,
		ToolCtx: agent.ToolContext{
			WorkspaceRoot: ws,
			SessionID:     "live-agent",
			TurnNumber:    1,
			Checkpoints:   ck,
			Shell:         agent.ShellCmd,
		},
		AskPermission: func(ctx context.Context, pr agent.PermissionPrompt) (agent.PermissionResponse, error) {
			mu.Lock()
			events = append(events, fmt.Sprintf("PERM:%s(%s)", pr.Tool, pr.Summary))
			mu.Unlock()
			return agent.PermissionResponse{Choice: agent.PermAllowOnce}, nil
		},
		OnEvent: func(ev agent.LoopEvent) {
			mu.Lock()
			defer mu.Unlock()
			switch ev.Kind {
			case agent.EvtToolStart:
				events = append(events, fmt.Sprintf("START:%s(%s)", ev.Tool, ev.Summary))
			case agent.EvtToolEnd:
				events = append(events, fmt.Sprintf("END:%s(err=%v)", ev.Tool, ev.Result.IsError))
			case agent.EvtDelta:
				if ev.Delta.Content != "" {
					transcript.WriteString(ev.Delta.Content)
				}
			}
		},
	})

	if res.Err != nil {
		t.Fatalf("live agent RunTurn err: %v\nevents=%v", res.Err, events)
	}
	alpha, errA := os.ReadFile(filepath.Join(ws, "pkgdemo", "alpha.txt"))
	beta, errB := os.ReadFile(filepath.Join(ws, "pkgdemo", "beta.txt"))
	if errA != nil || !strings.Contains(string(alpha), "hello-alpha") {
		t.Fatalf("alpha.txt missing/mismatch (err=%v): %q\nevents=%v\nreply=%s", errA, string(alpha), events, res.FinalContent)
	}
	if errB != nil || !strings.Contains(string(beta), "hello-beta") {
		t.Fatalf("beta.txt missing/mismatch (err=%v): %q\nevents=%v\nreply=%s", errB, string(beta), events, res.FinalContent)
	}

	evLog := fmt.Sprintf("steps=%d prompt_tok=%d cached_tok=%d completion_tok=%d gen_tps=%.1f\nevents=%s\nreply=\n%s\n",
		res.Steps, res.PromptTokens, res.CachedTokens, res.CompletionTokens, res.GenTokPerSec, strings.Join(events, " | "), res.FinalContent)
	_ = os.WriteFile(filepath.Join(evDir, "agent_live_run.txt"), []byte(evLog), 0o644)
	t.Log(evLog)
}
