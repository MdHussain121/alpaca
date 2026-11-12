package ui

import (
	"bytes"
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
	"github.com/charmbracelet/x/exp/golden"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/session"
)

func TestContextBreakdownGoldenAndFeatures(t *testing.T) {
	ws := t.TempDir()
	cfg := config.Load(ws)
	cfg.Theme = "mocha"
	cfg.ASCII = false
	cfg.LongGenBell = true
	cfg.Profiles = map[string]config.SamplingProfile{
		"turbo": {Description: "Custom turbo", Temp: 0.42, TopP: 0.88, TopK: 30, MinP: 0.04},
	}
	_ = cfg.Save()

	if err := os.WriteFile(filepath.Join(ws, "AGENTS.md"), []byte("Rule 1: Always mention PINEAPPLE."), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "small.go"), []byte("package main\nfunc Hello() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var bigBuf strings.Builder
	for i := 1; i <= 260; i++ {
		fmt.Fprintf(&bigBuf, "line %03d: payload data for large file range test\n", i)
	}
	if err := os.WriteFile(filepath.Join(ws, "big.txt"), []byte(bigBuf.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	m := New(ws, cfg)
	m.ready = true
	m.width, m.height = 120, 34
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.agentWorkDir = ws
	m.settings.Context = 8192
	m.resizeRunViews()
	m.ensureAgentReady()

	var mu sync.Mutex
	var seenSysPrompts []string
	var seenTemps []float64
	var seenUserContents []string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions/input_tokens":
			var req runner.ChatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			b, _ := json.Marshal(req)
			_ = json.NewEncoder(w).Encode(map[string]int{"input_tokens": len(b)/4 + 12})
		case "/v1/chat/completions":
			var req struct {
				Messages    []runner.APIMessage `json:"messages"`
				Temperature float64             `json:"temperature"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
				seenSysPrompts = append(seenSysPrompts, req.Messages[0].Content)
			}
			seenTemps = append(seenTemps, req.Temperature)
			for i := len(req.Messages) - 1; i >= 0; i-- {
				if req.Messages[i].Role == "user" {
					seenUserContents = append(seenUserContents, req.Messages[i].Content)
					break
				}
			}
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Ack reply.\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":350,\"completion_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":180}},\"timings\":{\"cache_n\":180,\"predicted_per_second\":55.0}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer srv.Close()
	m.chatBaseURL = srv.URL

	// 1. /profile switch applies sampling settings and status bar badge
	m.chatInput.SetValue("/profile turbo")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.effectiveProfileLabel() != "turbo" || m.settings.Temp != 0.42 {
		t.Fatalf("expected profile turbo (temp 0.42), got %q %.2f", m.effectiveProfileLabel(), m.settings.Temp)
	}
	if !strings.Contains(stripANSI(m.View()), "prof:turbo") {
		t.Fatalf("expected prof:turbo badge in status bar:\n%s", stripANSI(m.View()))
	}

	// 2. Turn 1: AGENTS.md picked up + @small.go attached inline
	m.chatInput.SetValue("Inspect @small.go please")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	pumpUntilModalOrDone(t, m, cmd)

	// Edit AGENTS.md dynamically before Turn 2 -> must reflect on next turn
	if err := os.WriteFile(filepath.Join(ws, "AGENTS.md"), []byte("Rule 2: Always mention MANGO."), 0o644); err != nil {
		t.Fatal(err)
	}

	// 3. @file sandbox escape rejected cleanly
	m.chatInput.SetValue("Read @../../secret.txt")
	_, cmdEsc := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	pumpUntilModalOrDone(t, m, cmdEsc)
	if !strings.Contains(m.chatMsgs[len(m.chatMsgs)-1].Content, "Attachment skipped") {
		t.Fatalf("expected hostile @../../secret.txt skipped safely")
	}

	// 4. Large file @big.txt -> triggers confirmation modal -> [r] range 10-15 -> attaches range
	m.chatInput.SetValue("Check @big.txt")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.attachModal == nil {
		t.Fatal("expected large file attachModal to open for @big.txt")
	}
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'r'}})
	m.attachModal.rangeInput.SetValue("10-15")
	_, cmdRng := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	pumpUntilModalOrDone(t, m, cmdRng)

	mu.Lock()
	if len(seenSysPrompts) < 2 || !strings.Contains(seenSysPrompts[0], "PINEAPPLE") || !strings.Contains(seenSysPrompts[1], "MANGO") {
		t.Fatalf("expected live AGENTS.md pickup across turns: %+v", seenSysPrompts)
	}
	if seenTemps[0] != 0.42 {
		t.Fatalf("expected turbo temperature 0.42 sent to server, got %.2f", seenTemps[0])
	}
	if !strings.Contains(seenUserContents[0], "func Hello()") || !strings.Contains(seenUserContents[1], "line 010:") || strings.Contains(seenUserContents[1], "line 020:") {
		t.Fatalf("expected @small.go inline and @big.txt lines 10-15 only:\n%v", seenUserContents)
	}
	mu.Unlock()

	// 5. Golden snapshot for /context breakdown modal (normalized fixed demo state)
	goldenM := testUI(t)
	goldenM.width, goldenM.height = 120, 34
	goldenM.screen = screenRun
	goldenM.runTab = runTabChat
	goldenM.status = "READY"
	goldenM.agentWorkDir = filepath.Join(t.TempDir(), "demo-ws")
	_ = os.MkdirAll(goldenM.agentWorkDir, 0o755)
	goldenM.settings.Context = 8192
	goldenM.cachedPrefixTokens = 180
	goldenM.attachmentTokens = 42
	goldenM.resizeRunViews()
	goldenM.contextModal = &contextModalState{
		bd: session.ContextBreakdown{
			SystemTokens:      210,
			AgentsMDTokens:    85,
			ToolDefsTokens:    940,
			SkillListTokens:   72,
			AttachmentsTokens: 42,
			HistoryTokens:     420,
			TotalPromptTokens: 1769,
			CachedTokens:      180,
			ContextLimit:      8192,
			LiveMeasured:      true,
		},
	}
	golden.RequireEqual(t, []byte(stripANSI(goldenM.View())))

	// 6. Edit + Regenerate branching & JSONL session load/search
	m.chatInput.SetValue("/edit Updated branched prompt")
	_, cmdEdit := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	pumpUntilModalOrDone(t, m, cmdEdit)
	if m.sessionDoc == nil || len(m.sessionDoc.Meta.Branches) < 2 {
		t.Fatalf("expected branched session after /edit, doc=%+v", m.sessionDoc)
	}

	m.chatInput.SetValue("/regen")
	_, cmdReg := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	pumpUntilModalOrDone(t, m, cmdReg)
	if len(m.sessionDoc.Meta.Branches) < 3 {
		t.Fatalf("expected 3rd branch after /regen, got %d", len(m.sessionDoc.Meta.Branches))
	}

	// Switch back to branch b1 and search/export cleanly
	m.chatInput.SetValue("/session branch b1")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m.chatInput.SetValue("/session export")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	expPath := filepath.Join(m.effectiveSessionsDir(), m.sessionID+".md")
	if _, err := os.Stat(expPath); err != nil {
		t.Fatalf("expected exported markdown file at %s: %v", expPath, err)
	}

	// 7. Pure OSC 52 fallback & >=20s generation terminal bell check
	var oscBuf bytes.Buffer
	oldOSC := osc52Writer
	osc52Writer = &oscBuf
	defer func() { osc52Writer = oldOSC }()
	_ = writeOSC52(osc52Writer, "hello")
	if !strings.HasPrefix(oscBuf.String(), "\x1b]52;c;") {
		t.Fatalf("expected OSC 52 escape sequence, got %q", oscBuf.String())
	}

	var bellBuf bytes.Buffer
	oldBell := bellWriter
	bellWriter = &bellBuf
	defer func() { bellWriter = oldBell }()
	m.chatGen = 99
	m.genStartTime = time.Now().Add(-21 * time.Second)
	m.chatStreaming = true
	m.chatMsgs = append(m.chatMsgs, chatTurn{Role: "assistant"})
	m.handleAgentLoopMsg(agentLoopMsg{gen: 99, kind: agentEvtDone})
	if bellBuf.String() != "\a" {
		t.Fatalf("expected BEL after >=20s generation with LongGenBell=true, got %q", bellBuf.String())
	}
}

func TestLiveCompactionAndRestart(t *testing.T) {
	if os.Getenv("ALPACA_LIVE_COMPACTION") != "1" {
		t.Skip("set ALPACA_LIVE_COMPACTION=1 to run 40-turn live compaction + restart test")
	}

	root := filepath.Clean(filepath.Join("..", ".."))
	exe := filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe")
	modelPath := testModelFile(t, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf")
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("missing llama-server.exe: %v", err)
	}

	evDir := t.TempDir()

	port := runner.NextFreePort(18460)
	proc, err := runner.Start(exe, filepath.Dir(exe), []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"-c", "4096",
		"-ngl", "99",
		"--no-webui",
		"--metrics",
	}, filepath.Join(t.TempDir(), "qwen-compaction.log"))
	if err != nil {
		t.Fatal(err)
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
			t.Fatal("llama-server health timeout")
		case <-time.After(300 * time.Millisecond):
		}
	}

	ws := t.TempDir()
	// Seed a multi-line file so read_file adds real tokens over turns
	var seedBuf strings.Builder
	for i := 1; i <= 120; i++ {
		fmt.Fprintf(&seedBuf, "line %03d: service module configuration entry %03d for compaction test\n", i, i)
	}
	_ = os.WriteFile(filepath.Join(ws, "notes.txt"), []byte(seedBuf.String()), 0o644)

	cfg := config.Load(ws)
	cfg.CompactionThresholdPct = 80
	m := New(ws, cfg)
	m.ready = true
	m.width, m.height = 120, 34
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.chatBaseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	m.agentWorkDir = ws
	m.skillsDirOverride = filepath.Join(ws, "skills")
	m.sessionsDirOv = filepath.Join(ws, "sessions")
	m.settings.Context = 4096
	m.settings.Reasoning = "off" // concise fast turns for 40-turn run
	m.resizeRunViews()
	m.ensureAgentReady()

	// Disable exec/web so Qwen3.5-4B cleanly focuses on local file tools & turns quickly.
	m.chatInput.SetValue("/permissions workspace-auto")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m.chatInput.SetValue("/tools web off")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m.chatInput.SetValue("/tools exec off")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})

	runLiveTurn := func(ui *Model, prompt string) string {
		ui.chatInput.SetValue(prompt)
		_, cmd := ui.Update(tea.KeyMsg{Type: tea.KeyEnter})
		deadline := time.Now().Add(45 * time.Second)
		for cmd != nil {
			if time.Now().After(deadline) {
				t.Fatalf("live turn timed out on prompt %q", prompt)
			}
			msg := cmd()
			_, cmd = ui.Update(msg)
			if ui.permModal != nil {
				_, cmd = ui.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
			}
		}
		if len(ui.chatMsgs) > 0 {
			return strings.TrimSpace(ui.chatMsgs[len(ui.chatMsgs)-1].Content)
		}
		return ""
	}

	// Turn 1: anchor key secret in conversation + write_file tool
	runLiveTurn(m, `Write "vault=AMBER-FOX-913" into vault.txt using write_file, and remember that our project vault code is AMBER-FOX-913. Confirm in one short sentence.`)

	// Turns 2..40: 39 more real turns mixing read_file/list_dir and follow-ups until auto-compaction fires
	compactionEvents := 0
	var compactCardTexts []string
	var maxPreCompactCtx int
	for turn := 2; turn <= 40; turn++ {
		if m.chatCtxUsed > maxPreCompactCtx {
			maxPreCompactCtx = m.chatCtxUsed
		}
		prompt := fmt.Sprintf("Turn %d/40: Read lines 1-40 of notes.txt with read_file and summarize line 1 plus confirm you are still tracking the vault task cleanly in 1 short sentence.", turn)
		if turn%2 == 0 {
			prompt = fmt.Sprintf("Turn %d/40: Note down checkpoint %d briefly in one short sentence while keeping the project vault code AMBER-FOX-913 in mind.", turn, turn)
		}
		prevCompactCount := countCompactionCards(m.chatMsgs)
		_ = runLiveTurn(m, prompt)
		curCompactCount := countCompactionCards(m.chatMsgs)
		if curCompactCount > prevCompactCount {
			compactionEvents += (curCompactCount - prevCompactCount)
			for _, ct := range m.chatMsgs {
				if ct.Role == "compaction" {
					compactCardTexts = append(compactCardTexts, ct.Content+" | "+ct.CompactionDetail)
				}
			}
		}
	}

	if compactionEvents == 0 {
		t.Fatalf("40 turns completed without triggering automatic compaction (final chatCtxUsed=%d / 4096)", m.chatCtxUsed)
	}

	savedSessionID := m.sessionID
	savedSessionsDir := m.effectiveSessionsDir()
	// Simulate alpaca.exe exit
	m.saveCurrentSession()

	// Simulate fresh alpaca.exe startup -> load session JSONL cleanly -> ask Turn 41 for the vault code
	m2 := New(ws, cfg)
	m2.ready = true
	m2.width, m2.height = 120, 34
	m2.screen = screenRun
	m2.runTab = runTabChat
	m2.status = "READY"
	m2.chatBaseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	m2.agentWorkDir = ws
	m2.skillsDirOverride = filepath.Join(ws, "skills")
	m2.sessionsDirOv = savedSessionsDir
	m2.settings.Context = 4096
	m2.settings.Reasoning = "off"
	m2.resizeRunViews()
	m2.ensureAgentReady()

	m2.chatInput.SetValue("/session load " + savedSessionID)
	m2.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m2.sessionID != savedSessionID || len(m2.chatMsgs) < 40 {
		t.Fatalf("expected restored session %s with >=40 UI turns, got id=%s len=%d", savedSessionID, m2.sessionID, len(m2.chatMsgs))
	}

	bdAfter := m2.computeContextBreakdown()
	finalReply := runLiveTurn(m2, "After restarting the app and compacting history, what is the exact project vault code we established in turn 1?")
	if !strings.Contains(finalReply, "AMBER-FOX-913") {
		t.Fatalf("restored compacted session lost vault code AMBER-FOX-913; reply:\n%s", finalReply)
	}

	ev := fmt.Sprintf("Compaction LIVE PASS (Qwen3.5-4B, 40+1 turns)\nSessionID: %s\nMax pre-compact ctx: %d / 4096\nAuto-compaction events: %d\nCompaction cards:\n%s\n\nRestored breakdown after restart:\n%s\n\nTurn 41 post-restart answer:\n%s\n",
		savedSessionID, maxPreCompactCtx, compactionEvents, strings.Join(compactCardTexts, "\n---\n"), formatBreakdownText(bdAfter), finalReply)
	_ = os.WriteFile(filepath.Join(evDir, "compaction_live_run.txt"), []byte(ev), 0o644)
}

func countCompactionCards(turns []chatTurn) int {
	n := 0
	for _, t := range turns {
		if t.Role == "compaction" {
			n++
		}
	}
	return n
}
