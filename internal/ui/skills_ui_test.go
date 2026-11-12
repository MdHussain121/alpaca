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

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestSkillsPanelAndStagingApprovalUseFlow(t *testing.T) {
	ws := t.TempDir()
	cfg := config.Load(ws)
	m := New(ws, cfg)
	m.ready = true
	m.width, m.height = 120, 34
	m.agentWorkDir = ws
	m.ensureAgentReady()

	// Add an invalid skill folder alongside the 3 starter skills
	badDir := filepath.Join(m.effectiveSkillsDir(), "Bad_Name")
	if err := os.MkdirAll(badDir, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(badDir, "SKILL.md"), []byte("---\nname: Bad_Name\ndescription: nope\n---\nbody"), 0o644)
	m.reloadSkills()

	if len(m.skillsCatalog.Valid) != 3 || len(m.skillsCatalog.Invalid) != 1 {
		t.Fatalf("expected 3 starter valid + 1 invalid skill, got valid=%d invalid=%d",
			len(m.skillsCatalog.Valid), len(m.skillsCatalog.Invalid))
	}

	// Open /skills panel
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.resizeRunViews()
	m.chatInput.SetValue("/skills")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.skillsModal == nil {
		t.Fatal("expected /skills modal to open")
	}
	v := stripANSI(m.View())
	if !strings.Contains(v, "create-skill") || !strings.Contains(v, "Bad_Name") || !strings.Contains(v, "ERR") {
		t.Fatalf("expected starter skills and ERR invalid skill in /skills modal:\n%s", v)
	}

	// Toggle first valid skill off & back on via Space
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{' '}})
	if m.skillsCatalog.Valid[0].Enabled {
		t.Fatal("expected first skill disabled after Space toggle")
	}
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{' '}})
	if !m.skillsCatalog.Valid[0].Enabled {
		t.Fatal("expected first skill re-enabled after second Space toggle")
	}

	// View details with Enter then Esc
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.skillsModal.mode != skillsModeView {
		t.Fatalf("expected skillsModeView, got %v", m.skillsModal.mode)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyEsc})

	// Delete invalid Bad_Name to trash (move cursor to Bad_Name row #3)
	m.skillsModal.cursor = len(m.skillsCatalog.Valid) // first invalid row
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	if len(m.skillsCatalog.Invalid) != 0 {
		t.Fatalf("expected invalid skill moved to trash, still have %d", len(m.skillsCatalog.Invalid))
	}

	// Create + Rename via modal
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	m.skillsModal.nameIn.SetValue("temp-manual")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m.skillsModal.descIn.SetValue("Temp manual checklist")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m.skillsModal.bodyIn.SetValue("# Temp Manual\nDo step one.")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if _, ok := m.skillsCatalog.Lookup("temp-manual", true); !ok {
		t.Fatal("expected temp-manual created via /skills modal")
	}

	// Find temp-manual cursor, rename to my-manual
	for i, s := range m.skillsCatalog.Valid {
		if s.Name == "temp-manual" {
			m.skillsModal.cursor = i
			break
		}
	}
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'r'}})
	m.skillsModal.nameIn.SetValue("my-manual")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if _, ok := m.skillsCatalog.Lookup("my-manual", true); !ok {
		t.Fatal("expected temp-manual renamed to my-manual")
	}

	// Close /skills modal
	m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if m.skillsModal != nil {
		t.Fatal("expected /skills modal closed")
	}

	// Scripted LLM server: reject v1, approve v2, then new chat invoke use_skill
	var mu sync.Mutex
	var sysPrompts []string
	var lastReqMsgs []runner.APIMessage
	step := 0

	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []runner.APIMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		if len(body.Messages) > 0 && body.Messages[0].Role == "system" {
			sysPrompts = append(sysPrompts, body.Messages[0].Content)
		}
		lastReqMsgs = body.Messages
		curStep := step
		step++
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		sendCall := func(id, name, args string) {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":%q,\"type\":\"function\",\"function\":{\"name\":%q,\"arguments\":%q}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", id, name, args)
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
		switch curStep {
		case 0:
			args, _ := json.Marshal(map[string]any{
				"name":        "go-verify",
				"description": "Verify Go modules cleanly.",
				"body":        "# Go Verify v1\nRun go test ./...",
			})
			sendCall("s1", "save_skill", string(args))
		case 1:
			args, _ := json.Marshal(map[string]any{
				"name":        "go-verify",
				"description": "Verify Go modules cleanly.",
				"body":        "# Go Verify v2\nRun `go vet ./...` then `go test -count=1 ./...`.",
			})
			sendCall("s2", "save_skill", string(args))
		case 2:
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Saved go-verify!\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		case 3:
			sendCall("u1", "use_skill", `{"name":"go-verify"}`)
		default:
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"VERIFIED_NEW_CHAT_SKILL\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer llmSrv.Close()
	m.chatBaseURL = llmSrv.URL

	// Turn 1: Ask model to create skill
	m.chatInput.SetValue("Turn our Go verification steps into a skill.")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	if m.skillApproveModal == nil || m.skillApproveModal.prompt.Name != "go-verify" {
		t.Fatalf("expected skillApproveModal for go-verify, got %+v", m.skillApproveModal)
	}
	pv := stripANSI(m.View())
	if !strings.Contains(pv, "model-authored") || !strings.Contains(pv, "Go Verify v1") {
		t.Fatalf("expected model-authored badge & markdown preview in staging modal:\n%s", pv)
	}

	// Reject with reason [d] -> enter note -> enter
	_, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}})
	for _, ch := range "add go vet" {
		_, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{ch}})
	}
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	cmd = pumpUntilModalOrDone(t, m, cmd)

	// Model received rejection and retried v2 -> modal pops again!
	if m.skillApproveModal == nil || !strings.Contains(m.skillApproveModal.prompt.Body, "Go Verify v2") {
		t.Fatalf("expected retried v2 staging modal, got %+v", m.skillApproveModal)
	}
	// Approve with 'y'
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	pumpUntilModalOrDone(t, m, cmd)

	skSaved, ok := m.skillsCatalog.Lookup("go-verify", false)
	if !ok || !skSaved.ModelAuthored || !strings.Contains(skSaved.Body, "go test -count=1") {
		t.Fatalf("expected approved model-authored go-verify skill in catalog, got ok=%v sk=%+v", ok, skSaved)
	}

	// New chat: /clear, verify <available_skills> advertises go-verify (summary only), invoke /skill go-verify
	m.chatInput.SetValue("/clear")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})

	m.chatInput.SetValue("/skill go-verify check repository")
	_, cmd2 := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	pumpUntilModalOrDone(t, m, cmd2)

	mu.Lock()
	defer mu.Unlock()
	lastSys := sysPrompts[len(sysPrompts)-1]
	if !strings.Contains(lastSys, "go-verify: Verify Go modules cleanly.") {
		t.Fatalf("expected go-verify summary in new chat system prompt:\n%s", lastSys)
	}
	if strings.Contains(lastSys, "go test -count=1") {
		t.Fatalf("system prompt must only contain progressive summary, NOT full body")
	}
	var sawUseToolResult bool
	for _, msg := range lastReqMsgs {
		if msg.Role == "tool" && strings.Contains(msg.Content, "go test -count=1") {
			sawUseToolResult = true
		}
	}
	if !sawUseToolResult {
		t.Fatal("expected use_skill tool result loaded into new chat conversation")
	}

	m.reloadSkills()
	skAfter, _ := m.skillsCatalog.Lookup("go-verify", false)
	if skAfter.UseCount != 1 {
		t.Fatalf("expected UseCount=1 after new chat use_skill, got %d", skAfter.UseCount)
	}
}

func TestLiveSkillsLifecycle(t *testing.T) {
	if os.Getenv("ALPACA_LIVE_SKILLS") != "1" {
		t.Skip("set ALPACA_LIVE_SKILLS=1 to run live model skill create -> preview -> approve -> new chat use test")
	}

	root := filepath.Clean(filepath.Join("..", ".."))
	exe := filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe")
	modelPath := testModelFile(t, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf")
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("missing llama-server.exe: %v", err)
	}

	evDir := t.TempDir()

	port := runner.NextFreePort(18360)
	proc, err := runner.Start(exe, filepath.Dir(exe), []string{
		"-m", modelPath,
		"--port", fmt.Sprintf("%d", port),
		"-c", "8192",
		"-ngl", "99",
		"--no-webui",
		"--metrics",
	}, filepath.Join(t.TempDir(), "qwen-skills.log"))
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

	ws := t.TempDir()
	cfg := config.Load(ws)
	m := New(ws, cfg)
	m.ready = true
	m.width, m.height = 120, 36
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.chatBaseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	m.agentWorkDir = ws
	m.skillsDirOverride = filepath.Join(ws, "skills")
	m.resizeRunViews()
	m.ensureAgentReady()

	// Turn 1: ask Qwen3.5-4B to create a skill via save_skill -> TUI preview appears -> user approves with 'y'
	prompt1 := `Turn this procedure into a reusable Agent Skill by calling save_skill immediately:
- name: "git-status-check"
- description: "Pre-commit verification ensuring git working tree is clean."
- body: Step 1: Run git status --short. Step 2: Confirm no untracked temp files exist before tagging.`
	m.chatInput.SetValue(prompt1)
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})

	approved := false
	deadline := time.Now().Add(75 * time.Second)
	var modalView string
	for cmd != nil {
		if time.Now().After(deadline) {
			t.Fatalf("live skills turn 1 timed out (approved=%v modal=%v)", approved, m.skillApproveModal != nil)
		}
		msg := cmd()
		_, cmd = m.Update(msg)
		if m.skillApproveModal != nil {
			modalView = stripANSI(m.View())
			approved = true
			_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
		} else if m.permModal != nil {
			_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
		}
	}

	if !approved {
		t.Fatal("model did not trigger save_skill approval modal")
	}
	if !strings.Contains(modalView, "model-authored") || !strings.Contains(modalView, "git-status-check") {
		t.Fatalf("unexpected skill preview modal rendering:\n%s", modalView)
	}

	sk, ok := m.skillsCatalog.Lookup("git-status-check", false)
	if !ok || !sk.ModelAuthored {
		t.Fatalf("expected git-status-check saved & marked model-authored, got ok=%v sk=%+v", ok, sk)
	}
	rawMD, err := os.ReadFile(sk.SkillFile)
	if err != nil {
		t.Fatal(err)
	}

	// Turn 2 (new chat via /clear): ask model to use the newly saved skill
	m.chatInput.SetValue("/clear")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})

	m.chatInput.SetValue(`Use the git-status-check skill (call use_skill with name "git-status-check") and state the exact two steps it defines.`)
	_, cmd2 := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	deadline2 := time.Now().Add(60 * time.Second)
	for cmd2 != nil {
		if time.Now().After(deadline2) {
			t.Fatal("live skills turn 2 timed out")
		}
		msg := cmd2()
		_, cmd2 = m.Update(msg)
		if m.permModal != nil {
			_, cmd2 = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
		}
	}

	m.reloadSkills()
	skAfter, _ := m.skillsCatalog.Lookup("git-status-check", false)
	if skAfter.UseCount < 1 {
		t.Fatalf("expected git-status-check UseCount >= 1 in new chat, got %d", skAfter.UseCount)
	}

	lastTurn := m.chatMsgs[len(m.chatMsgs)-1]
	reply := strings.TrimSpace(lastTurn.Content)
	if !strings.Contains(strings.ToLower(reply), "git status") {
		t.Fatalf("expected reply referencing loaded skill's git status step, got:\n%s", reply)
	}

	evidence := fmt.Sprintf("Skills LIVE PASS (Qwen3.5-4B)\nApproved modal preview:\n%s\n\nSaved SKILL.md:\n%s\nNew chat UseCount=%d\nNew chat reply:\n%s\n",
		modalView, string(rawMD), skAfter.UseCount, reply)
	_ = os.WriteFile(filepath.Join(evDir, "skills_live_run.txt"), []byte(evidence), 0o644)
}
