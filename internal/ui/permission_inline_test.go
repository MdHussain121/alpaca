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
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func sseToolCall(w http.ResponseWriter, id, name, args string) {
	argsJSON, _ := json.Marshal(args)
	fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":%q,\"type\":\"function\",\"function\":{\"name\":%q,\"arguments\":%s}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", id, name, argsJSON)
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func sseText(w http.ResponseWriter, text string) {
	b, _ := json.Marshal(text)
	fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":%s},\"finish_reason\":\"stop\"}]}\n\n", b)
	fmt.Fprint(w, "data: [DONE]\n\n")
}

// Choosing "always allow" must stop prompts for later calls to the same tool even
// when their arguments differ, otherwise a tool-heavy turn traps the user in modals.
func TestAlwaysAllowStopsRepeatPromptsForDifferentArgs(t *testing.T) {
	ws := t.TempDir()
	step := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		step++
		switch step {
		case 1:
			sseToolCall(w, "c1", "write_file", `{"path":"a.txt","content":"one\n"}`)
		case 2:
			sseToolCall(w, "c2", "write_file", `{"path":"b.txt","content":"two\n"}`)
		default:
			sseText(w, "both files written")
		}
	}))
	defer srv.Close()

	m := testUI(t)
	m.agentWorkDir = ws
	m.checkpoints = agent.NewCheckpointStore(ws, "always-allow")
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 34
	m.chatBaseURL = srv.URL
	m.resizeRunViews()

	m.chatInput.SetValue("write two files")
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	for m.permModal == nil && cmd != nil {
		msg := cmd()
		_, cmd = m.Update(msg)
	}
	if m.permModal == nil {
		t.Fatal("expected the first write_file permission prompt")
	}
	view := stripANSI(m.View())
	if !strings.Contains(view, "Always allow write_file") || !strings.Contains(view, "Don't allow") {
		t.Fatalf("inline prompt missing allow/always/deny choices:\n%s", view)
	}

	// "2" = always allow write_file for this session.
	_, cmd = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'2'}})
	for cmd != nil {
		msg := cmd()
		_, cmd = m.Update(msg)
		if m.permModal != nil {
			t.Fatalf("unexpected second permission prompt after always allow: %+v", m.permModal.prompt)
		}
	}
	if m.chatStreaming {
		t.Fatal("expected the turn to finish")
	}
	for _, name := range []string{"a.txt", "b.txt"} {
		if _, err := os.Stat(filepath.Join(ws, name)); err != nil {
			t.Fatalf("expected %s to be written: %v", name, err)
		}
	}
	entries := m.permEngine.Allowlist()
	if len(entries) != 1 || !entries[0].AllArgs {
		t.Fatalf("expected a tool-wide allowlist entry, got %+v", entries)
	}
}

// Options are shown wrapped and the user can type a free-form answer instead.
func TestAskUserWrapsOptionsAndAcceptsTypedAnswer(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 34
	m.resizeRunViews()

	longOpt := "casual and friendly, with a short opening, no jargon, and a clear closing line at the end"
	rep := make(chan askUserReply, 1)
	_, _ = m.handleAgentLoopMsg(agentLoopMsg{
		gen:      m.chatGen,
		kind:     agentEvtAskReq,
		question: "Which tone should the greeting use?",
		options:  []string{"formal", longOpt},
		askRep:   rep,
		ch:       make(chan agentLoopMsg),
		doneCh:   make(chan agent.LoopResult, 1),
	})
	if m.askModal == nil {
		t.Fatal("expected ask modal")
	}
	flat := strings.Join(strings.Fields(stripANSI(m.View())), " ")
	if !strings.Contains(flat, "closing line at the end") {
		t.Fatalf("long option was truncated instead of wrapped:\n%s", stripANSI(m.View()))
	}
	if !strings.Contains(flat, "Type a different answer") {
		t.Fatalf("missing typed-answer choice:\n%s", stripANSI(m.View()))
	}

	for _, r := range "very casual" {
		m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	select {
	case ans := <-rep:
		if ans.Answer != "very casual" {
			t.Fatalf("expected typed answer, got %q", ans.Answer)
		}
	case <-time.After(time.Second):
		t.Fatal("typed answer was not delivered")
	}
	if m.askModal != nil {
		t.Fatal("expected ask modal to close after answering")
	}
}

func TestPermissionEscCancelsWholeTurn(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 34
	m.resizeRunViews()
	m.chatMsgs = []chatTurn{{Role: "assistant", thinkParser: &runner.ThinkStreamParser{}}}
	m.chatStreaming = true
	_, cancel := context.WithCancel(context.Background())
	m.chatCancel = cancel
	rep := make(chan agent.PermissionResponse, 1)
	m.permModal = &permModalState{
		gen:     m.chatGen,
		replyCh: rep,
		ch:      make(chan agentLoopMsg),
		doneCh:  make(chan agent.LoopResult, 1),
	}

	m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	select {
	case resp := <-rep:
		if resp.Choice != agent.PermDeny {
			t.Fatalf("expected deny on esc, got %v", resp.Choice)
		}
	default:
		t.Fatal("expected a deny response before cancelling")
	}
	if m.permModal != nil {
		t.Fatal("expected permission prompt to clear")
	}
	if m.chatStreaming {
		t.Fatal("expected esc to stop the turn")
	}
	if len(m.chatMsgs) == 0 || !m.chatMsgs[len(m.chatMsgs)-1].Aborted {
		t.Fatal("expected the turn to be marked aborted")
	}
}

func TestAgentTurnPersistsToolCallHistory(t *testing.T) {
	m := testUI(t)
	m.chatMsgs = []chatTurn{{Role: "assistant"}}
	m.chatStreaming = true

	msgs := []runner.APIMessage{
		{Role: "assistant", ToolCalls: []runner.ToolCall{{ID: "c1", Type: "function", Function: runner.ToolFunctionCall{Name: "read_file", Arguments: `{"path":"a.txt"}`}}}},
		{Role: "tool", ToolCallID: "c1", Content: "file body"},
	}
	_, _ = m.handleAgentLoopMsg(agentLoopMsg{gen: m.chatGen, kind: agentEvtDone, res: agent.LoopResult{MessagesAppended: msgs}})
	if got := m.chatMsgs[0].APIMessages; len(got) != 2 || got[1].Role != "tool" || got[1].ToolCallID != "c1" {
		t.Fatalf("tool-call history was not persisted: %+v", got)
	}
}
