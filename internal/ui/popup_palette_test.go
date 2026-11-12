package ui

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/exp/teatest"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/runner"
)

type termCatcher struct {
	buf bytes.Buffer
	tm  *teatest.TestModel
}

func (tc *termCatcher) wait(t *testing.T, want string) {
	t.Helper()
	r := tc.tm.Output()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		chunk := make([]byte, 4096)
		for {
			n, _ := r.Read(chunk)
			if n == 0 {
				break
			}
			tc.buf.Write(chunk[:n])
		}
		if strings.Contains(stripANSI(tc.buf.String()), want) {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for %q\nCaptured output (%d bytes):\n%s", want, tc.buf.Len(), stripANSI(tc.buf.String()))
}

func TestTeatestSlashPopupFlowAndPalette(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.chatInput.Focus()

	// Real workspace files for @ completion inside m.root.
	if err := os.WriteFile(filepath.Join(m.root, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(m.root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(m.root, "docs", "notes.md"), []byte("# Notes\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 32))
	tc := &termCatcher{tm: tm}

	// 1. Open popup by typing '/', verify prefix filter '/th', move ↓/↑, Tab complete -> '/theme ', pick 'dracula' + Enter.
	tm.Type("/")
	tc.wait(t, "Commands")
	tm.Type("th")
	tc.wait(t, "/theme")
	tc.wait(t, "/think")

	// Move down to second item (/think) and back up to first item (/theme), then Tab to complete '/theme '.
	tm.Send(tea.KeyMsg{Type: tea.KeyDown})
	tm.Send(tea.KeyMsg{Type: tea.KeyUp})
	tm.Send(tea.KeyMsg{Type: tea.KeyTab})
	tc.wait(t, "tokyo-night")
	tc.wait(t, "dracula")

	// Filter theme arg 'drac' and run with Enter -> theme changes to dracula.
	tm.Type("drac")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})
	tc.wait(t, "theme: dracula")

	// 2. Esc closes popup WITHOUT clearing the input.
	tm.Type("/he")
	tc.wait(t, "/help")
	tm.Send(tea.KeyMsg{Type: tea.KeyEsc})
	// Now backspace twice + type 'them' -> unknown command should show closest suggestion (/theme).
	tm.Send(tea.KeyMsg{Type: tea.KeyBackspace})
	tm.Send(tea.KeyMsg{Type: tea.KeyBackspace})
	tm.Send(tea.KeyMsg{Type: tea.KeyBackspace})
	tm.Type("/them")
	tm.Send(tea.KeyMsg{Type: tea.KeyEsc}) // dismiss popup so Enter submits raw /them
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})
	tc.wait(t, "Did you mean /theme")

	// 3. @ file completion inserts '@docs/notes.md ' without submitting prompt.
	tm.Type("check @not")
	tc.wait(t, "docs/notes.md")
	tm.Send(tea.KeyMsg{Type: tea.KeyTab})
	// Wait briefly by opening command palette ctrl+p and running /quit!
	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlP})
	tc.wait(t, "Command Palette")
	tm.Type("quit")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
	finalM, ok := tm.FinalModel(t).(*Model)
	if !ok {
		t.Fatalf("expected *Model final model")
	}
	if finalM.sty.t.Name != "dracula" {
		t.Fatalf("expected theme switched to dracula, got %q", finalM.sty.t.Name)
	}
	if !strings.Contains(finalM.chatInput.Value(), "@docs/notes.md") {
		t.Fatalf("expected @file completion in input before quit, got %q", finalM.chatInput.Value())
	}
}

func TestPopupArgSuggestionsModelPresetCopyThinkAndDisabled(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 32
	m.resizeRunViews()

	// Seed two discovered models and a saved preset.
	m.models = []modelEntry{
		{
			cand: gguf.Candidate{RelPath: "Qwen3.5-4B.gguf", Size: 2 << 30},
			meta: &gguf.Metadata{Path: filepath.Join(m.root, "Qwen3.5-4B.gguf"), Name: "Qwen3.5-4B", Quant: "Q4_K_M", ContextLength: 8192},
		},
		{
			cand: gguf.Candidate{RelPath: "LFM2.5-8B.gguf", Size: 4 << 30},
			meta: &gguf.Metadata{Path: filepath.Join(m.root, "LFM2.5-8B.gguf"), Name: "LFM2.5-8B", Quant: "Q4_0", ContextLength: 8192},
		},
	}
	m.visible = []int{0, 1}
	m.selectCurrent()

	p := config.Preset{
		Name:     "FastCoder",
		Model:    "Qwen3.5-4B.gguf",
		Settings: flags.Default(8, 8080),
	}
	p.Settings.Context = 16384
	if _, err := m.cfg.SavePreset(p); err != nil {
		t.Fatal(err)
	}
	m.presets = nil

	// Disabled /stop and /copy when idle with no messages.
	m.chatInput.Focus()
	m.chatInput.SetValue("/")
	m.syncChatPopup()
	if !m.popup.open {
		t.Fatalf("expected slash popup open on '/'")
	}
	// Verify scrolling across >8 items reaches end cleanly.
	total := len(m.popup.items)
	if total <= popupMaxVisible {
		t.Fatalf("expected >%d registered commands to exercise scroll, got %d", popupMaxVisible, total)
	}
	var sawStopDisabled, sawCopyDisabled bool
	for _, it := range m.popup.items {
		if it.ID == "stop" && it.Disabled && it.DisabledReason != "" {
			sawStopDisabled = true
		}
		if it.ID == "copy" && it.Disabled && it.DisabledReason != "" {
			sawCopyDisabled = true
		}
	}
	if !sawStopDisabled || !sawCopyDisabled {
		t.Fatalf("expected disabled reason on /stop (%v) and /copy (%v)", sawStopDisabled, sawCopyDisabled)
	}
	// Move down past row 8 so popup.scroll advances!
	for i := 0; i < 10; i++ {
		m.Update(tea.KeyMsg{Type: tea.KeyDown})
	}
	if m.popup.scroll == 0 {
		t.Fatalf("expected popup scroll > 0 after moving past 8 rows (idx=%d)", m.popup.idx)
	}

	// /model arg completion & switching to LFM2.5-8B
	m.chatInput.SetValue("/model lfm")
	m.syncChatPopup()
	if !m.popup.open || m.popup.kind != popupSlashArg || len(m.popup.items) != 1 || m.popup.items[0].ID != "LFM2.5-8B.gguf" {
		t.Fatalf("expected /model lfm to suggest LFM2.5-8B.gguf, got open=%v kind=%v items=%+v", m.popup.open, m.popup.kind, m.popup.items)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.meta == nil || m.meta.DisplayName() != "LFM2.5-8B" {
		t.Fatalf("expected /model enter to switch active model to LFM2.5-8B, got %v", m.meta)
	}

	// /preset arg completion & apply
	m.chatInput.SetValue("/preset fast")
	m.syncChatPopup()
	if !m.popup.open || len(m.popup.items) == 0 || m.popup.items[0].ID != "FastCoder" {
		t.Fatalf("expected /preset fast to suggest FastCoder, got %+v", m.popup.items)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.settings.Context != 16384 {
		t.Fatalf("expected preset context 16384 applied, got %d", m.settings.Context)
	}

	// Prefix matches ranked ahead of fuzzy subsequence matches.
	ranked := rankPopupItems("th", []popupItem{
		{ID: "paths", Label: "paths"},
		{ID: "theme", Label: "theme"},
		{ID: "think", Label: "think"},
	})
	if len(ranked) < 3 || ranked[0].Label != "theme" || ranked[1].Label != "think" || ranked[2].Label != "paths" {
		t.Fatalf("expected prefix matches first then fuzzy, got %+v", ranked)
	}

	// Streaming no-flicker/no-layout-jump check: chatView.Height must be identical with popup closed vs open.
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "line 1\nline 2\nline 3"},
	}
	m.chatStreaming = true
	m.refreshChatView(true)
	closedView := m.View()
	closedChatH := m.chatView.Height

	m.chatInput.SetValue("/th")
	m.syncChatPopup()
	openView := m.View()
	openChatH := m.chatView.Height
	if closedChatH != openChatH {
		t.Fatalf("popup changed chatView.Height (%d -> %d), violating no-layout-jump rule", closedChatH, openChatH)
	}
	if len(strings.Split(closedView, "\n")) != len(strings.Split(openView, "\n")) {
		t.Fatalf("terminal height jumped when popup opened")
	}
	// Streaming delta arriving while popup is open must NOT close or steal popup.
	m.handleChatDelta(runner.ChatDelta{Content: "\nline 4"})
	if !m.popup.open {
		t.Fatalf("streaming delta closed active slash popup")
	}
}

func TestPopupAndPaletteGoldenSnapshots(t *testing.T) {
	update := os.Getenv("UPDATE_GOLDEN") == "1"

	for _, sz := range []struct {
		w, h int
		tag  string
	}{
		{80, 24, "80x24"},
		{120, 32, "120x32"},
		{160, 40, "160x40"},
	} {
		t.Run("popup-"+sz.tag, func(t *testing.T) {
			m := testUI(t)
			m.agentWorkDir = "demo-ws"
			m.screen = screenRun
			m.runTab = runTabChat
			m.status = "READY"
			m.width, m.height = sz.w, sz.h
			m.resizeRunViews()
			m.chatMsgs = []chatTurn{
				{Role: "user", Content: "Explain slash commands"},
				{Role: "assistant", Content: "Type `/` to open command suggestions or `ctrl+p` for the palette."},
			}
			m.refreshChatView(false)
			m.chatInput.Focus()
			m.chatInput.SetValue("/th")
			m.syncChatPopup()

			got := strings.TrimSpace(stripANSI(m.View()))
			assertNamedGolden(t, "popup-"+sz.tag+".golden", got, update)
		})

		t.Run("palette-"+sz.tag, func(t *testing.T) {
			m := testUI(t)
			m.agentWorkDir = "demo-ws"
			m.screen = screenRun
			m.runTab = runTabChat
			m.status = "READY"
			m.width, m.height = sz.w, sz.h
			m.resizeRunViews()
			m.openCommandPalette()
			m.palette.input.SetValue("th")
			m.syncPaletteItems()

			got := strings.TrimSpace(stripANSI(m.View()))
			assertNamedGolden(t, "palette-"+sz.tag+".golden", got, update)
		})
	}
}

func assertNamedGolden(t *testing.T, name, got string, update bool) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if update {
		if err := os.WriteFile(path, []byte(got+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	wantB, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("missing %s (run UPDATE_GOLDEN=1): %v", path, err)
	}
	want := strings.TrimSpace(string(bytes.ReplaceAll(wantB, []byte("\r\n"), []byte("\n"))))
	if got != want {
		t.Fatalf("golden mismatch %s:\n--- GOT ---\n%s\n--- WANT ---\n%s", name, got, want)
	}
}
