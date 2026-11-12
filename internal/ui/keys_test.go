package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func press(m *Model, s string) tea.Cmd {
	var msg tea.KeyMsg
	switch s {
	case "enter":
		msg = tea.KeyMsg{Type: tea.KeyEnter}
	case "tab":
		msg = tea.KeyMsg{Type: tea.KeyTab}
	case "esc":
		msg = tea.KeyMsg{Type: tea.KeyEsc}
	case "up":
		msg = tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		msg = tea.KeyMsg{Type: tea.KeyDown}
	case "left":
		msg = tea.KeyMsg{Type: tea.KeyLeft}
	case "right":
		msg = tea.KeyMsg{Type: tea.KeyRight}
	case "backspace":
		msg = tea.KeyMsg{Type: tea.KeyBackspace}
	default:
		msg = tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
	_, cmd := m.Update(msg)
	return cmd
}

func TestFocusMovesBetweenPanels(t *testing.T) {
	m := testUI(t)
	if m.focus != 0 {
		t.Fatalf("focus = %d", m.focus)
	}
	press(m, "tab")
	if m.focus != 1 {
		t.Fatalf("tab did not focus settings: %d", m.focus)
	}
	press(m, "tab")
	if m.focus != 0 {
		t.Fatalf("tab did not wrap: %d", m.focus)
	}
}

// The cursor indexes visible rows, so a dense model must not have an off-by-one
// from the hidden MoE row.
func TestFieldCursorMatchesVisibleRowsOnDenseModel(t *testing.T) {
	m := testUI(t)
	m.focus = 1

	var lan int
	for i, f := range m.visibleFields() {
		if f.label == "Expose on LAN" {
			lan = i
		}
	}
	m.fieldIdx = lan
	for i := 0; i < lan; i++ {
		press(m, "up")
	}
	if m.fieldIdx != 0 {
		t.Fatalf("fieldIdx = %d after walking up", m.fieldIdx)
	}
	press(m, "down")
	if m.fieldIdx != 1 {
		t.Fatalf("fieldIdx = %d after one down", m.fieldIdx)
	}
	if got := m.visibleFields()[1].label; got != "K cache" {
		t.Fatalf("row 1 = %q, want K cache (MoE row must not be counted)", got)
	}
}

func TestFilterTypingAndEscape(t *testing.T) {
	m := testUI(t)
	press(m, "/")
	if !m.filtering {
		t.Fatal("filter mode not entered")
	}
	press(m, "z")
	press(m, "z")
	if m.filter != "zz" {
		t.Fatalf("filter = %q", m.filter)
	}
	if len(m.visible) != 0 {
		t.Fatalf("visible = %d, want no match", len(m.visible))
	}
	press(m, "backspace")
	if m.filter != "z" {
		t.Fatalf("filter after backspace = %q", m.filter)
	}
	press(m, "esc")
	if m.filtering || m.filter != "" {
		t.Fatalf("esc did not clear the filter: %q", m.filter)
	}
	if len(m.visible) != 1 {
		t.Fatalf("visible = %d after clearing", len(m.visible))
	}
}

func TestHelpOverlayOpensAndCloses(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 100, 30
	press(m, "?")
	if !m.showHelp {
		t.Fatal("help did not open")
	}
	if !strings.Contains(m.View(), "keys") {
		t.Fatal("help view missing")
	}
	press(m, "?")
	if m.showHelp {
		t.Fatal("help did not close")
	}
}

func TestThemeTogglePersistsChoice(t *testing.T) {
	m := testUI(t)
	names := ThemeNames()
	if len(names) < 10 {
		t.Fatalf("expected expanded themes, got %d (%v)", len(names), names)
	}
	for _, want := range names[1:] {
		cmd := press(m, "t")
		if m.cfg.Theme != want {
			t.Fatalf("theme = %q, want %q", m.cfg.Theme, want)
		}
		if cmd == nil {
			t.Fatalf("theme toggle to %q did not dispatch save command", want)
		}
		cmd()
	}
	loaded := config.Load(m.root)
	if loaded.Theme != names[len(names)-1] {
		t.Fatalf("persisted theme = %q, want %q", loaded.Theme, names[len(names)-1])
	}
	press(m, "t")
	if m.cfg.Theme != names[0] {
		t.Fatalf("wrap-around theme = %q, want %q", m.cfg.Theme, names[0])
	}
}

func TestStopDispatchesStoppedMessage(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.status = "READY"
	m.proc = &runner.Process{}
	m.meta = testMeta(1, 2)

	cmd := press(m, "x")
	if m.status != "STOPPING" {
		t.Fatalf("status = %q", m.status)
	}
	if cmd == nil {
		t.Fatal("stop did not dispatch a command")
	}
}

func TestPresetOverlayOpensAndCloses(t *testing.T) {
	m := testUI(t)
	if pm, ok := m.saveCurrentPreset()().(presetMsg); ok && pm.err != nil {
		t.Fatalf("save: %v", pm.err)
	}
	m.presets, _ = m.cfg.ListPresets()
	m.width, m.height = 100, 30

	press(m, "p")
	if !m.showPreset {
		t.Fatal("preset overlay did not open")
	}
	if !strings.Contains(m.View(), "ctx") {
		t.Fatal("preset row missing from the overlay")
	}
	press(m, "esc")
	if m.showPreset {
		t.Fatal("preset overlay did not close")
	}
}

func TestEditingExtraArgsUpdatesTheCommand(t *testing.T) {
	m := testUI(t)
	m.focus = 1

	var idx int
	for i, f := range m.visibleFields() {
		if f.label == "Extra args" {
			idx = i
		}
	}
	m.fieldIdx = idx
	press(m, "enter")
	if !m.editing {
		t.Fatal("extra args field did not open the editor")
	}
	m.input.SetValue("--cache-reuse 256")
	press(m, "enter")
	if m.editing {
		t.Fatal("editor did not close")
	}
	if m.settings.ExtraArgs != "--cache-reuse 256" {
		t.Fatalf("extra args = %q", m.settings.ExtraArgs)
	}
	if !strings.Contains(m.cmdPreview, "--cache-reuse 256") {
		t.Fatalf("preview not updated: %s", m.cmdPreview)
	}
}

func TestRunScreenRendersStatusAndHelp(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 120, 32
	m.screen = screenRun
	m.status = "READY"

	view := m.View()
	if !strings.Contains(view, "READY") {
		t.Fatal("status badge missing from the run screen")
	}
	for _, key := range []string{"stop", "restart", "back"} {
		if !strings.Contains(view, key) {
			t.Fatalf("help bar missing %q", key)
		}
	}
	if !strings.Contains(view, "llama-server log") {
		t.Fatal("log panel missing")
	}
}

func TestModelSwitchRecallsLastUsedSettings(t *testing.T) {
	m := testUI(t)
	m.settings.Context = 32768
	m.cfg.Remember("demo.gguf", m.settings)

	m.settings = flags.Default(4, 8080)
	m.selectCurrent()
	if m.settings.Context != 32768 {
		t.Fatalf("context = %d, want the remembered 32768", m.settings.Context)
	}
}

func TestScanMessagePopulatesModelList(t *testing.T) {
	m := testUI(t)
	m.models = nil
	m.applyFilter()

	m.Update(scanMsg{entries: []modelEntry{
		{cand: gguf.Candidate{RelPath: "b.gguf", Path: "b.gguf"}},
		{cand: gguf.Candidate{RelPath: "a.gguf", Path: "a.gguf"}},
	}})

	if len(m.models) != 2 || len(m.visible) != 2 {
		t.Fatalf("models = %d, visible = %d", len(m.models), len(m.visible))
	}
	if m.scanning {
		t.Fatal("scan still flagged as running")
	}
}

func TestQuitConfirmationStopsTheServer(t *testing.T) {
	m := testUI(t)
	m.status = "READY"
	m.proc = &runner.Process{}

	press(m, "q")
	if !m.quitConfirm {
		t.Fatal("quit while running must ask first")
	}
	press(m, "n")
	if m.quit || m.quitConfirm {
		t.Fatal("n must cancel the quit")
	}
	press(m, "q")
	press(m, "y")
	if !m.quit {
		t.Fatal("y must quit")
	}
}
