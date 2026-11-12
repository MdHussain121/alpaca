package ui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestBuiltInSystemPromptAndCommands(t *testing.T) {
	m := testUI(t)
	ws := t.TempDir()
	m.agentWorkDir = ws
	m.root = ws
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.settings.Context = 16384
	m.ensureAgentReady()
	m.resizeRunViews()

	// Default effectiveSystemPrompt renders built-in template with runtime values and no raw markers.
	eff := m.effectiveSystemPrompt()
	if strings.Contains(eff, "{{") || strings.Contains(eff, "}}") {
		t.Fatalf("unrendered placeholders in effectiveSystemPrompt:\n%s", eff)
	}
	if !strings.Contains(eff, ws) || !strings.Contains(eff, "read-auto") || !strings.Contains(eff, "- read_file:") {
		t.Fatalf("missing runtime values in built-in system prompt:\n%s", eff)
	}

	// < 8K context selects compact template automatically.
	m.settings.Context = 4096
	compactRes := m.effectivePromptRenderResult(true)
	if !compactRes.Compact {
		t.Fatalf("expected compact prompt when Context=4096")
	}

	// /system show appends info turn without mutating settings.SystemPrompt.
	m.settings.Context = 16384
	m.chatInput.SetValue("/system")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if len(m.chatMsgs) == 0 || !strings.Contains(m.chatMsgs[len(m.chatMsgs)-1].Content, "built-in full") {
		t.Fatalf("expected /system to show current built-in system prompt, got: %+v", m.chatMsgs)
	}

	// /system off disables built-in default and saves config.
	m.chatInput.SetValue("/system off")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if !m.cfg.DisableSystemPrompt {
		t.Fatal("expected DisableSystemPrompt=true after /system off")
	}
	if offEff := m.effectiveSystemPrompt(); strings.Contains(offEff, "You are running inside alpaca") {
		t.Fatalf("built-in prompt should be omitted when off:\n%s", offEff)
	}

	// Custom prompt still works as standalone when built-in is off, or suffix when on.
	m.chatInput.SetValue("/system Reply in haiku")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.settings.SystemPrompt != "Reply in haiku" {
		t.Fatalf("expected custom suffix set, got %q", m.settings.SystemPrompt)
	}
	m.chatInput.SetValue("/system on")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	combined := m.effectiveSystemPrompt()
	if !strings.Contains(combined, "You are running inside alpaca") || !strings.Contains(combined, "Reply in haiku") {
		t.Fatalf("expected built-in + custom suffix, got:\n%s", combined)
	}

	// /system reset restores default on & clears custom suffix.
	m.chatInput.SetValue("/system reset")
	m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if m.cfg.DisableSystemPrompt || m.settings.SystemPrompt != "" {
		t.Fatalf("expected reset state, got disable=%v custom=%q", m.cfg.DisableSystemPrompt, m.settings.SystemPrompt)
	}

	// Disk override at <root>/prompts/system.md takes precedence when present.
	ovrPath := filepath.Join(ws, "prompts", "system.md")
	if err := os.MkdirAll(filepath.Dir(ovrPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ovrPath, []byte("DISK_OVERRIDE {{workspace}}"), 0o644); err != nil {
		t.Fatal(err)
	}
	ovrEff := m.effectiveSystemPrompt()
	if ovrEff != "DISK_OVERRIDE "+ws {
		t.Fatalf("expected disk override, got %q", ovrEff)
	}
}
