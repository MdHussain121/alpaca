package prompts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/runner"
)

func fixtureTools(withWeb, withMCP bool) []runner.ToolDef {
	tools := []runner.ToolDef{
		{Type: "function", Function: runner.ToolFunctionDef{Name: "read_file", Description: "Read a file with line bounds."}},
		{Type: "function", Function: runner.ToolFunctionDef{Name: "list_dir", Description: "List directory contents."}},
		{Type: "function", Function: runner.ToolFunctionDef{Name: "edit_file", Description: "Apply exact single match edits."}},
		{Type: "function", Function: runner.ToolFunctionDef{Name: "run_command", Description: "Run a Windows shell command in workspace."}},
	}
	if withWeb {
		tools = append(tools,
			runner.ToolDef{Type: "function", Function: runner.ToolFunctionDef{Name: "web_search", Description: "Search the public web."}},
			runner.ToolDef{Type: "function", Function: runner.ToolFunctionDef{Name: "web_fetch", Description: "Fetch a web URL."}},
		)
	}
	if withMCP {
		tools = append(tools,
			runner.ToolDef{Type: "function", Function: runner.ToolFunctionDef{Name: "git.status", Description: "Show git status."}},
			runner.ToolDef{Type: "function", Function: runner.ToolFunctionDef{Name: "sqlite.query", Description: "Run read-only SQLite query."}},
		)
	}
	return tools
}

func TestRenderPopulatesPlaceholdersAndOmitsEmptySections(t *testing.T) {
	opts := RenderOptions{
		Workspace:      "demo-workspace",
		PermissionMode: "read-auto",
		Date:           "2026-11-12",
		Tools:          fixtureTools(false, false),
		ContextSize:    16384,
	}
	res := Render(opts)
	if res.Compact || res.Disabled {
		t.Fatalf("unexpected flags: %+v", res)
	}
	if strings.Contains(res.Text, "{{") || strings.Contains(res.Text, "}}") {
		t.Fatalf("unsubstituted placeholders remain:\n%s", res.Text)
	}
	for _, want := range []string{"demo-workspace", "read-auto", "2026-11-12", "16384", "- edit_file:", "- read_file:"} {
		if !strings.Contains(res.Text, want) {
			t.Fatalf("missing %q in:\n%s", want, res.Text)
		}
	}
	for _, unwanted := range []string{"Web content", "MCP tools", "Workspace instructions", "Skills\n"} {
		if strings.Contains(res.Text, unwanted) {
			t.Fatalf("empty section %q should be omitted:\n%s", unwanted, res.Text)
		}
	}

	// Optional sections appear when populated.
	fullOpts := opts
	fullOpts.Tools = fixtureTools(true, true)
	fullOpts.SkillsList = "<available_skills>\n- go-verify: Run go test\n</available_skills>"
	fullOpts.AgentsMD = "Follow repo style."
	fullOpts.CustomPrompt = "Be extra terse."
	full := Render(fullOpts)
	if strings.Contains(full.Text, "{{") {
		t.Fatalf("unsubstituted placeholders remain in full:\n%s", full.Text)
	}
	for _, want := range []string{"Web content", "MCP tools", "git (git.status); sqlite (sqlite.query)", "Workspace instructions", "Follow repo style.", "<available_skills>", "Be extra terse."} {
		if !strings.Contains(full.Text, want) {
			t.Fatalf("missing %q in full render:\n%s", want, full.Text)
		}
	}

	// Deterministic output on repeated calls (KV-cache prefix stability).
	again := Render(fullOpts)
	if again.Text != full.Text {
		t.Fatalf("non-deterministic render output:\n1: %s\n2: %s", full.Text, again.Text)
	}
}

func TestRenderCompactVsFullAndOverrides(t *testing.T) {
	compact := Render(RenderOptions{
		Workspace:      "ws",
		PermissionMode: "ask",
		Date:           "2026-11-12",
		Tools:          fixtureTools(false, false),
		ContextSize:    4096,
	})
	if !compact.Compact {
		t.Fatal("expected Compact=true for 4096 context")
	}
	if strings.Contains(compact.Text, "{{") {
		t.Fatalf("raw marker in compact:\n%s", compact.Text)
	}
	tokEst := (len(compact.Text) + 3) / 4
	if tokEst > 260 {
		t.Fatalf("compact prompt too large: %d est tokens (%d chars)\n%s", tokEst, len(compact.Text), compact.Text)
	}

	// Disk override supersedes embedded default.
	root := t.TempDir()
	ovr := filepath.Join(root, "prompts", "system.md")
	if err := os.MkdirAll(filepath.Dir(ovr), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ovr, []byte("Custom override {{workspace}} ({{leftover_tag}})"), 0o644); err != nil {
		t.Fatal(err)
	}
	resOvr := Render(RenderOptions{
		AppRoot:        root,
		Workspace:      "demo",
		PermissionMode: "read-auto",
		Date:           "2026-11-12",
		ContextSize:    16384,
	})
	if resOvr.OverridePath != ovr {
		t.Fatalf("OverridePath = %q, want %q", resOvr.OverridePath, ovr)
	}
	if resOvr.Text != "Custom override demo ()" || strings.Contains(resOvr.Text, "{{") {
		t.Fatalf("unexpected override render: %q", resOvr.Text)
	}

	// DisableDefault keeps only custom prompt / AGENTS.md / skills.
	disabled := Render(RenderOptions{
		DisableDefault: true,
		CustomPrompt:   "Manual prompt",
		AgentsMD:       "Repo note",
	})
	if !disabled.Disabled || disabled.Text != "Manual prompt\n\nRepo note" {
		t.Fatalf("unexpected disabled output: %+v", disabled)
	}
}
