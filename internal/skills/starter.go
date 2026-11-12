package skills

import (
	"os"
	"path/filepath"
)

type starterDef struct {
	Name        string
	Description string
	Body        string
}

var starterSkills = []starterDef{
	{
		Name:        "create-skill",
		Description: "How to structure, validate, and stage reusable Agent Skills via save_skill.",
		Body: `# Create Skill Guide

Use this guide when converting a repeatable workflow or domain playbook into a reusable Agent Skill via ` + "`save_skill`" + `.

## Naming & Front Matter Rules
- ` + "`name`" + `: lowercase ASCII letters, digits, and hyphens only (` + "`^[a-z0-9-]{1,40}$`" + `), matching folder ` + "`skills/<name>/`" + `.
- ` + "`description`" + `: concise one-line summary telling the assistant *when* to load the skill.
- Keep instructions clear, step-by-step, and grounded in concrete workspace commands or checks.

## Size & Safety Constraints
- Body must stay under 16 KB of Markdown.
- Up to 10 helper files (` + "`files`" + ` map), each under 16 KB, total bundle <= 64 KB.
- Never embed secrets, API keys, or instructions to bypass permissions.
- ` + "`save_skill`" + ` always stages into ` + "`.alpaca/skills-staging/`" + ` for user preview and approval before saving.`,
	},
	{
		Name:        "scaffold-project",
		Description: "Checklist for scaffolding clean, testable project layouts with starter docs and checks.",
		Body: `# Scaffold Project

Follow this sequence when asked to initialize a new project or module cleanly inside the active workspace:

1. Inspect existing workspace files with ` + "`list_dir`" + ` so you do not overwrite existing work unintentionally.
2. Confirm language/runtime, entrypoints, and test setup if ambiguous using ` + "`ask_user`" + `.
3. Create directories cleanly with ` + "`make_dir`" + ` and write minimal idiomatic entry files plus a small README and smoke test.
4. Verify build/tests via ` + "`run_command`" + ` before finishing and report exact verified paths and command outputs.`,
	},
	{
		Name:        "code-review",
		Description: "Structured diff and code review focusing on correctness, edge cases, security, and tests.",
		Body: `# Code Review

When reviewing changes in the workspace:

1. Gather context using ` + "`glob`" + `, ` + "`grep`" + `, and ` + "`read_file`" + ` (or ` + "`git diff`" + ` via ` + "`run_command`" + ` if needed).
2. Check correctness first: nil/boundary checks, resource leaks, error propagation, concurrency hazards, and path/SSRF boundaries.
3. Verify test coverage exercises failure paths as well as happy paths.
4. Present findings grouped by severity (` + "`Critical`" + `, ` + "`Warning`" + `, ` + "`Suggestion`" + `) citing exact file paths and line ranges.`,
	},
}

// EnsureStarterSkills seeds the 3 starter skills once into skillsDir unless already seeded
// (tracked via <dataDir>/starter-skills-seeded so deleted starter skills stay deleted).
func EnsureStarterSkills(skillsDir, dataDir string) error {
	if skillsDir == "" {
		return nil
	}
	marker := ""
	if dataDir != "" {
		marker = filepath.Join(dataDir, "starter-skills-seeded")
		if _, err := os.Stat(marker); err == nil {
			return nil
		}
	}
	if err := os.MkdirAll(skillsDir, 0o755); err != nil {
		return err
	}
	for _, s := range starterSkills {
		dir := filepath.Join(skillsDir, s.Name)
		if _, err := os.Stat(dir); err == nil {
			continue
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		md := RenderSkillMarkdown(s.Name, s.Description, s.Body, false)
		if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(md), 0o644); err != nil {
			return err
		}
	}
	if marker != "" {
		_ = os.MkdirAll(dataDir, 0o755)
		_ = os.WriteFile(marker, []byte("seeded\n"), 0o644)
	}
	return nil
}
