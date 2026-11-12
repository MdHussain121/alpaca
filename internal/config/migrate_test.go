package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMigrateLegacyUSBLayoutRunsOnce(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALAPPDATA", t.TempDir())

	// Seed a legacy USB layout with llm.exe, llm.toml, presets, skills, sessions, and .llm/.
	for _, dir := range []string{
		filepath.Join(root, "presets"),
		filepath.Join(root, "skills", "demo"),
		filepath.Join(root, "sessions"),
		filepath.Join(root, "workspace", ".llm", "checkpoints", "s1", "turn-0001"),
		filepath.Join(root, ".llm", "skill-history"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "llm.exe"), []byte("legacy"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "llm.toml"), []byte("theme = \"nord\"\nport = 8099\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	presetBody := "name = \"Quick\"\nmodel = \"m.gguf\"\nctx = 8192\n"
	if err := os.WriteFile(filepath.Join(root, "presets", "quick.toml"), []byte(presetBody), 0o644); err != nil {
		t.Fatal(err)
	}
	skillBody := "---\nname: demo\ndescription: demo skill\n---\nbody\n"
	if err := os.WriteFile(filepath.Join(root, "skills", "demo", "SKILL.md"), []byte(skillBody), 0o644); err != nil {
		t.Fatal(err)
	}
	sessionBody := "{\"type\":\"meta\"}\n"
	if err := os.WriteFile(filepath.Join(root, "sessions", "s1.jsonl"), []byte(sessionBody), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "workspace", ".llm", "checkpoints", "s1", "turn-0001", "manifest.json"), []byte(`{"turn":1}`), 0o644); err != nil {
		t.Fatal(err)
	}

	first := MigrateLegacy(root)
	if len(first.Notices) == 0 {
		t.Fatal("expected migration notices on first startup")
	}

	cfg := Load(root)
	if cfg.Theme != "nord" || cfg.Port != 8099 {
		t.Fatalf("legacy config not loaded: %+v", cfg)
	}
	if _, err := os.Stat(filepath.Join(root, "workspace", ".alpaca", "checkpoints", "s1", "turn-0001", "manifest.json")); err != nil {
		t.Fatalf("expected migrated checkpoint in .alpaca: %v", err)
	}

	// Presets, skills, and sessions stay untouched.
	gotPreset, _ := os.ReadFile(filepath.Join(root, "presets", "quick.toml"))
	gotSkill, _ := os.ReadFile(filepath.Join(root, "skills", "demo", "SKILL.md"))
	gotSession, _ := os.ReadFile(filepath.Join(root, "sessions", "s1.jsonl"))
	if string(gotPreset) != presetBody || string(gotSkill) != skillBody || string(gotSession) != sessionBody {
		t.Fatal("presets, skills, or sessions were modified during migration")
	}

	// Second run must not repeat notices.
	second := MigrateLegacy(root)
	if len(second.Notices) != 0 {
		t.Fatalf("expected no repeated notices on second run, got %v", second.Notices)
	}
}

func TestMigrateLegacyPllamaConfigFallback(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALAPPDATA", t.TempDir())

	if err := os.WriteFile(filepath.Join(root, "pllama.toml"), []byte("theme = \"dracula\"\nport = 8123\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res := MigrateLegacy(root)
	if len(res.Notices) == 0 || !strings.Contains(strings.Join(res.Notices, " "), "pllama.toml") {
		t.Fatalf("expected pllama.toml notice, got %v", res.Notices)
	}
	cfg := Load(root)
	if cfg.Theme != "dracula" || cfg.Port != 8123 {
		t.Fatalf("pllama config not loaded: %+v", cfg)
	}
}
