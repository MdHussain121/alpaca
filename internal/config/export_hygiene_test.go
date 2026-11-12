package config

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestExportExclusionRulesInCI(t *testing.T) {
	strict := os.Getenv("GITHUB_ACTIONS") == "true" || os.Getenv("ALPACA_STRICT_PUBLIC") == "1"

	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}

	// Ensure export helper files themselves exist in repository.
	for _, req := range []string{
		"LICENSE",
		"README.md",
		"THIRD_PARTY_NOTICES.md",
		filepath.Join("scripts", "check-clean.ps1"),
		filepath.Join("scripts", "release.ps1"),
	} {
		if _, err := os.Stat(filepath.Join(root, req)); err != nil {
			t.Fatalf("missing required repository file %s: %v", req, err)
		}
	}

	if !strict {
		return
	}

	forbiddenFiles := map[string]bool{
		"AGENTS.md":                true,
		"CLAUDE.md":                true,
		"DECISIONS.md":             true,
		"PERF.md":                  true,
		"IMPLEMENTATION_STATUS.md": true,
		"TO" + "DO.md":             true,
		".gguf-cache.json":         true,
		"config.toml":              true,
		"mcp.toml":                 true,
	}
	forbiddenDirs := map[string]bool{
		"perf":             true,
		"evidence":         true,
		"release-evidence": true,
		"staging":          true,
		"dist":             true,
	}
	piiPathRe := regexp.MustCompile(`(?i)` + `C:\\` + `Users\\|` + `C:/` + `Users/|@gmail\.com`)

	err = filepath.WalkDir(root, func(p string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, _ := filepath.Rel(root, p)
		relSlash := filepath.ToSlash(rel)
		name := d.Name()
		if d.IsDir() {
			if name == ".git" {
				return filepath.SkipDir
			}
			if forbiddenDirs[strings.ToLower(name)] || strings.EqualFold(relSlash, "eval") || strings.EqualFold(relSlash, "build/llamacpp") {
				t.Errorf("forbidden internal directory present in public tree: %s", relSlash)
				return filepath.SkipDir
			}
			return nil
		}
		if forbiddenFiles[name] || strings.HasPrefix(name, "AGENT_") || strings.Contains(name, "AGENT_PROMPT") {
			t.Errorf("forbidden internal file present in public tree: %s", relSlash)
		}
		ext := strings.ToLower(filepath.Ext(name))
		if ext == ".gguf" || ext == ".exe" || ext == ".dll" || ext == ".zip" || ext == ".bundle" {
			t.Errorf("binary/model artifact leaked into public tree: %s", relSlash)
		}
		if (ext == ".go" || ext == ".md" || ext == ".ps1" || ext == ".yml" || ext == ".toml") && relSlash != "scripts/check-clean.ps1" {
			if b, rErr := os.ReadFile(p); rErr == nil && piiPathRe.Match(b) {
				t.Errorf("PII or hardcoded local user path detected in %s", relSlash)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
