package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSandboxRejectsHostilePaths(t *testing.T) {
	base := t.TempDir()
	ws := filepath.Join(base, "ws")
	outside := filepath.Join(base, "ws2")
	if err := os.MkdirAll(ws, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("nope"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "ok.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}

	hostile := []struct {
		name string
		path string
		mode AccessMode
	}{
		{"empty", "", AccessRead},
		{"whitespace", "   ", AccessRead},
		{"nul byte", "ok.txt\x00.png", AccessRead},
		{"ctrl char", "bad\x07.txt", AccessWrite},
		{"dotdot simple", "../ws2/secret.txt", AccessRead},
		{"dotdot backslash", `..\ws2\secret.txt`, AccessRead},
		{"dotdot nested", "sub/../../ws2/secret.txt", AccessRead},
		{"prefix sibling abs", filepath.Join(outside, "secret.txt"), AccessRead},
		{"unc backslash", `\\127.0.0.1\c$\Windows\win.ini`, AccessRead},
		{"unc forward", `//server/share/file.txt`, AccessRead},
		{"extended win path", `\\?\` + filepath.Join(ws, "ok.txt"), AccessRead},
		{"device namespace", `\\.\PhysicalDrive0`, AccessRead},
		{"nt namespace", `\??\C:\Windows\System32\cmd.exe`, AccessRead},
		{"drive relative", `C:secret.txt`, AccessRead},
		{"root relative backslash", `\Windows\win.ini`, AccessRead},
		{"root relative forward", `/etc/passwd`, AccessRead},
		{"ads relative", "ok.txt:hiddenstream", AccessWrite},
		{"ads $DATA", `ok.txt::$DATA`, AccessWrite},
		{"ads subdir", "dir/a:b.txt", AccessCreate},
		{"ads absolute", filepath.Join(ws, "ok.txt:stream"), AccessWrite},
		{"reserved CON", "CON", AccessWrite},
		{"reserved con lower", "con.txt", AccessWrite},
		{"reserved PRN", "sub/PRN.log", AccessWrite},
		{"reserved AUX", "AUX", AccessRead},
		{"reserved NUL", "NUL", AccessWrite},
		{"reserved COM1", "com1", AccessWrite},
		{"reserved COM9", "COM9.bin", AccessWrite},
		{"reserved LPT1", "LPT1.txt", AccessWrite},
		{"reserved LPT9", "nested/lpt9", AccessWrite},
		{"trailing dot file", "notes.txt.", AccessWrite},
		{"trailing dot dir", "sub./file.txt", AccessCreate},
		{"trailing space file", "notes.txt ", AccessWrite},
		{"trailing space dir", "sub /file.txt", AccessCreate},
		{"write .alpaca root", ".alpaca", AccessWrite},
		{"write .alpaca nested", ".alpaca/checkpoints/foo.json", AccessCreate},
		{"write .alpaca uppercase", `.ALPACA\trash\x`, AccessWrite},
		{"write .llm root", ".llm", AccessWrite},
		{"write .llm nested", ".llm/checkpoints/foo.json", AccessCreate},
		{"write .llm uppercase", `.LLM\trash\x`, AccessWrite},
		{"write skills root", "skills", AccessCreate},
		{"write skills subfile", "skills/demo/SKILL.md", AccessCreate},
		{"write workspace root", ".", AccessWrite},
		{"other volume abs", `Z:\nonexistent\ws\file.txt`, AccessRead},
	}

	if len(hostile) < 30 {
		t.Fatalf("hostile suite must have >=30 cases, got %d", len(hostile))
	}

	for _, tc := range hostile {
		t.Run(tc.name, func(t *testing.T) {
			abs, rel, err := ResolveWorkspacePath(ws, tc.path, tc.mode)
			if err == nil {
				t.Fatalf("expected hostile path %q (mode=%v) to be rejected, but resolved abs=%q rel=%q", tc.path, tc.mode, abs, rel)
			}
		})
	}

	// Optional symlink/junction escape check (when OS allows creating symlinks without elevation).
	linkPath := filepath.Join(ws, "escape_link")
	if err := os.Symlink(outside, linkPath); err == nil {
		if _, _, rErr := ResolveWorkspacePath(ws, "escape_link/secret.txt", AccessRead); rErr == nil {
			t.Fatalf("expected symlink pointing outside workspace to be rejected")
		}
	}

	// Valid paths inside workspace must succeed.
	if abs, rel, err := ResolveWorkspacePath(ws, "sub/newfile.txt", AccessCreate); err != nil {
		t.Fatalf("expected valid nested workspace create path to succeed: %v", err)
	} else if rel != "sub/newfile.txt" || abs == "" {
		t.Fatalf("unexpected resolved valid path: abs=%q rel=%q", abs, rel)
	}

	// Reading existing file inside skills/ MUST be allowed for Skill progressive loading.
	skillFile := filepath.Join(ws, "skills", "demo", "SKILL.md")
	if err := os.MkdirAll(filepath.Dir(skillFile), 0o755); err == nil {
		_ = os.WriteFile(skillFile, []byte("# demo\n"), 0o644)
		if _, rel, err := ResolveWorkspacePath(ws, "skills/demo/SKILL.md", AccessRead); err != nil || rel != "skills/demo/SKILL.md" {
			t.Fatalf("expected reading skills/SKILL.md to be permitted, got rel=%q err=%v", rel, err)
		}
	}
}
