package config

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const migrationMarkerFile = ".migrated"

// MigrationResult reports legacy config or data directories found and migrated.
type MigrationResult struct {
	Notices []string
}

func findLegacyConfigFile(root string) string {
	candidates := []string{
		filepath.Join(root, "llm.toml"),
		filepath.Join(root, "pllama.toml"),
		filepath.Join(root, ".llm", "config.toml"),
		filepath.Join(root, ".pllama", "config.toml"),
	}
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}
	return ""
}

// MigrateLegacy checks root (and workspace/) for prior llm or pllama config
// files and internal data folders, migrates readable data into .alpaca/
// without overwriting existing files, and returns one-time user notices.
func MigrateLegacy(root string) MigrationResult {
	var res MigrationResult
	if strings.TrimSpace(root) == "" {
		return res
	}

	rootMarker := filepath.Join(root, ".alpaca", migrationMarkerFile)
	alreadyNotified := fileExists(rootMarker)

	// Migrate legacy config file if config.toml does not yet exist.
	cfgPath := filepath.Join(root, configName)
	if !fileExists(cfgPath) {
		if legacyCfg := findLegacyConfigFile(root); legacyCfg != "" {
			if !alreadyNotified {
				res.Notices = append(res.Notices, fmt.Sprintf("loaded legacy config %s (kept presets/skills/sessions untouched)", filepath.Base(legacyCfg)))
			}
			_ = copyFileIfMissing(legacyCfg, cfgPath)
		}
	}

	// Migrate legacy internal data folders at root and workspace.
	pairs := []struct {
		srcName string
		base    string
	}{
		{".llm", root},
		{".pllama", root},
		{".llm", filepath.Join(root, "workspace")},
		{".pllama", filepath.Join(root, "workspace")},
	}
	for _, pair := range pairs {
		src := filepath.Join(pair.base, pair.srcName)
		if !dirExists(src) {
			continue
		}
		dst := filepath.Join(pair.base, ".alpaca")
		marker := filepath.Join(dst, migrationMarkerFile)
		if fileExists(marker) {
			continue
		}
		copied, err := copyDirMissing(src, dst)
		if err != nil {
			res.Notices = append(res.Notices, fmt.Sprintf("found legacy %s data (read-only mode; keeping existing layout)", pair.srcName))
			continue
		}
		_ = writeMigrationMarker(marker, pair.srcName)
		if copied > 0 || !alreadyNotified {
			res.Notices = append(res.Notices, fmt.Sprintf("migrated legacy %s data to .alpaca/", pair.srcName))
		}
	}

	// Detect legacy executables or local model cache once.
	if !alreadyNotified {
		for _, legacyBin := range []string{"llm.exe", "pllama.exe"} {
			if fileExists(filepath.Join(root, legacyBin)) {
				res.Notices = append(res.Notices, fmt.Sprintf("found legacy %s; presets, skills, and sessions remain compatible", legacyBin))
				_ = writeMigrationMarker(rootMarker, legacyBin)
				break
			}
		}
		if base := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); base != "" {
			oldCache := filepath.Join(base, "llm-launcher", "models")
			newCache := filepath.Join(base, "alpaca", "models")
			if dirHasGGUF(oldCache) && !dirHasGGUF(newCache) {
				res.Notices = append(res.Notices, "found cached models in %LOCALAPPDATA%\\llm-launcher\\models")
				_ = writeMigrationMarker(rootMarker, "local-cache")
			}
		}
	}

	return res
}

func writeMigrationMarker(markerPath, source string) error {
	if err := os.MkdirAll(filepath.Dir(markerPath), 0o755); err != nil {
		return err
	}
	if fileExists(markerPath) {
		return nil
	}
	return os.WriteFile(markerPath, []byte("migrated from "+source+"\n"), 0o644)
}

func copyDirMissing(srcDir, dstDir string) (int, error) {
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return 0, err
	}
	copied := 0
	err := filepath.WalkDir(srcDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil || rel == "." {
			return err
		}
		if d.Name() == migrationMarkerFile {
			return nil
		}
		target := filepath.Join(dstDir, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if fileExists(target) {
			return nil
		}
		if err := copyFileIfMissing(path, target); err != nil {
			return err
		}
		copied++
		return nil
	})
	return copied, err
}

func copyFileIfMissing(src, dst string) error {
	if fileExists(dst) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func dirExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

func dirHasGGUF(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".gguf") {
			return true
		}
	}
	return false
}
