package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/MdHussain121/alpaca/internal/flags"
)

func TestCorruptConfigFallsBackToDefaults(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.toml"), []byte("theme = [oops"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := Load(dir)
	if cfg.Theme == "" || cfg.Port == 0 || cfg.Root() != dir {
		t.Fatalf("defaults not applied: %+v", cfg)
	}
}

func TestPresetNameCannotEscapeTheFolder(t *testing.T) {
	dir := t.TempDir()
	cfg := Load(dir)

	p, err := cfg.SavePreset(Preset{Name: `..\..\x`, Settings: flags.Default(4, 8080)})
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	if p.ID != "x" {
		t.Fatalf("id = %q, want the path stripped to x", p.ID)
	}
	if _, err := os.Stat(filepath.Join(dir, "presets", "x.toml")); err != nil {
		t.Fatalf("preset not inside presets/: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "x.toml")); err == nil {
		t.Fatal("preset escaped the presets folder")
	}
}

func TestRenamePresetMovesTheFile(t *testing.T) {
	dir := t.TempDir()
	cfg := Load(dir)

	p, err := cfg.SavePreset(Preset{Name: "first", Settings: flags.Default(4, 8080)})
	if err != nil {
		t.Fatal(err)
	}
	renamed, err := cfg.RenamePreset(p, "second name")
	if err != nil {
		t.Fatalf("rename: %v", err)
	}
	if renamed.ID != "second-name" || renamed.Name != "second name" {
		t.Fatalf("renamed = %+v", renamed)
	}
	ps, err := cfg.ListPresets()
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 1 || ps[0].Name != "second name" {
		t.Fatalf("presets = %+v", ps)
	}
}

func TestListPresetsSkipsCorruptAndSorts(t *testing.T) {
	dir := t.TempDir()
	cfg := Load(dir)
	if _, err := cfg.SavePreset(Preset{Name: "zeta", Settings: flags.Default(4, 8080)}); err != nil {
		t.Fatal(err)
	}
	if _, err := cfg.SavePreset(Preset{Name: "Alpha", Settings: flags.Default(4, 8080)}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "presets", "broken.toml"), []byte("= not toml"), 0o644); err != nil {
		t.Fatal(err)
	}

	ps, err := cfg.ListPresets()
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 2 {
		t.Fatalf("presets = %d, corrupt file must be skipped", len(ps))
	}
	if ps[0].Name != "Alpha" {
		t.Fatalf("order = %q", ps[0].Name)
	}
	if ps[0].String() == "" {
		t.Error("empty preset description")
	}
}

func TestDeletePresetIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	cfg := Load(dir)
	if err := cfg.DeletePreset("never-existed"); err != nil {
		t.Fatalf("delete of a missing preset: %v", err)
	}
	if err := cfg.DeletePreset(""); err == nil {
		t.Fatal("empty id must be refused")
	}
}

func TestBuiltinPresetsRespectTrainedContext(t *testing.T) {
	ps := BuiltinPresets("m.gguf", 8192, true)
	if len(ps) != 3 {
		t.Fatalf("presets = %d", len(ps))
	}
	for _, p := range ps {
		if p.Context > 8192 {
			t.Errorf("%s context %d exceeds the trained 8192", p.Name, p.Context)
		}
	}
	if ps[2].NCpuMoe == 0 {
		t.Error("the MoE preset should keep some experts in RAM")
	}
}

func TestModelsPathDefaultsToTheAppFolder(t *testing.T) {
	dir := t.TempDir()
	cfg := Load(dir)
	if got := cfg.ModelsPath(); got != dir {
		t.Fatalf("default models path = %q, want the app folder %q", got, dir)
	}

	cfg.ModelsDir = "models"
	if got := cfg.ModelsPath(); got != filepath.Join(dir, "models") {
		t.Fatalf("relative models path = %q", got)
	}

	cfg.ModelsDir = filepath.Join(dir, "elsewhere")
	if got := cfg.ModelsPath(); got != cfg.ModelsDir {
		t.Fatalf("absolute models path = %q", got)
	}
}

func TestLastUsedRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cfg := Load(dir)

	s := flags.Default(4, 8080)
	s.Context = 16384
	cfg.Remember("m.gguf", s)
	if err := cfg.Clone().Save(); err != nil {
		t.Fatal(err)
	}

	got, ok := Load(dir).Recall("m.gguf")
	if !ok || got.Context != 16384 {
		t.Fatalf("recall = %+v, %v", got, ok)
	}
}

func TestConfigDefaultsAndPaths(t *testing.T) {
	dir := t.TempDir()
	customLocal := t.TempDir()
	t.Setenv("LOCALAPPDATA", customLocal)

	cfg := Load(dir)
	if cfg.ScanDepth != 3 {
		t.Fatalf("ScanDepth = %d, want 3", cfg.ScanDepth)
	}
	if got := cfg.GGUFCachePath(); got != filepath.Join(dir, ".gguf-cache.json") {
		t.Fatalf("GGUFCachePath = %q", got)
	}
	if got := cfg.LocalModelsDir(); got != filepath.Join(customLocal, "alpaca", "models") {
		t.Fatalf("LocalModelsDir = %q", got)
	}
}
