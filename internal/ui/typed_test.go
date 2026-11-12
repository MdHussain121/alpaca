package ui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/config"
)

func fieldIndexByLabel(m *Model, label string) int {
	for i, f := range m.visibleFields() {
		if f.label == label {
			return i
		}
	}
	return -1
}

func TestTypedValueOverridesThePresetStops(t *testing.T) {
	m := testUI(t)
	idx := fieldIndexByLabel(m, "Context")
	if idx < 0 {
		t.Fatal("no context field")
	}
	m.fieldIdx = idx
	if !m.beginEdit() {
		t.Fatal("context is not editable")
	}
	m.applyTypedValue("20000")

	if m.settings.Context != 20000 {
		t.Fatalf("context = %d, want the typed 20000", m.settings.Context)
	}
	if !strings.Contains(m.cmdPreview, "-c 20000") {
		t.Fatalf("preview not updated: %s", m.cmdPreview)
	}
}

func TestTypedValuesAreValidated(t *testing.T) {
	m := testUI(t)
	m.fieldIdx = fieldIndexByLabel(m, "Threads")
	if !m.beginEdit() {
		t.Fatal("threads is not editable")
	}
	before := m.settings.Threads
	m.applyTypedValue("many")

	if m.settings.Threads != before {
		t.Fatalf("garbage changed threads to %d", m.settings.Threads)
	}
	if !strings.Contains(m.toast, "Threads") {
		t.Fatalf("no error surfaced: %q", m.toast)
	}
}

// Research bug: changing a row used to change the row above it. Nothing may
// mutate a preceding row.
func TestChangingARowLeavesTheRowAboveAlone(t *testing.T) {
	m := testUI(t)
	m.focus = 1

	fs := m.visibleFields()
	prev := make([]string, len(fs))
	for i := range fs {
		prev[i] = fs[i].value()
	}

	for i := 1; i < len(fs); i++ {
		m.fieldIdx = i
		m.adjustField(1)

		cur := m.visibleFields()
		if len(cur) != len(prev) {
			t.Fatalf("field list changed size after %q", fs[i].label)
		}
		if cur[i-1].value() != prev[i-1] {
			t.Fatalf("changing %q also changed %q: %q -> %q",
				fs[i].label, fs[i-1].label, prev[i-1], cur[i-1].value())
		}
		for j := range cur {
			prev[j] = cur[j].value()
		}
	}
}

func TestWebUIToggleReachesTheCommand(t *testing.T) {
	m := testUI(t)
	if !m.settings.NoWebUI || !strings.Contains(m.cmdPreview, "--no-webui") {
		t.Fatalf("the web UI must be off (--no-webui) by default: %s", m.cmdPreview)
	}

	m.fieldIdx = fieldIndexByLabel(m, "Web UI")
	m.adjustField(1)
	if m.settings.NoWebUI || strings.Contains(m.cmdPreview, "--no-webui") {
		t.Fatalf("toggle on failed: %s", m.cmdPreview)
	}

	m.adjustField(-1)
	if !m.settings.NoWebUI || !strings.Contains(m.cmdPreview, "--no-webui") {
		t.Fatalf("toggle off failed: %s", m.cmdPreview)
	}
}

func TestProjectorIsFoundNextToTheModelAndPassedThrough(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "vision-model.gguf")
	proj := filepath.Join(dir, "mmproj-vision-f16.gguf")
	for _, p := range []string{model, proj} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	if got := findProjector(model); got != proj {
		t.Fatalf("projector = %q, want %q", got, proj)
	}

	m := testUI(t)
	m.settings.MMProj = proj
	m.settings.NoWebUI = true // must work with the web UI off
	m.recompute()

	if !strings.Contains(m.cmdPreview, "--mmproj") {
		t.Fatalf("projector missing from the command: %s", m.cmdPreview)
	}
	if !strings.Contains(m.cmdPreview, "--no-webui") {
		t.Fatalf("web UI toggle lost: %s", m.cmdPreview)
	}
}

// Dropping a .gguf next to alpaca.exe must be enough: no config edits.
func TestScanFindsModelsInTheAppFolderByDefault(t *testing.T) {
	root := t.TempDir()
	model := filepath.Join(root, "dropped-here.gguf")
	if err := os.WriteFile(model, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := New(root, config.Load(root))
	sm, ok := m.scanCmd()().(scanMsg)
	if !ok {
		t.Fatal("unexpected scan message")
	}
	for _, e := range sm.entries {
		if e.cand.Path == model {
			return
		}
	}
	t.Fatalf("model in the app folder not found: %#v", sm.entries)
}

func TestScanIncludesExplicitModelPaths(t *testing.T) {
	root := t.TempDir()
	cfg := config.Load(root)
	other := t.TempDir()
	extra := filepath.Join(other, "D-drive-model.gguf")
	if err := os.WriteFile(extra, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg.AddExtraModel(extra)

	m := New(root, cfg)
	msg := m.scanCmd()()

	sm, ok := msg.(scanMsg)
	if !ok {
		t.Fatalf("unexpected message %#v", msg)
	}
	for _, e := range sm.entries {
		if e.cand.Path == extra {
			return
		}
	}
	t.Fatalf("explicit path missing from the scan: %#v", sm.entries)
}

func TestOpenPathAcceptsFolderFileAndRejectsJunk(t *testing.T) {
	dir := t.TempDir()

	if pm, ok := openPathCmd(dir)().(pathMsg); !ok || pm.modelsDir == "" || pm.err != nil {
		t.Fatalf("folder rejected: %#v", pm)
	}

	file := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if pm, ok := openPathCmd(file)().(pathMsg); !ok || pm.model == "" || pm.err != nil {
		t.Fatalf("model file rejected: %#v", pm)
	}

	if pm, ok := openPathCmd(filepath.Join(dir, "missing.gguf"))().(pathMsg); !ok || pm.err == nil {
		t.Fatalf("missing path accepted: %#v", pm)
	}

	txt := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(txt, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if pm, ok := openPathCmd(txt)().(pathMsg); !ok || pm.err == nil {
		t.Fatalf("non-model file accepted: %#v", pm)
	}
}
