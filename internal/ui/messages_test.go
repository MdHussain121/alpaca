package ui

import (
	"errors"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestAppendLogBoundsAndParsesMetrics(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.status = "LOADING"

	for i := 0; i < 6000; i++ {
		m.appendLog("filler line")
	}
	if len(m.logs) > 5000 {
		t.Fatalf("log buffer = %d lines, want it bounded", len(m.logs))
	}
	if len(m.logs) < 4000 {
		t.Fatalf("log buffer dropped too much history: %d lines", len(m.logs))
	}

	m.appendLog("llama_kv_cache: size = 1024.00 MiB (8192 cells, 36 layers, 1/1 seqs), K (q8_0): 512.00 MiB, V (f16): 512.00 MiB")
	if m.metrics.KVActualMiB != 1024 {
		t.Fatalf("KV actual = %v", m.metrics.KVActualMiB)
	}

	m.appendLog("llama_model_load: error loading model: bad tensor")
	if !strings.Contains(m.statusDe, "error loading model") {
		t.Fatalf("load error not surfaced: %q", m.statusDe)
	}
}

func TestHealthMessageOnlyTrustsTheCurrentProcess(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.proc = &runner.Process{}
	m.status = "LOADING"

	m.Update(healthMsg{err: nil, proc: m.proc})
	if m.status != "READY" {
		t.Fatalf("status = %q", m.status)
	}

	// A poll left over from a stopped server must not change anything.
	m.status = "LOADING"
	m.Update(healthMsg{err: errors.New("late"), proc: &runner.Process{}})
	if m.status != "LOADING" {
		t.Fatalf("stale health message changed the status to %q", m.status)
	}
}

func TestLaunchWithoutModelIsRefused(t *testing.T) {
	m := testUI(t)
	m.meta = nil

	m.launch()

	if !strings.Contains(m.toast, "no model") {
		t.Fatalf("toast = %q", m.toast)
	}
	if m.screen == screenRun {
		t.Fatal("launch must not switch screens without a model")
	}
}

func TestPresetDeleteKeyRemovesTheFile(t *testing.T) {
	m := testUI(t)
	msg := m.saveCurrentPreset()()
	pm, ok := msg.(presetMsg)
	if !ok || pm.err != nil {
		t.Fatalf("save: %#v", msg)
	}
	m.presets = pm.presets
	m.showPreset = true
	m.presetIdx = 0

	cmd := press(m, "d")
	if m.showPreset {
		t.Fatal("overlay should close after deleting")
	}
	if cmd == nil {
		t.Fatal("delete did not dispatch")
	}
	if dmsg, ok := cmd().(presetMsg); !ok || dmsg.err != nil {
		t.Fatalf("delete result: %#v", dmsg)
	}
	ps, err := m.cfg.ListPresets()
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 0 {
		t.Fatalf("presets left = %d", len(ps))
	}
}

func TestMemoryPanelWarnsWhenOverBudget(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 120, 32
	m.meta = testMeta(8_000_000_000, 36)
	m.settings.Context = 131072
	m.settings.GPULayers = flags.AllCount
	m.recompute()

	view := m.View()
	if !strings.Contains(view, "over budget") && !strings.Contains(view, "tight fit") {
		t.Fatalf("no memory warning at 128K context:\n%s", view)
	}
}

func TestHelpAndTooSmallViews(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 100, 30
	m.showHelp = true
	if !strings.Contains(m.View(), "run screen") {
		t.Fatal("help view missing the run-screen hint")
	}

	m.showHelp = false
	m.width, m.height = 60, 18
	if !strings.Contains(m.View(), "too small") {
		t.Fatal("60x18 must show the friendly screen")
	}
}

func TestEmptyAndFailingModelListViews(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 120, 32
	m.models = nil
	m.visible = nil
	m.scanning = false
	m.meta = nil
	m.recompute()
	if !strings.Contains(m.View(), "no .gguf files") {
		t.Fatal("empty model list not explained")
	}

	m.scanErr = errors.New("access denied")
	if !strings.Contains(m.View(), "scan failed") {
		t.Fatal("scan error not shown")
	}
}

func TestToastAndLongCommandRendering(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 80, 24
	m.toastf("hello toast")
	if !strings.Contains(m.View(), "hello toast") {
		t.Fatal("toast not rendered")
	}

	m.meta.Path = strings.Repeat("a-very-long-folder-name\\", 8) + "model.gguf"
	m.recompute()
	if !strings.Contains(m.View(), "…") {
		t.Fatal("long command line not elided")
	}
}

func TestPresetOverlayLoadsTheSelectedPreset(t *testing.T) {
	m := testUI(t)
	m.settings.Context = 16384
	msg := m.saveCurrentPreset()()
	if pm, ok := msg.(presetMsg); ok && pm.err == nil {
		m.presets = pm.presets
	}
	m.width, m.height = 100, 30
	m.showPreset = true
	m.presetIdx = 0
	m.settings.Context = 8192

	press(m, "enter")

	if m.settings.Context != 16384 {
		t.Fatalf("context = %d, want the preset value", m.settings.Context)
	}
	if m.showPreset {
		t.Fatal("overlay stayed open after loading")
	}
}

func TestSettingsPanelScrollsToTheFocusedField(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 100, 30
	m.focus = 1
	m.fieldIdx = fieldIndexByLabel(m, "Jinja template")

	view := m.View()
	if !strings.Contains(view, "Jinja") {
		t.Fatalf("focused field is off screen:\n%s", view)
	}
}

func TestRunScreenFitsEverySize(t *testing.T) {
	sizes := [][2]int{{80, 24}, {100, 30}, {120, 32}, {160, 40}, {200, 50}}
	for _, sz := range sizes {
		m := testUI(t)
		m.width, m.height = sz[0], sz[1]
		m.screen = screenRun
		m.status = "READY"
		m.meta = testMeta(8_000_000_000, 36)
		m.recompute()

		lines := strings.Split(m.View(), "\n")
		if len(lines) != sz[1] {
			t.Fatalf("%dx%d rendered %d lines", sz[0], sz[1], len(lines))
		}
		if !strings.Contains(m.View(), "llama-server log") {
			t.Fatalf("%dx%d lost the log panel", sz[0], sz[1])
		}
	}
}
