package ui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func testMeta(params uint64, layers int) *gguf.Metadata {
	m := &gguf.Metadata{
		Path:          "demo.gguf",
		Name:          "Demo-Coder-8B",
		Arch:          "qwen3",
		Quant:         "Q4_K_M",
		BlockCount:    uint64(layers),
		ContextLength: 32768,
		EmbeddingLen:  4096,
		HeadCount:     32,
		KeyLength:     128,
		ValueLength:   128,
		TotalParams:   params,
		GlobalBytes:   512 << 20,
		TotalBytes:    4 << 30,
	}
	m.HeadCountKV = []uint64{8}
	for i := 0; i < layers; i++ {
		m.Layers = append(m.Layers, gguf.LayerBytes{Total: 120 << 20})
	}
	return m
}

func testUI(t *testing.T) *Model {
	t.Helper()
	dir := t.TempDir()
	cfg := config.Load(dir)
	m := New(dir, cfg)
	m.ready = true
	m.hw = hw.Info{
		GPUName:     "RTX 4060 Laptop GPU",
		VRAMTotal:   8 << 30,
		VRAMFree:    7 << 30,
		HasCUDA:     true,
		RAMTotal:    16 << 30,
		RAMAvail:    12 << 30,
		PhysCores:   10,
		PCores:      6,
		PLogicCores: 12,
		ECores:      4,
		LogicCores:  16,
	}
	meta := testMeta(8_000_000_000, 36)
	m.models = []modelEntry{
		{cand: gguf.Candidate{RelPath: "Demo-Coder-8B.gguf", Path: "demo.gguf"}, meta: meta},
	}
	m.applyFilter()
	m.scanning = false
	m.selectCurrent()
	m.recompute()
	return m
}

func testModelFile(t *testing.T, name string) string {
	t.Helper()
	dir := strings.TrimSpace(os.Getenv("ALPACA_TEST_MODELS_DIR"))
	if dir == "" {
		t.Skipf("set ALPACA_TEST_MODELS_DIR to run live model tests (%s)", name)
	}
	p := filepath.Join(dir, name)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("missing test model %s: %v", p, err)
	}
	return p
}

func checkGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name+".golden")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("missing golden %s (create it with UPDATE_GOLDEN=1 go test ./internal/ui): %v", path, err)
	}
	if string(want) != got {
		t.Errorf("view differs from %s\n--- got ---\n%s\n--- want ---\n%s", path, got, string(want))
	}
}

func TestViewSnapshotSizes(t *testing.T) {
	cases := []struct {
		name string
		w, h int
	}{
		{"launcher-80x24", 80, 24},
		{"launcher-100x30", 100, 30},
		{"launcher-120x32", 120, 32},
		{"launcher-160x40", 160, 40},
		{"launcher-200x50", 200, 50},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := testUI(t)
			m.width, m.height = tc.w, tc.h
			m.resizeLogView()

			got := m.View()
			lines := strings.Split(got, "\n")
			if len(lines) != tc.h {
				t.Fatalf("rendered %d lines, want exactly %d", len(lines), tc.h)
			}
			for i, l := range lines {
				if w := visibleWidth(l); w > tc.w {
					t.Fatalf("line %d is %d cells wide, exceeds %d: %q", i, w, tc.w, l)
				}
			}
			if !strings.Contains(got, "llama-server") {
				t.Error("command preview missing from the launcher frame")
			}
			if !strings.Contains(got, "Demo-Coder-8B") {
				t.Error("selected model missing from the launcher frame")
			}
			checkGolden(t, tc.name, got)
		})
	}
}

// visibleWidth counts runes, ignoring ANSI escapes, which is exact for the
// characters this UI draws.
func visibleWidth(s string) int {
	n := 0
	esc := false
	for _, r := range s {
		switch {
		case r == 0x1b:
			esc = true
		case esc && (r == 'm' || r == 'K'):
			esc = false
		case !esc:
			n++
		}
	}
	return n
}

func TestTooSmallScreen(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 40, 12
	got := m.View()
	if !strings.Contains(got, "too small") {
		t.Fatalf("expected the too-small screen, got:\n%s", got)
	}
}

func TestQuantizedVCacheForcesFlashAttnByKeyboard(t *testing.T) {
	m := testUI(t)
	m.settings.AutoKV = false
	m.settings.KCache = "q8_0"
	m.settings.VCache = "f16"
	m.settings.FlashAttn = false
	m.recompute()

	m.focus = 1
	fs := m.visibleFields()

	idx := -1
	for i, f := range fs {
		if f.label == "V cache" {
			idx = i
			break
		}
	}
	if idx < 0 {
		t.Fatal("no V cache field")
	}
	m.fieldIdx = idx

	// f16 -> q8_0
	m.adjustField(1)
	if m.settings.VCache != "q8_0" {
		t.Fatalf("V cache = %q", m.settings.VCache)
	}
	if !m.settings.FlashAttn {
		t.Fatal("flash attention must be forced on for a quantized V cache")
	}
	if !strings.Contains(m.cmdPreview, "-fa on") {
		t.Fatalf("command preview missing -fa on: %s", m.cmdPreview)
	}
	if !strings.Contains(m.cmdPreview, "-ctv q8_0") {
		t.Fatalf("command preview missing -ctv q8_0: %s", m.cmdPreview)
	}

	// q8_0 -> q4_0 must surface the quality tradeoff note.
	m.adjustField(1)
	if m.settings.VCache != "q4_0" || m.est.KVQualityNote == "" {
		t.Fatalf("expected q4_0 + quality tradeoff note, got v=%q note=%q", m.settings.VCache, m.est.KVQualityNote)
	}
}

func TestContextChangeUpdatesPreviewAndEstimate(t *testing.T) {
	m := testUI(t)
	kvBefore := m.est.KVVRAM

	m.focus = 1
	m.fieldIdx = 0
	m.adjustField(1) // 8K -> 16K
	m.adjustField(1) // 16K -> 32K

	if m.settings.Context != 32768 {
		t.Fatalf("context = %d", m.settings.Context)
	}
	if !strings.Contains(m.cmdPreview, "-c 32768") {
		t.Fatalf("preview not updated: %s", m.cmdPreview)
	}
	if m.est.KVVRAM <= kvBefore {
		t.Fatalf("KV estimate did not grow: %d -> %d", kvBefore, m.est.KVVRAM)
	}
}

func TestMoEFieldHiddenForDenseModels(t *testing.T) {
	m := testUI(t)
	for _, f := range m.visibleFields() {
		if f.label == "MoE experts on CPU" || f.label == "MoE cache" {
			t.Fatalf("MoE field %q must be hidden for a dense model", f.label)
		}
	}

	// With small MoE model (~4 GiB) and 12 GiB avail RAM (>= model + 4 GiB), MoE cache
	// is available in Settings as manual "off" (never auto).
	m.meta.ExpertCount = 128
	m.meta.TotalBytes = 4 << 30
	m.settings.NCpuMoe = 24
	m.recompute()
	foundExp, foundCache := false, false
	for _, f := range m.visibleFields() {
		if f.label == "MoE experts on CPU" {
			foundExp = true
		}
		if f.label == "MoE cache" {
			foundCache = true
			if f.value() != "off" || f.note() == "" {
				t.Fatalf("MoE cache must default to manual off & show cap note, got val=%q note=%q", f.value(), f.note())
			}
		}
	}
	if !foundExp || !foundCache {
		t.Fatalf("MoE fields missing for MoE model with RAM headroom: experts=%v cache=%v", foundExp, foundCache)
	}

	// When model >= avail RAM - 4 GiB, MoE cache field must be hidden to protect page cache.
	m.meta.TotalBytes = 10 << 30
	m.recompute()
	for _, f := range m.visibleFields() {
		if f.label == "MoE cache" {
			t.Fatalf("MoE cache must be hidden when RAMAvail < model + 4 GiB")
		}
	}

	// Forcing none/dio on MoE must display an explicit warning in the Memory panel.
	m.width, m.height = 120, 32
	m.settings.LoadMode = "none"
	m.recompute()
	if v := m.View(); !strings.Contains(v, "unsafe for MoE") {
		t.Fatalf("forced -lm none on MoE must warn visibly in View:\n%s", v)
	}
}

func TestSlowDriveOversizedRAMWarning(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 140, 32
	m.meta.FileSize = 18 << 30 // 18 GiB model > 12 GiB RAMAvail
	m.modelDriveKind = "removable"
	m.recompute()
	if v := m.View(); !strings.Contains(v, "copy local") || !strings.Contains(v, "slow drive") {
		t.Fatalf("Memory panel must warn on slow drive when model > avail RAM and suggest copy local:\n%s", v)
	}
}

func TestGPULauncherShowsAllLayers(t *testing.T) {
	m := testUI(t)
	m.focus = 1
	for i, f := range m.visibleFields() {
		if f.label == "GPU layers" {
			m.fieldIdx = i
			if !strings.Contains(f.value(), "all") {
				t.Fatalf("expected all layers by default, got %q", f.value())
			}
			return
		}
	}
	t.Fatal("no GPU layers field")
}

func TestPresetRoundTripIsKeyedByFileName(t *testing.T) {
	m := testUI(t)
	m.settings.Context = 16384
	m.settings.VCache = "q4_0"
	if pm, ok := m.saveCurrentPreset()().(presetMsg); ok && pm.err != nil {
		t.Fatalf("save: %v", pm.err)
	}

	ps, err := m.cfg.ListPresets()
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 1 {
		t.Fatalf("presets = %d", len(ps))
	}
	if ps[0].Model != "demo.gguf" {
		t.Errorf("preset model = %q", ps[0].Model)
	}
	if ps[0].Context != 16384 {
		t.Errorf("preset ctx = %d", ps[0].Context)
	}

	// Settings must restore from the preset regardless of drive letter.
	m.settings = flags.Default(4, 8080)
	loaded := ps[0].Settings
	if loaded.VCache != "q4_0" || !loaded.FlashAttn {
		t.Errorf("preset did not preserve V cache/flash attn: %+v", loaded)
	}
}

func TestPresetRenameReachesDisk(t *testing.T) {
	m := testUI(t)
	msg := m.saveCurrentPreset()()
	pm, ok := msg.(presetMsg)
	if !ok || pm.err != nil {
		t.Fatalf("save: %#v", msg)
	}
	m.presets = pm.presets

	rmsg := renamePresetCmd(m.cfg, m.presets[0], "Renamed preset")()
	rp, ok := rmsg.(presetMsg)
	if !ok || rp.err != nil {
		t.Fatalf("rename: %#v", rmsg)
	}
	ps, err := m.cfg.ListPresets()
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 1 {
		t.Fatalf("presets after rename = %d", len(ps))
	}
	if ps[0].Name != "Renamed preset" {
		t.Fatalf("name = %q", ps[0].Name)
	}
}

func TestCrashedServerIsReportedAsError(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.proc = &runner.Process{}
	m.status = "READY"
	m.logs = []string{"llama_model_load: error loading model: bad tensor"}

	m.Update(exitMsg{})

	if m.status != "ERROR" {
		t.Fatalf("status = %q, a crash must not keep READY", m.status)
	}
	if !strings.Contains(m.statusDe, "error loading model") {
		t.Fatalf("reason = %q", m.statusDe)
	}
}

func TestLANToggleAsksForConfirmation(t *testing.T) {
	m := testUI(t)
	m.focus = 1
	idx := -1
	for i, f := range m.visibleFields() {
		if f.label == "Expose on LAN" {
			idx = i
		}
	}
	if idx < 0 {
		t.Fatal("no LAN field")
	}
	m.fieldIdx = idx
	m.width, m.height = 100, 30
	m.adjustField(1)
	if !m.settings.LAN {
		t.Fatal("LAN should be on after the toggle")
	}
	if !m.lanConfirm {
		t.Fatal("exposing on the LAN must ask for confirmation")
	}
	if !strings.Contains(m.View(), "Expose llama-server on the LAN?") {
		t.Fatal("warning not rendered")
	}

	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	if m.settings.LAN {
		t.Fatal("n must keep the server on 127.0.0.1")
	}
}

func TestBelowMinimumHeightUsesFriendlyScreen(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 80, 23
	if !strings.Contains(m.View(), "too small") {
		t.Fatal("80x23 is below the 80x24 minimum and must show the friendly screen")
	}
}

func TestAutoKVAbove16KSuggestsNKVO(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 140, 40
	// Large KV demand at 32K where f16 & q8_0 exceed 8 GiB VRAM.
	m.meta.TotalBytes = 5 << 30
	m.meta.Layers = make([]gguf.LayerBytes, 36)
	for i := range m.meta.Layers {
		m.meta.Layers[i].Total = 120 << 20
	}
	m.settings.AutoKV = true
	m.settings.Context = 32768
	m.settings.NoKVOffload = false
	m.recompute()
	if m.resolved.KCache == "q4_0" || m.resolved.VCache == "q4_0" {
		t.Fatalf("silent q4_0 selected at 32K: %s/%s", m.resolved.KCache, m.resolved.VCache)
	}
	v := m.View()
	if !strings.Contains(v, "16K") || !strings.Contains(v, "-nkvo") {
		t.Fatalf("expected >16K q4_0-blocked and -nkvo suggestion in Memory view:\n%s", v)
	}
}

func TestAllV11ControlsReachArgsNoMock(t *testing.T) {
	m := testUI(t)
	m.meta.ExpertCount = 128
	m.meta.TotalBytes = 4 << 30
	m.hw.RAMTotal = 32 << 30
	m.hw.RAMAvail = 24 << 30
	m.recompute()

	adjustByLabel := func(label string, dir int) {
		t.Helper()
		m.focus = 1
		for i, f := range m.visibleFields() {
			if f.label == label {
				m.fieldIdx = i
				key := tea.KeyRight
				if dir < 0 {
					key = tea.KeyLeft
				}
				m.Update(tea.KeyMsg{Type: key})
				return
			}
		}
		t.Fatalf("field %q not visible", label)
	}

	m.settings.AutoFit = true
	m.userEditedAutoFit = true
	m.recompute()
	adjustByLabel("Auto fit", 1)
	if m.settings.AutoFit || !strings.Contains(m.cmdPreview, "--fit off") || !strings.Contains(m.cmdPreview, "-ngl all") {
		t.Fatalf("Auto fit toggle missing --fit off -ngl all: %s", m.cmdPreview)
	}
	adjustByLabel("Auto KV", 1)
	adjustByLabel("V cache", 1)
	if !strings.Contains(m.cmdPreview, "-ctv ") {
		t.Fatalf("KV change missing -ctv: %s", m.cmdPreview)
	}
	adjustByLabel("Load mode", 1)
	if !strings.Contains(m.cmdPreview, "-lm ") {
		t.Fatalf("Load mode change missing -lm: %s", m.cmdPreview)
	}
	m.settings.Warmup = true
	m.recompute()
	adjustByLabel("Warmup", 1)
	if !strings.Contains(m.cmdPreview, "--no-warmup") {
		t.Fatalf("Warmup toggle missing --no-warmup: %s", m.cmdPreview)
	}
	m.settings.SpecType = "none"
	m.recompute()
	adjustByLabel("Speculative", 1)
	if !strings.Contains(m.cmdPreview, "--spec-type ngram-mod") {
		t.Fatalf("Speculative toggle missing --spec-type ngram-mod: %s", m.cmdPreview)
	}
	adjustByLabel("Priority", 1)
	if !strings.Contains(m.cmdPreview, "--prio ") {
		t.Fatalf("Priority change missing --prio: %s", m.cmdPreview)
	}
	m.settings.NoWebUI = true
	m.recompute()
	adjustByLabel("Web UI", 1)
	if strings.Contains(m.cmdPreview, "--no-webui") {
		t.Fatalf("Web UI toggle on should remove --no-webui: %s", m.cmdPreview)
	}
	m.settings.NCpuMoe = 24
	m.recompute()
	adjustByLabel("MoE cache", 1)
	if !strings.Contains(m.cmdPreview, "--moe-cache-mib ") {
		t.Fatalf("MoE cache change missing --moe-cache-mib: %s", m.cmdPreview)
	}
}
