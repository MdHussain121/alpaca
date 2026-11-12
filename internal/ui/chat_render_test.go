package ui

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/gguf"
)

func writeMinimalGGUF(t *testing.T, path string) {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(3))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0)) // tensor count
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0)) // kv count
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFastStartupAndAsyncGPUUpdatesSmartDefaults(t *testing.T) {
	dir := t.TempDir()
	cfg := config.Load(dir)
	m := New(dir, cfg)
	if m.hw.RAMTotal == 0 || m.hw.PhysCores == 0 {
		t.Fatalf("New() must run DetectFast synchronously, got RAMTotal=%d PhysCores=%d", m.hw.RAMTotal, m.hw.PhysCores)
	}
	if m.hw.HasCUDA || m.gpuProbed {
		t.Fatalf("New() must not block probing GPU synchronously: HasCUDA=%v gpuProbed=%v", m.hw.HasCUDA, m.gpuProbed)
	}

	m.ready = true
	// Simulate 16 GiB host RAM with plenty of headroom so a compact GPU-resident model clears the 50% RAM guard.
	m.hw.RAMTotal = 16 << 30
	m.hw.RAMAvail = 14 << 30
	m.models = []modelEntry{
		{cand: gguf.Candidate{RelPath: "Demo.gguf", Path: "demo.gguf"}, meta: testMeta(4_000_000_000, 36)},
	}
	// Shrink mock layer sizes to ~70M/layer (~3G total <= 50% of 14G RAMAvail).
	for i := range m.models[0].meta.Layers {
		m.models[0].meta.Layers[i].Total = 70 << 20
	}
	m.applyFilter()
	m.scanning = false
	m.selectCurrent()

	// Async GPU probe completes -> GPU-resident safe fit switches AutoFit off, omits --no-warmup & -t, uses -lm none.
	m.Update(gpuMsg{name: "RTX 4060 Laptop GPU", total: 8 << 30, free: 7 << 30, ok: true})
	if !m.gpuProbed || !m.hw.HasCUDA {
		t.Fatal("gpuMsg did not merge CUDA device info")
	}
	if m.settings.AutoFit {
		t.Fatal("GPU-resident fit must turn AutoFit off automatically")
	}
	if strings.Contains(m.cmdPreview, "--no-warmup") {
		t.Fatalf("warmup=true default must omit --no-warmup: %s", m.cmdPreview)
	}
	if !strings.Contains(m.cmdPreview, "--fit off") || !strings.Contains(m.cmdPreview, "-lm none") {
		t.Fatalf("unexpected resolved command preview: %s", m.cmdPreview)
	}
}

func TestScanAndMetaCachePersistenceInUI(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "tiny.gguf")
	writeMinimalGGUF(t, modelPath)

	cfg := config.Load(dir)
	m := New(dir, cfg)

	sm, ok := m.scanCmd()().(scanMsg)
	if !ok || len(sm.entries) != 1 || sm.cache == nil {
		t.Fatalf("unexpected scanMsg: %#v", sm)
	}
	m.Update(sm)

	pm, ok := parseMetaCmd(0, modelPath, m.ggufCache)().(metaMsg)
	if !ok || pm.err != nil {
		t.Fatalf("parseMetaCmd = %#v", pm)
	}
	_, saveCmd := m.Update(pm)
	if saveCmd == nil {
		t.Fatal("expected saveCacheCmd once all model headers parsed")
	}
	saveCmd()

	if _, err := os.Stat(cfg.GGUFCachePath()); err != nil {
		t.Fatalf("expected persistent GGUF cache file %s: %v", cfg.GGUFCachePath(), err)
	}
	c2 := gguf.LoadCache(cfg.GGUFCachePath())
	_, hit, err := gguf.ParseCached(c2, modelPath)
	if err != nil || !hit {
		t.Fatalf("expected cache hit on reload, got hit=%v err=%v", hit, err)
	}
}

func TestLoadETAMetricsAndHealthTimeout(t *testing.T) {
	if got := healthTimeoutForETA(0); got != 10*time.Minute {
		t.Fatalf("healthTimeoutForETA(0) = %v, want 10m", got)
	}
	if got := healthTimeoutForETA(12 * time.Minute); got != 24*time.Minute {
		t.Fatalf("healthTimeoutForETA(12m) = %v, want 24m", got)
	}
	if d := fmtETA(135 * time.Second); d != "~2m15s" {
		t.Fatalf("fmtETA(135s) = %q", d)
	}

	m := testUI(t)
	m.screen = screenRun
	m.status = "LOADING"
	m.launchDriveKind = "removable"
	m.launchETA = 2 * time.Minute
	m.launchAt = time.Now().Add(-15 * time.Second)
	v := m.viewMetrics(120)
	if !strings.Contains(v, "USB") || !strings.Contains(v, "left") {
		t.Fatalf("expected honest USB load ETA in metrics, got %q", v)
	}
}

func TestCopyModelToLocalAndInsufficientSpaceAndCancel(t *testing.T) {
	srcDir := t.TempDir()
	destDir := filepath.Join(t.TempDir(), "local-models")

	s1 := filepath.Join(srcDir, "m-00001-of-00002.gguf")
	s2 := filepath.Join(srcDir, "m-00002-of-00002.gguf")
	mm := filepath.Join(srcDir, "mmproj-f16.gguf")
	payload1 := bytes.Repeat([]byte("A"), 256*1024)
	payload2 := bytes.Repeat([]byte("B"), 128*1024)
	payloadM := bytes.Repeat([]byte("M"), 64*1024)
	for path, data := range map[string][]byte{s1: payload1, s2: payload2, mm: payloadM} {
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// 1. Insufficient disk space must abort cleanly without creating .gguf in destDir.
	chFail := make(chan copyProgressMsg, 8)
	lowDisk := func(string) (uint64, uint64, error) { return 500 << 20, 1 << 30, nil }
	copyModelToLocal(context.Background(), s1, mm, destDir, lowDisk, chFail)
	var last copyProgressMsg
	for msg := range chFail {
		last = msg
	}
	if last.err == nil || !strings.Contains(last.err.Error(), "insufficient local disk space") {
		t.Fatalf("expected insufficient space error, got %+v", last)
	}

	// 2. Successful copy replicates all shards + mmproj cleanly.
	chOK := make(chan copyProgressMsg, 16)
	plentyDisk := func(string) (uint64, uint64, error) { return 20 << 30, 100 << 30, nil }
	copyModelToLocal(context.Background(), s1, mm, destDir, plentyDisk, chOK)
	for msg := range chOK {
		last = msg
	}
	if last.err != nil || !last.done || last.alreadyExists {
		t.Fatalf("unexpected copy result: %+v", last)
	}
	if got, err := os.ReadFile(filepath.Join(destDir, filepath.Base(s2))); err != nil || !bytes.Equal(got, payload2) {
		t.Fatalf("shard 2 mismatch: err=%v len=%d", err, len(got))
	}
	if got, err := os.ReadFile(filepath.Join(destDir, filepath.Base(mm))); err != nil || !bytes.Equal(got, payloadM) {
		t.Fatalf("mmproj mismatch: err=%v len=%d", err, len(got))
	}

	// 3. Repeat copy sees matching files and reports alreadyExists without rewriting.
	chDup := make(chan copyProgressMsg, 8)
	copyModelToLocal(context.Background(), s1, mm, destDir, plentyDisk, chDup)
	for msg := range chDup {
		last = msg
	}
	if last.err != nil || !last.alreadyExists {
		t.Fatalf("expected alreadyExists on matching dest, got %+v", last)
	}

	// 4. Cancelled copy removes .part file and rejects corrupt output.
	cancelDir := filepath.Join(t.TempDir(), "cancel-dest")
	chCancel := make(chan copyProgressMsg, 8)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	copyModelToLocal(ctx, s1, "", cancelDir, plentyDisk, chCancel)
	for msg := range chCancel {
		last = msg
	}
	if !errors.Is(last.err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", last.err)
	}
	if _, err := os.Stat(filepath.Join(cancelDir, filepath.Base(s1)+".part")); !os.IsNotExist(err) {
		t.Fatalf("expected .part cleanup after cancel, stat err=%v", err)
	}

	// 5. Overlay renders cleanly and Esc cancels active UI copy state cleanly.
	m := testUI(t)
	m.width, m.height = 100, 30
	m.copying = true
	m.copyTitle = "Demo-Coder-8B.gguf"
	m.copyProgress = copyProgressMsg{doneBytes: 2 << 30, totalBytes: 4 << 30, speedMBps: 200, eta: 10 * time.Second}
	canceledCalled := false
	m.copyCancel = func() { canceledCalled = true }
	view := m.View()
	if !strings.Contains(view, "Copying to local disk") || !strings.Contains(view, "200 MB/s") {
		t.Fatalf("overlayCopyProgress missing expected text:\n%s", view)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if m.copying || !canceledCalled {
		t.Fatal("esc must cancel active copy overlay")
	}
}
