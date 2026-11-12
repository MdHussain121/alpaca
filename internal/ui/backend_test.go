package ui

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

// A drive must launch on a machine with no GPU at all, and must never pick a
// backend folder that is not there.
func TestBackendSelectionFallsBackToCPU(t *testing.T) {
	dir := t.TempDir()
	m := New(dir, config.Load(dir))

	m.hw = hw.Info{HasCUDA: false, HasAVX2: true}
	if got := m.backendDir(); got != "cpu" {
		t.Fatalf("no GPU: backend = %q, want cpu", got)
	}

	m.hw = hw.Info{HasCUDA: true, HasAVX2: true}
	if got := m.backendDir(); got != "cpu" {
		t.Fatalf("GPU but no bin/cuda: backend = %q, want cpu", got)
	}

	if err := os.MkdirAll(filepath.Join(dir, "bin", "vulkan"), 0o755); err != nil {
		t.Fatal(err)
	}
	m.hw = hw.Info{HasCUDA: false, HasVulkan: false, HasAVX2: true}
	if got := m.backendDir(); got != "cpu" {
		t.Fatalf("vulkan folder without vulkan-1.dll must fall back to cpu, got %q", got)
	}
	m.hw = hw.Info{HasCUDA: false, HasVulkan: true, HasAVX2: true}
	if got := m.backendDir(); got != "vulkan" {
		t.Fatalf("vulkan with HasVulkan=true = %q, want vulkan", got)
	}

	if err := os.MkdirAll(filepath.Join(dir, "bin", "cuda"), 0o755); err != nil {
		t.Fatal(err)
	}
	m.hw = hw.Info{HasCUDA: true, HasVulkan: true, HasAVX2: true}
	if got := m.backendDir(); got != "cuda" {
		t.Fatalf("GPU with bin/cuda: backend = %q, want cuda", got)
	}

	if err := os.MkdirAll(filepath.Join(dir, "bin", "cpu-compat"), 0o755); err != nil {
		t.Fatal(err)
	}
	m.hw = hw.Info{HasCUDA: false, HasVulkan: false, HasAVX2: false}
	if got := m.backendDir(); got != "cpu-compat" {
		t.Fatalf("non-AVX2 host with bin/cpu-compat: backend = %q, want cpu-compat", got)
	}

	m.cfg.Backend = "cpu"
	m.hw.HasAVX2 = true
	if got := m.backendDir(); got != "cpu" {
		t.Fatalf("explicit cpu on AVX2 host: backend = %q", got)
	}
}

// Quitting while a server runs must be confirmed, not immediate.
func TestRunScreenRequiresConfirmationToQuit(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.status = "READY"
	m.proc = &runner.Process{}

	model, _ := m.updateRun(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'q'}})
	got := model.(*Model)
	if got.quit {
		t.Fatal("q on the run screen must ask first")
	}
	if !got.runQuitConfirm {
		t.Fatal("no confirmation shown")
	}
}

func TestCompatibilityDegradationModes(t *testing.T) {
	t.Run("tiny resize degrades gracefully without panic", func(t *testing.T) {
		m := testUI(t)
		m.Update(tea.WindowSizeMsg{Width: 40, Height: 10})
		out := m.View()
		if !strings.Contains(out, "resize") && !strings.Contains(out, "80x24") {
			t.Fatalf("expected small-window hint, got %q", out)
		}
	})

	t.Run("ASCII and NO_COLOR mode avoid Unicode box drawing", func(t *testing.T) {
		t.Setenv("NO_COLOR", "1")
		dir := t.TempDir()
		cfg := config.Load(dir)
		cfg.ASCII = true
		m := New(dir, cfg)
		m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
		out := m.View()
		if strings.ContainsAny(out, "╭╮╰╯━") {
			t.Fatalf("expected ASCII borders in ASCII mode, got %q", out)
		}
	})

	t.Run("OSC 52 clipboard fallback writes valid sequence", func(t *testing.T) {
		var buf bytes.Buffer
		if err := writeOSC52(&buf, "alpaca"); err != nil {
			t.Fatalf("writeOSC52: %v", err)
		}
		if !strings.HasPrefix(buf.String(), "\x1b]52;c;") {
			t.Fatalf("unexpected OSC 52 payload: %q", buf.String())
		}
	})

	t.Run("paths with spaces and Unicode resolve without hardcoded drives", func(t *testing.T) {
		base := t.TempDir()
		uniDir := filepath.Join(base, "alpaca portable café 日本語")
		if err := os.MkdirAll(uniDir, 0o755); err != nil {
			t.Fatal(err)
		}
		cfg := config.Load(uniDir)
		m := New(uniDir, cfg)
		if m.root != uniDir {
			t.Fatalf("root mismatch: got %q, want %q", m.root, uniDir)
		}
	})

	t.Run("offline local copy to missing/read-only path fails clearly", func(t *testing.T) {
		srcDir := t.TempDir()
		src := filepath.Join(srcDir, "m.gguf")
		if err := os.WriteFile(src, []byte("gguf"), 0o644); err != nil {
			t.Fatal(err)
		}
		ch := make(chan copyProgressMsg, 4)
		copyModelToLocal(context.Background(), src, "", filepath.Join(srcDir, "nul\x00bad"), nil, ch)
		var last copyProgressMsg
		for msg := range ch {
			last = msg
		}
		if last.err == nil {
			t.Fatal("expected clear error for invalid destination path")
		}
	})
}
