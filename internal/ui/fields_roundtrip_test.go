package ui

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
)

func TestSelectorAndSliderArrowBugsRepro(t *testing.T) {
	t.Run("Batch slider left then right from 256 preserves stop", func(t *testing.T) {
		m := testUI(t)
		m.focus = 1
		m.settings.Batch = 256
		m.recompute()
		m.fieldIdx = fieldIndexByLabel(m, "Batch")

		press(m, "left")
		press(m, "right")
		if m.settings.Batch != 256 {
			t.Fatalf("Batch after <- then -> from 256 = %d, want 256", m.settings.Batch)
		}
	})

	t.Run("Ubatch slider left then right from 128 preserves stop", func(t *testing.T) {
		m := testUI(t)
		m.focus = 1
		m.settings.Batch = 2048
		m.settings.UBatch = 128
		m.recompute()
		m.fieldIdx = fieldIndexByLabel(m, "Ubatch")

		press(m, "left")
		press(m, "right")
		if m.settings.UBatch != 128 {
			t.Fatalf("Ubatch after <- then -> from 128 = %d, want 128", m.settings.UBatch)
		}
	})

	t.Run("Float sliders Temp/Top-p/Min-p round-trip without IEEE-754 drift", func(t *testing.T) {
		m := testUI(t)
		m.focus = 1

		m.settings.Temp = 0.7
		m.fieldIdx = fieldIndexByLabel(m, "Temp")
		press(m, "right")
		press(m, "left")
		if m.settings.Temp != 0.7 {
			t.Fatalf("Temp after -> then <- = %.17g, want 0.7", m.settings.Temp)
		}

		m.settings.TopP = 0.85
		m.fieldIdx = fieldIndexByLabel(m, "Top-p")
		press(m, "right")
		press(m, "left")
		if m.settings.TopP != 0.85 {
			t.Fatalf("TopP after -> then <- = %.17g, want 0.85", m.settings.TopP)
		}

		m.settings.MinP = 0.15
		m.fieldIdx = fieldIndexByLabel(m, "Min-p")
		press(m, "right")
		press(m, "left")
		if m.settings.MinP != 0.15 {
			t.Fatalf("MinP after -> then <- = %.17g, want 0.15", m.settings.MinP)
		}
	})

	t.Run("Reasoning budget left from custom value 300 steps to 256, not 8192", func(t *testing.T) {
		m := testUI(t)
		m.focus = 1
		m.settings.ReasoningBudget = "300"
		m.recompute()
		m.fieldIdx = fieldIndexByLabel(m, "Reasoning budget")

		press(m, "left")
		if m.settings.ReasoningBudget != "256" {
			t.Fatalf("ReasoningBudget after <- from 300 = %q, want 256", m.settings.ReasoningBudget)
		}
	})

	t.Run("K cache cycles through every option cleanly on arrows", func(t *testing.T) {
		m := testUI(t)
		m.focus = 1
		m.fieldIdx = fieldIndexByLabel(m, "K cache")

		// Even starting from AutoKV=true default, stepping right through all options and left back
		// must visit distinct options cleanly and return cleanly.
		startVal := m.visibleFields()[m.fieldIdx].value()
		seen := map[string]bool{startVal: true}
		for i := 0; i < len(flags.CacheOptions())-1; i++ {
			press(m, "right")
			v := m.visibleFields()[m.fieldIdx].value()
			if seen[v] {
				t.Fatalf("K cache repeated %q early after %d right steps: seen=%v", v, i+1, seen)
			}
			seen[v] = true
		}
		for i := 0; i < len(flags.CacheOptions())-1; i++ {
			press(m, "left")
		}
		endVal := m.visibleFields()[m.fieldIdx].value()
		if endVal != startVal {
			t.Fatalf("K cache value after ->.. then <-.. = %q, want %q", endVal, startVal)
		}
	})
}

func TestEverySelectToggleSliderRoundTrip(t *testing.T) {
	runModelCases := func(name string, setup func(*testing.T) *Model) {
		t.Run(name, func(t *testing.T) {
			m := setup(t)
			m.focus = 1

			// Interior settings so numeric sliders can step both +1 and -1 cleanly.
			m.settings.AutoKV = false
			m.settings.KCache = "q8_0"
			m.settings.VCache = "q8_0"
			m.settings.FlashAttn = true
			m.settings.Context = 8192
			m.settings.GPULayers = 16
			m.settings.NCpuMoe = 4
			m.settings.Batch = 1024
			m.settings.UBatch = 256
			m.settings.Parallel = 2
			m.settings.Temp = 0.7
			m.settings.TopP = 0.85
			m.settings.TopK = 40
			m.settings.MinP = 0.15
			m.settings.Threads = 4
			m.settings.MaxTokens = 1024
			m.settings.ReasoningBudget = "1024"
			m.settings.Port = 8080
			m.recompute()

			fs := m.visibleFields()
			for i, f := range fs {
				if f.delta == nil {
					continue
				}
				// Skip locked Flash attn when quantized V cache forces it on.
				if f.label == "Flash attn" && m.settings.FlashAttnLocked() {
					continue
				}
				// Skip AutoKV right->left if VCache would change resolution or verify displayed value returns.
				if f.label == "Expose on LAN" {
					// Toggling LAN on triggers confirmation banner; test round-trip value still restores cleanly.
				}

				beforeVal := f.value()
				beforeSettings := m.settings
				beforeTheme := m.cfg.Theme

				m.fieldIdx = i
				press(m, "right")
				if m.lanConfirm {
					press(m, "y")
				}
				afterRight := m.visibleFields()[m.fieldIdx].value()
				if afterRight == beforeVal && f.label != "MoE cache" {
					t.Fatalf("field %q right arrow had no effect from %q", f.label, beforeVal)
				}
				press(m, "left")

				curFields := m.visibleFields()
				if m.fieldIdx >= len(curFields) || curFields[m.fieldIdx].label != f.label {
					t.Fatalf("field focus drifted from %q to index %d", f.label, m.fieldIdx)
				}
				afterBack := curFields[m.fieldIdx].value()
				if afterBack != beforeVal {
					t.Fatalf("field %q value after -> then <- = %q, want %q", f.label, afterBack, beforeVal)
				}
				// Compare core field value preservation; Theme updates m.cfg.Theme instead of m.settings.
				if f.label == "Theme" && m.cfg.Theme != beforeTheme {
					t.Fatalf("Theme after -> then <- = %q, want %q", m.cfg.Theme, beforeTheme)
				}
				_ = beforeSettings
			}
		})
	}

	runModelCases("dense", func(t *testing.T) *Model {
		return testUI(t)
	})
	runModelCases("moe", func(t *testing.T) *Model {
		m := testUI(t)
		moe := testMeta(35_000_000_000, 48)
		moe.ExpertCount = 128
		moe.ExpertUsed = 8
		m.models = []modelEntry{{cand: gguf.Candidate{RelPath: "MoE.gguf", Path: "moe.gguf"}, meta: moe}}
		m.selectCurrent()
		return m
	})
}

func TestRandomSettingsKeySequenceNeverBreaksInvariants(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 120, 32
	m.focus = 1

	rng := rand.New(rand.NewSource(42))
	keys := []string{"left", "right", "h", "l", "up", "down", "j", "k"}
	for step := 0; step < 500; step++ {
		if m.lanConfirm {
			press(m, "y")
			continue
		}
		k := keys[rng.Intn(len(keys))]
		press(m, k)
		fs := m.visibleFields()
		if len(fs) == 0 {
			t.Fatalf("step %d: empty visibleFields", step)
		}
		if m.fieldIdx < 0 || m.fieldIdx >= len(fs) {
			t.Fatalf("step %d (%s): fieldIdx %d out of range [0,%d)", step, k, m.fieldIdx, len(fs))
		}
		norm := m.settings
		norm.Normalize()
		if m.settings.KCache != norm.KCache || m.settings.VCache != norm.VCache ||
			m.settings.Batch != norm.Batch || m.settings.UBatch != norm.UBatch {
			t.Fatalf("step %d (%s): settings not normalized: %+v vs %+v", step, k, m.settings, norm)
		}
		if !strings.Contains(m.cmdPreview, "-m ") {
			t.Fatalf("step %d (%s): invalid cmdPreview: %q", step, k, m.cmdPreview)
		}
	}
}
