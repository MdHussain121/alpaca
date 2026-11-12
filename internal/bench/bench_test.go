package bench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSummarize(t *testing.T) {
	s := summarize([]float64{10, 50, 20, 40, 30})
	if s.Min != 10 || s.Median != 30 || s.Max != 50 {
		t.Fatalf("unexpected summary: %+v", s)
	}
	even := summarize([]float64{10, 20, 30, 40})
	if even.Median != 25 {
		t.Fatalf("expected median 25, got %v", even.Median)
	}
}

func TestUpdateMilestones(t *testing.T) {
	var sb StartupBreakdown
	updateMilestones(&sb, 120_000_000, "ggml_cuda_init: found 1 CUDA devices")
	updateMilestones(&sb, 450_000_000, "llama_model_loader: loaded meta data")
	updateMilestones(&sb, 900_000_000, "warming up the model for the first run")
	updateMilestones(&sb, 1400_000_000, "srv    load_model: initializing, n_slots = 1")
	updateMilestones(&sb, 1600_000_000, "srv  server_on_li: listening on http://127.0.0.1:8080")
	if sb.BackendInitMs != 120 || sb.ModelLoadStartMs != 450 || sb.WarmupStartMs != 900 || sb.WarmupDoneMs != 1400 || sb.ListeningMs != 1600 {
		t.Fatalf("unexpected milestones: %+v", sb)
	}
}

func TestPromptsDatasetValid(t *testing.T) {
	path := filepath.Join("..", "bench", "data", "prompts.json")
	if _, err := os.Stat(path); err != nil {
		path = filepath.Join("..", "..", "perf", "data", "prompts.json")
	}
	all, err := loadAndFilterPrompts(path, "all", 0)
	if err != nil {
		t.Fatalf("load prompts: %v", err)
	}
	if len(all) != 20 {
		t.Fatalf("expected 20 prompts, got %d", len(all))
	}
	var short, med, long int
	for _, p := range all {
		switch p.Tier {
		case "short":
			short++
		case "medium":
			med++
		case "long":
			long++
		default:
			t.Fatalf("unexpected tier %q", p.Tier)
		}
	}
	if short != 10 || med != 5 || long != 5 {
		t.Fatalf("unexpected tier split: short=%d med=%d long=%d", short, med, long)
	}
}

func TestNeedlePromptContainsSecret(t *testing.T) {
	p := buildNeedlePrompt(1000, "739418")
	if !strings.Contains(p, "739418") {
		t.Fatalf("needle prompt missing secret")
	}
}

