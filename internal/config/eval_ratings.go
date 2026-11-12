package config

import (
	_ "embed"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

//go:embed eval_ratings.json
var embeddedEvalRatingsJSON []byte

const (
	EvalBadgeRecommended    = "recommended"
	EvalBadgeWorksLimits    = "works with limits"
	EvalBadgeNotRecommended = "not recommended"
)

// EvalModelResult holds measured tool-use evaluation metrics for a GGUF file.
type EvalModelResult struct {
	File           string  `json:"file"`
	ModelName      string  `json:"model_name"`
	TasksTotal     int     `json:"tasks_total"`
	AttemptsTotal  int     `json:"attempts_total"`
	Passed         int     `json:"passed"`
	SuccessRatePct float64 `json:"success_rate_pct"`
	AvgSteps       float64 `json:"avg_steps"`
	AvgSec         float64 `json:"avg_sec"`
	Badge          string  `json:"badge"`
}

// EvalRatingsDoc is serialized by cmd/eval into internal/config/eval_ratings.json.
type EvalRatingsDoc struct {
	GeneratedAt string            `json:"generated_at"`
	Models      []EvalModelResult `json:"models"`
}

// ComputeEvalBadge derives the UI recommendation tier purely from measured pass rate.
func ComputeEvalBadge(successPct float64) string {
	switch {
	case successPct >= 80.0:
		return EvalBadgeRecommended
	case successPct >= 50.0:
		return EvalBadgeWorksLimits
	default:
		return EvalBadgeNotRecommended
	}
}

// LoadEvalRatings parses eval ratings (checking embedded defaults merged with any disk overrides)
// and recomputes Badge from Passed/AttemptsTotal.
func LoadEvalRatings() EvalRatingsDoc {
	modelsByFile := make(map[string]EvalModelResult)

	// 1. Load embedded defaults
	if len(embeddedEvalRatingsJSON) > 0 {
		var embeddedDoc EvalRatingsDoc
		if err := json.Unmarshal(embeddedEvalRatingsJSON, &embeddedDoc); err == nil {
			for _, m := range embeddedDoc.Models {
				modelsByFile[strings.ToLower(filepath.Base(m.File))] = m
			}
		}
	}

	// 2. Overlay disk overrides from working directory or app root
	candidates := []string{
		"eval_ratings.json",
		filepath.Join(".alpaca", "eval_ratings.json"),
		filepath.Join("internal", "config", "eval_ratings.json"),
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(dir, "eval_ratings.json"),
			filepath.Join(dir, ".alpaca", "eval_ratings.json"),
		)
	}
	for _, c := range candidates {
		if b, err := os.ReadFile(c); err == nil && len(b) > 0 {
			var diskDoc EvalRatingsDoc
			if err := json.Unmarshal(b, &diskDoc); err == nil {
				for _, m := range diskDoc.Models {
					modelsByFile[strings.ToLower(filepath.Base(m.File))] = m
				}
			}
		}
	}

	var list []EvalModelResult
	for _, m := range modelsByFile {
		if m.AttemptsTotal > 0 {
			m.SuccessRatePct = float64(m.Passed) * 100.0 / float64(m.AttemptsTotal)
		}
		m.Badge = ComputeEvalBadge(m.SuccessRatePct)
		list = append(list, m)
	}
	return EvalRatingsDoc{
		Models: list,
	}
}

// EvalRatingForFile looks up a model's data-driven eval rating by filename.
func EvalRatingForFile(filePathOrBase string) (EvalModelResult, bool) {
	base := strings.ToLower(strings.TrimSpace(filepath.Base(filePathOrBase)))
	if base == "" || base == "." {
		return EvalModelResult{}, false
	}
	doc := LoadEvalRatings()
	for _, m := range doc.Models {
		if strings.ToLower(filepath.Base(m.File)) == base {
			return m, true
		}
	}
	return EvalModelResult{}, false
}
