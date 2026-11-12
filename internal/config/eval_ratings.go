package config

import (
	_ "embed"
	"encoding/json"
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

// LoadEvalRatings parses the embedded eval_ratings.json and recomputes Badge from Passed/AttemptsTotal.
func LoadEvalRatings() EvalRatingsDoc {
	var doc EvalRatingsDoc
	if len(embeddedEvalRatingsJSON) == 0 {
		return doc
	}
	if err := json.Unmarshal(embeddedEvalRatingsJSON, &doc); err != nil {
		return EvalRatingsDoc{}
	}
	for i := range doc.Models {
		if doc.Models[i].AttemptsTotal > 0 {
			doc.Models[i].SuccessRatePct = float64(doc.Models[i].Passed) * 100.0 / float64(doc.Models[i].AttemptsTotal)
		}
		doc.Models[i].Badge = ComputeEvalBadge(doc.Models[i].SuccessRatePct)
	}
	return doc
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
