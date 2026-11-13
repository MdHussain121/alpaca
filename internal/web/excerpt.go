package web

import (
	"sort"
	"strings"
	"unicode"
)

// EstimateTokens is a conservative tokenizer-independent estimate. It counts
// words and punctuation at roughly 4 characters per token.
func EstimateTokens(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	chars := len([]rune(s))
	words := len(strings.Fields(s))
	est := (chars + 3) / 4
	if words > est {
		est = words
	}
	return est
}

// TruncateToTokens trims text to at most maxTokens using word boundaries.
func TruncateToTokens(s string, maxTokens int) string {
	if maxTokens <= 0 || EstimateTokens(s) <= maxTokens {
		return strings.TrimSpace(s)
	}
	maxChars := maxTokens * 4
	if maxChars >= len(s) {
		return strings.TrimSpace(s)
	}
	cut := maxChars
	if i := strings.LastIndexAny(s[:cut], " \n\t"); i > cut/2 {
		cut = i
	}
	trimmed := strings.TrimSpace(s[:cut])
	marker := "\n… [truncated]"
	if EstimateTokens(trimmed+marker) <= maxTokens {
		return trimmed + marker
	}
	return trimmed
}

// SelectExcerpt returns the passages of markdown most relevant to query within
// a token budget. Short documents are returned whole.
func SelectExcerpt(markdown, query string, maxTokens int) string {
	md := strings.TrimSpace(markdown)
	if maxTokens <= 0 || EstimateTokens(md) <= maxTokens {
		return md
	}
	blocks := splitBlocks(md)
	if len(blocks) == 0 {
		return TruncateToTokens(md, maxTokens)
	}
	terms := queryTerms(query)
	scores := make([]float64, len(blocks))
	for i, b := range blocks {
		scores[i] = float64(i) * 0.01
		low := strings.ToLower(b)
		for _, t := range terms {
			if strings.Contains(low, t) {
				scores[i] += 3
				if strings.HasPrefix(low, "#") {
					scores[i] += 1
				}
			}
		}
		if strings.HasPrefix(low, "#") {
			scores[i] += 0.2
		}
		if len(strings.Fields(b)) < 4 {
			scores[i] -= 0.2
		}
	}
	order := make([]int, len(blocks))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })

	chosen := make(map[int]bool)
	used := 0
	for _, idx := range order {
		t := EstimateTokens(blocks[idx]) + 1
		if used+t > maxTokens {
			continue
		}
		chosen[idx] = true
		used += t
		if used >= maxTokens-20 {
			break
		}
	}
	if len(chosen) == 0 {
		return TruncateToTokens(md, maxTokens)
	}
	var out []string
	for i, b := range blocks {
		if chosen[i] {
			out = append(out, b)
		}
	}
	result := strings.Join(out, "\n\n")
	if EstimateTokens(result) > maxTokens {
		return TruncateToTokens(result, maxTokens)
	}
	return result
}

func splitBlocks(md string) []string {
	raw := strings.Split(strings.ReplaceAll(md, "\r\n", "\n"), "\n\n")
	out := make([]string, 0, len(raw))
	for _, b := range raw {
		b = strings.TrimSpace(b)
		if b == "" {
			continue
		}
		if len(b) > 1200 {
			for _, line := range strings.Split(b, "\n") {
				if line = strings.TrimSpace(line); line != "" {
					out = append(out, line)
				}
			}
			continue
		}
		out = append(out, b)
	}
	return out
}

var excerptStop = map[string]bool{
	"the": true, "and": true, "for": true, "with": true, "what": true, "when": true,
	"where": true, "which": true, "who": true, "why": true, "how": true, "does": true,
	"did": true, "is": true, "are": true, "was": true, "were": true, "a": true,
	"an": true, "of": true, "to": true, "in": true, "on": true, "at": true,
	"by": true, "or": true, "as": true, "it": true, "this": true, "that": true,
	"from": true, "can": true, "could": true, "should": true, "would": true,
	"please": true, "tell": true, "me": true, "about": true, "best": true,
	"latest": true, "current": true, "today": true, "now": true, "2024": true,
	"2025": true, "2026": true,
}

func queryTerms(q string) []string {
	seen := map[string]bool{}
	var out []string
	for _, f := range strings.FieldsFunc(strings.ToLower(q), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '+' && r != '#'
	}) {
		if len(f) < 2 || excerptStop[f] || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, f)
	}
	return out
}

// BM25 is exported for tests; pages are split into blocks and scored with a
// simple term-frequency model.
func BM25(query string, docs []string) []float64 {
	terms := queryTerms(query)
	scores := make([]float64, len(docs))
	if len(terms) == 0 || len(docs) == 0 {
		return scores
	}
	avg := 0.0
	for _, d := range docs {
		avg += float64(len(strings.Fields(d)))
	}
	avg /= float64(len(docs))
	if avg == 0 {
		avg = 1
	}
	for i, d := range docs {
		words := strings.Fields(strings.ToLower(d))
		dl := float64(len(words))
		tf := map[string]int{}
		for _, w := range words {
			tf[strings.TrimFunc(w, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })]++
		}
		for _, t := range terms {
			f := float64(tf[t])
			if f == 0 {
				continue
			}
			denom := f + 1.2*(0.25+0.75*dl/avg)
			scores[i] += (f * 2.2) / denom
		}
	}
	return scores
}
