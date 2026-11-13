package web

import (
	"net/url"
	"sort"
	"strings"
)

var domainBoost = map[string]float64{
	"go.dev":                6,
	"pkg.go.dev":            6,
	"developer.mozilla.org": 6,
	"docs.python.org":       6,
	"nodejs.org":            6,
	"react.dev":             6,
	"kubernetes.io":         5,
	"learn.microsoft.com":   5,
	"docs.aws.amazon.com":   5,
	"cloud.google.com":      4,
	"rust-lang.org":         5,
	"doc.rust-lang.org":     6,
	"typescriptlang.org":    5,
	"wikipedia.org":         4,
	"en.wikipedia.org":      4,
	"arxiv.org":             4,
	"github.com":            4,
	"stackoverflow.com":     3,
	"news.ycombinator.com":  3,
	"npmjs.com":             4,
	"pypi.org":              4,
	"rfc-editor.org":        5,
	"ietf.org":              4,
	"w3.org":                4,
	"kernel.org":            4,
	"llama.cpp":             4,
	"github.blog":           4,
	"openai.com":            2,
	"huggingface.co":        3,
}

var domainBlock = map[string]bool{
	"pinterest.com":        true,
	"quora.com":            true,
	"answers.com":          true,
	"ehow.com":             true,
	"slideshare.net":       true,
	"scribd.com":           true,
	"coursehero.com":       true,
	"chegg.com":            true,
	"doubleclick.net":      true,
	"googleadservices.com": true,
}

// DomainOf returns a lowercase host without a leading www.
func DomainOf(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return ""
	}
	h := strings.ToLower(u.Hostname())
	return strings.TrimPrefix(h, "www.")
}

func domainScore(host string) float64 {
	if host == "" {
		return 0
	}
	if domainBlock[host] {
		return -100
	}
	best := 0.0
	for d, b := range domainBoost {
		if host == d || strings.HasSuffix(host, "."+d) {
			if b > best {
				best = b
			}
		}
	}
	return best
}

// RankResults de-duplicates by canonical URL and orders official/primary
// sources above generic results, with a small query-term match bonus.
func RankResults(results []SearchResult, query string) []SearchResult {
	seen := map[string]bool{}
	terms := queryTerms(query)
	type scored struct {
		r     SearchResult
		score float64
	}
	var list []scored
	for _, r := range results {
		canon, err := CanonicalHTTPURL(r.URL)
		if err != nil || seen[canon] {
			continue
		}
		seen[canon] = true
		r.URL = canon
		host := DomainOf(canon)
		r.Domain = host
		score := domainScore(host)
		lowTitle := strings.ToLower(r.Title)
		lowSnip := strings.ToLower(r.Snippet)
		for _, t := range terms {
			if strings.Contains(lowTitle, t) {
				score += 1.5
			}
			if strings.Contains(lowSnip, t) {
				score += 0.4
			}
		}
		list = append(list, scored{r: r, score: score})
	}
	sort.SliceStable(list, func(i, j int) bool { return list[i].score > list[j].score })
	out := make([]SearchResult, 0, len(list))
	for _, s := range list {
		out = append(out, s.r)
	}
	return out
}
