package web

import (
	"os"
	"strings"
)

type SearchResult struct {
	Title   string
	URL     string
	Snippet string
	Domain  string
}

// RedactSecrets replaces secret-looking environment values inside s. The
// keyless backends do not use keys, but errors from an optional SearXNG URL
// can still contain credentials.
func RedactSecrets(s string, secrets []string) string {
	out := s
	for _, sec := range secrets {
		sec = strings.TrimSpace(sec)
		if len(sec) >= 4 {
			out = strings.ReplaceAll(out, sec, "[REDACTED]")
		}
	}
	for _, envName := range []string{"BRAVE_API_KEY", "TAVILY_API_KEY", "ALPACA_WEB_SEARCH_KEY", "LLM_WEB_SEARCH_KEY"} {
		if v := strings.TrimSpace(os.Getenv(envName)); len(v) >= 4 {
			out = strings.ReplaceAll(out, v, "[REDACTED]")
		}
	}
	return out
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
