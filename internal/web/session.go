package web

import (
	"fmt"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
)

var urlInTextRe = regexp.MustCompile(`https?://[^\s<>"'` + "`" + `]+`)

const MaxFetchFailuresPerTurn = 2

// Session holds the URLs a web tool is allowed to fetch during one chat
// session, the most recent numbered search results, and the per-turn fetch
// failure budget.
type Session struct {
	mu       sync.Mutex
	allowed  map[string]string
	results  []SearchResult
	query    string
	turn     int
	failures int
	attempts map[string]int
	lastErr  string
	fetched  map[string]bool
}

func NewSession() *Session {
	return &Session{
		allowed:  make(map[string]string),
		attempts: make(map[string]int),
		fetched:  make(map[string]bool),
	}
}

// BeginTurn resets the failure budget when the agent loop advances a turn.
func (s *Session) BeginTurn(turn int) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if turn == 0 || turn == s.turn {
		return
	}
	s.turn = turn
	s.failures = 0
	s.attempts = make(map[string]int)
	s.lastErr = ""
}

// SeedText allowlists http(s) URLs that the user typed in a message.
func (s *Session) SeedText(text string) {
	if s == nil || strings.TrimSpace(text) == "" {
		return
	}
	for _, raw := range urlInTextRe.FindAllString(text, -1) {
		s.AllowURL(strings.TrimRight(raw, ".,;:!?)]}"))
	}
}

// AllowURL records a URL if it is a valid http(s) URL. It returns the
// canonical form, or "" when the URL is unusable.
func (s *Session) AllowURL(raw string) string {
	if s == nil {
		return ""
	}
	canon, err := CanonicalHTTPURL(raw)
	if err != nil {
		return ""
	}
	s.mu.Lock()
	s.allowed[canon] = canon
	s.mu.Unlock()
	return canon
}

// AllowResults records every URL in a search result set.
func (s *Session) AllowResults(results []SearchResult) {
	for _, r := range results {
		s.AllowURL(r.URL)
	}
}

// AllowLinks resolves and records links found on a fetched page against base.
func (s *Session) AllowLinks(base string, links []string) {
	b, err := url.Parse(strings.TrimSpace(base))
	if err != nil {
		return
	}
	for _, raw := range links {
		ref, err := url.Parse(strings.TrimSpace(raw))
		if err != nil {
			continue
		}
		s.AllowURL(b.ResolveReference(ref).String())
	}
}

// IsAllowed reports whether the URL came from a search result, a fetched
// page's links, or the user's own message.
func (s *Session) IsAllowed(raw string) bool {
	if s == nil {
		return false
	}
	canon, err := CanonicalHTTPURL(raw)
	if err != nil {
		return false
	}
	s.mu.Lock()
	_, ok := s.allowed[canon]
	s.mu.Unlock()
	return ok
}

// RememberResults stores the latest numbered list used by fetch result=<n>.
func (s *Session) RememberResults(results []SearchResult) {
	s.RememberSearch("", results)
}

func (s *Session) RememberSearch(query string, results []SearchResult) {
	if s == nil {
		return
	}
	cp := append([]SearchResult(nil), results...)
	s.mu.Lock()
	s.results = cp
	if strings.TrimSpace(query) != "" {
		s.query = strings.TrimSpace(query)
	}
	s.mu.Unlock()
	s.AllowResults(cp)
}

// Query returns the latest search query for query-aware excerpts.
func (s *Session) Query() string {
	if s == nil {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.query
}

// Result returns a 1-based search result from the latest search.
func (s *Session) Result(n int) (SearchResult, bool) {
	if s == nil {
		return SearchResult{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if n < 1 || n > len(s.results) {
		return SearchResult{}, false
	}
	return s.results[n-1], true
}

// ResultCount returns the number of remembered results.
func (s *Session) ResultCount() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.results)
}

// AllowedSnapshot returns the current allowlist for diagnostics and tests.
func (s *Session) AllowedSnapshot() []string {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.allowed))
	for u := range s.allowed {
		out = append(out, u)
	}
	return out
}

// CheckFetch applies the per-turn failure and duplicate budget before a fetch.
func (s *Session) CheckFetch(raw string) error {
	if s == nil {
		return nil
	}
	canon, err := CanonicalHTTPURL(raw)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failures >= MaxFetchFailuresPerTurn {
		msg := fmt.Sprintf("Fetch budget exhausted (%d failed fetches this turn). Answer from what you already have, or say you could not find the information.", s.failures)
		if s.lastErr != "" {
			msg += " Last error: " + s.lastErr
		}
		return fmt.Errorf("%s", msg)
	}
	if s.attempts[canon] >= 2 {
		return fmt.Errorf("Already tried this URL twice this turn. Use a different result, or answer from what you have.")
	}
	s.attempts[canon]++
	return nil
}

// RecordFetch records success or failure. Successful fetches do not consume
// the failure budget.
func (s *Session) RecordFetch(raw string, err error) {
	if s == nil {
		return
	}
	canon, canonErr := CanonicalHTTPURL(raw)
	s.mu.Lock()
	defer s.mu.Unlock()
	if err == nil {
		if canonErr == nil {
			s.fetched[canon] = true
		}
		return
	}
	s.failures++
	s.lastErr = ShortError(err.Error(), 180)
}

// FetchedSnapshot returns pages successfully read in this session, for a
// deterministic Sources line.
func (s *Session) FetchedSnapshot() []string {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.fetched))
	for u := range s.fetched {
		out = append(out, u)
	}
	sort.Strings(out)
	return out
}

// CanonicalHTTPURL validates the scheme and removes fragments and default
// ports. It does not resolve DNS; Fetch does that.
func CanonicalHTTPURL(raw string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", fmt.Errorf("only http and https URLs are allowed")
	}
	if u.Host == "" {
		return "", fmt.Errorf("URL has no host")
	}
	if u.User != nil {
		return "", fmt.Errorf("URL credentials are not allowed")
	}
	u.Scheme = scheme
	u.Host = strings.ToLower(u.Host)
	u.Fragment = ""
	if u.Path == "" {
		u.Path = "/"
	} else if len(u.Path) > 1 {
		u.Path = strings.TrimSuffix(u.Path, "/")
	}
	if (scheme == "https" && u.Port() == "443") || (scheme == "http" && u.Port() == "80") {
		u.Host = u.Hostname()
	}
	return u.String(), nil
}

// ShortError trims an error to a single short line for tool output.
func ShortError(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if max > 0 && len(s) > max {
		s = s[:max] + "…"
	}
	return s
}
