package web

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

type BackendAttempt struct {
	Backend  string        `json:"backend"`
	Query    string        `json:"query"`
	Results  int           `json:"results"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"duration"`
}

type SearchReport struct {
	Query     string           `json:"query"`
	Rewritten string           `json:"rewritten,omitempty"`
	Backend   string           `json:"backend"`
	Results   []SearchResult   `json:"results"`
	Attempts  []BackendAttempt `json:"attempts"`
	FromCache bool             `json:"from_cache,omitempty"`
	Err       error            `json:"-"`
}

type ChainOptions struct {
	ClientOpts ClientOptions
	CacheDir   string
	SearXNGURL string
	Timeout    time.Duration
	Backends   []Searcher
	Cache      *Cache
}

type breaker struct {
	fails     int
	openUntil time.Time
}

type Chain struct {
	backends []Searcher
	timeout  time.Duration
	cache    *Cache

	mu       sync.Mutex
	breakers map[string]*breaker
	lastCall map[string]time.Time
}

func NewKeylessChain(opts ChainOptions) *Chain {
	backends := opts.Backends
	if len(backends) == 0 {
		backends = []Searcher{
			&arxivSearcher{opts: opts.ClientOpts},
			&githubSearcher{opts: opts.ClientOpts},
			&npmSearcher{opts: opts.ClientOpts},
			&stackSearcher{opts: opts.ClientOpts},
			&hnSearcher{opts: opts.ClientOpts},
			&wikipediaSearcher{opts: opts.ClientOpts},
			&wikidataSearcher{opts: opts.ClientOpts},
			&ddgSearcher{opts: opts.ClientOpts},
			&mojeekSearcher{opts: opts.ClientOpts},
		}
		if strings.TrimSpace(opts.SearXNGURL) != "" {
			backends = append([]Searcher{&searxSearcher{base: opts.SearXNGURL, opts: opts.ClientOpts}}, backends...)
		}
	}
	cache := opts.Cache
	if cache == nil {
		cache = NewCache(opts.CacheDir, SearchCacheTTL)
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	return &Chain{
		backends: backends,
		timeout:  timeout,
		cache:    cache,
		breakers: make(map[string]*breaker),
		lastCall: make(map[string]time.Time),
	}
}

func (c *Chain) Backends() []string {
	out := make([]string, 0, len(c.backends))
	for _, b := range c.backends {
		out = append(out, b.Name())
	}
	return out
}

func (c *Chain) Search(ctx context.Context, query string, maxResults int) SearchReport {
	report := SearchReport{Query: strings.TrimSpace(query)}
	if maxResults < 1 || maxResults > 10 {
		maxResults = 5
	}
	norm := NormalizeQuery(report.Query)
	if norm == "" {
		report.Err = fmt.Errorf("query is empty after removing filler words")
		return report
	}
	report.Rewritten = norm
	key := CacheKey("search", norm, fmt.Sprint(maxResults))
	var cached cachedSearch
	if c.cache != nil && c.cache.GetJSON(key, &cached) && len(cached.Results) > 0 {
		report.Backend = cached.Backend
		report.Results = cached.Results
		report.FromCache = true
		return report
	}

	order := c.route(norm)
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	queries := []string{norm}
	if simplified := simplifyQuery(norm); simplified != norm && simplified != "" {
		queries = append(queries, simplified)
	}

	for _, b := range order {
		if err := c.wait(ctx, b.Name()); err != nil {
			report.Attempts = append(report.Attempts, BackendAttempt{Backend: b.Name(), Error: err.Error()})
			continue
		}
		if open, until := c.isOpen(b.Name()); open {
			report.Attempts = append(report.Attempts, BackendAttempt{Backend: b.Name(), Error: "circuit open until " + until.Format("15:04:05")})
			continue
		}
		for _, q := range queries {
			start := time.Now()
			callCtx, cancelCall := context.WithTimeout(ctx, 7*time.Second)
			results, err := b.Search(callCtx, q, maxResults)
			cancelCall()
			att := BackendAttempt{Backend: b.Name(), Query: q, Duration: time.Since(start)}
			if err != nil {
				att.Error = ShortError(err.Error(), 160)
				report.Attempts = append(report.Attempts, att)
				c.recordFailure(b.Name(), err)
				if ctx.Err() != nil {
					cancel()
					report.Err = fmt.Errorf("search timed out: %w", ctx.Err())
					return report
				}
				continue
			}
			results = RankResults(results, norm)
			if len(results) > maxResults {
				results = results[:maxResults]
			}
			att.Results = len(results)
			report.Attempts = append(report.Attempts, att)
			if len(results) == 0 {
				c.recordFailure(b.Name(), nil)
				continue
			}
			c.recordSuccess(b.Name())
			report.Backend = b.Name()
			report.Results = results
			if c.cache != nil {
				c.cache.PutJSON(key, cachedSearch{Backend: b.Name(), Results: results})
			}
			return report
		}
	}
	report.Err = fmt.Errorf("no keyless search backend returned results (%d attempts); try a shorter query", len(report.Attempts))
	return report
}

func (c *Chain) route(query string) []Searcher {
	low := strings.ToLower(query)
	var first []string
	switch {
	case containsAny(low, "arxiv", "paper", "preprint", "citation"):
		first = []string{"arxiv"}
	case containsAny(low, "github", "repository", "repo", "golang", "go package", "golang package"):
		first = []string{"github"}
	case containsAny(low, "npm", "node package", "javascript package", "typescript package"):
		first = []string{"npm"}
	case containsAny(low, "stack overflow", "stackoverflow", "exception", "traceback", "segfault", "error:"):
		first = []string{"stackexchange"}
	case containsAny(low, "hacker news", " show hn", "ask hn"):
		first = []string{"hackernews"}
	case containsAny(low, "who is", "what is", "wikipedia", "wikidata", "born", "capital of"):
		first = []string{"wikipedia", "wikidata"}
	default:
		first = []string{"duckduckgo", "mojeek"}
	}
	byName := make(map[string]Searcher, len(c.backends))
	for _, b := range c.backends {
		byName[b.Name()] = b
	}
	seen := map[string]bool{}
	var out []Searcher
	add := func(name string) {
		if b, ok := byName[name]; ok && !seen[name] {
			seen[name] = true
			out = append(out, b)
		}
	}
	for _, n := range first {
		add(n)
	}
	for _, n := range []string{"duckduckgo", "mojeek", "wikipedia", "wikidata", "hackernews", "github", "stackexchange", "arxiv", "npm", "searxng"} {
		add(n)
	}
	for _, b := range c.backends {
		add(b.Name())
	}
	return out
}

func (c *Chain) wait(ctx context.Context, name string) error {
	c.mu.Lock()
	last := c.lastCall[name]
	c.mu.Unlock()
	if !last.IsZero() {
		if d := time.Second - time.Since(last); d > 0 {
			t := time.NewTimer(d)
			defer t.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-t.C:
			}
		}
	}
	c.mu.Lock()
	c.lastCall[name] = time.Now()
	c.mu.Unlock()
	return nil
}

func (c *Chain) isOpen(name string) (bool, time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	b := c.breakers[name]
	if b == nil || b.openUntil.IsZero() {
		return false, time.Time{}
	}
	if time.Now().After(b.openUntil) {
		b.openUntil = time.Time{}
		b.fails = 0
		return false, time.Time{}
	}
	return true, b.openUntil
}

func (c *Chain) recordFailure(name string, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	b := c.breakers[name]
	if b == nil {
		b = &breaker{}
		c.breakers[name] = b
	}
	b.fails++
	if b.fails >= 3 || (err != nil && strings.Contains(strings.ToLower(err.Error()), "429")) {
		b.openUntil = time.Now().Add(5 * time.Minute)
	}
}

func (c *Chain) recordSuccess(name string) {
	c.mu.Lock()
	delete(c.breakers, name)
	c.mu.Unlock()
}

type cachedSearch struct {
	Backend string         `json:"backend"`
	Results []SearchResult `json:"results"`
}

// NormalizeQuery trims conversational filler and adds a current-year hint
// only for clearly time-sensitive questions.
func NormalizeQuery(q string) string {
	q = strings.TrimSpace(strings.Trim(q, `"'`))
	low := strings.ToLower(q)
	for _, prefix := range []string{"can you ", "could you ", "please ", "tell me ", "i want to know ", "i need to know "} {
		if strings.HasPrefix(low, prefix) {
			q = strings.TrimSpace(q[len(prefix):])
			low = strings.ToLower(q)
		}
	}
	q = strings.Join(strings.Fields(q), " ")
	words := strings.Fields(q)
	if len(words) > 12 {
		q = strings.Join(words[:12], " ")
	}
	if IsRecencyQuery(q) && !strings.Contains(q, fmt.Sprint(time.Now().Year())) {
		q += " " + fmt.Sprint(time.Now().Year())
	}
	return q
}

func simplifyQuery(q string) string {
	words := strings.Fields(q)
	if len(words) > 6 {
		words = words[:6]
	}
	return strings.Join(words, " ")
}

func IsRecencyQuery(q string) bool {
	low := strings.ToLower(q)
	return containsAny(low, "latest", "current", "today", "this year", "now", "news", "release notes", "changelog")
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
