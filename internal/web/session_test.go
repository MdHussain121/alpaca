package web

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSessionAllowlistFailureBudgetAndDuplicates(t *testing.T) {
	s := NewSession()
	s.SeedText("Read https://example.com/start, then summarize it.")
	if !s.IsAllowed("https://example.com/start") {
		t.Fatal("user-provided URL must be allowed")
	}
	if s.IsAllowed("https://invented.example/nope") {
		t.Fatal("guessed URL must not be allowed")
	}
	s.RememberSearch("go docs", []SearchResult{
		{Title: "Go", URL: "https://go.dev/doc/"},
		{Title: "Example", URL: "https://example.com/a"},
	})
	if !s.IsAllowed("https://go.dev/doc") {
		t.Fatal("search result URL must be allowed, including canonical slash variants")
	}
	if _, ok := s.Result(2); !ok {
		t.Fatal("result 2 must exist")
	}

	s.BeginTurn(1)
	target := "https://example.com/fail"
	if err := s.CheckFetch(target); err != nil {
		t.Fatalf("first fetch: %v", err)
	}
	s.RecordFetch(target, errors.New("boom"))
	if err := s.CheckFetch(target); err != nil {
		t.Fatalf("second fetch: %v", err)
	}
	s.RecordFetch(target, errors.New("boom"))
	if err := s.CheckFetch(target); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("expected exhausted budget, got %v", err)
	}

	s.BeginTurn(2)
	if err := s.CheckFetch("https://example.com/dup"); err != nil {
		t.Fatal(err)
	}
	if err := s.CheckFetch("https://example.com/dup"); err != nil {
		t.Fatal(err)
	}
	if err := s.CheckFetch("https://example.com/dup"); err == nil {
		t.Fatal("third identical fetch must be rejected")
	}
}

func TestRankDedupesAndPrefersPrimarySources(t *testing.T) {
	results := []SearchResult{
		{Title: "SEO farm", URL: "https://quora.com/What-is-Go", Snippet: "answer"},
		{Title: "Go docs mirror", URL: "https://go.dev/doc/", Snippet: "official go documentation"},
		{Title: "Go docs duplicate", URL: "https://go.dev/doc", Snippet: "duplicate"},
	}
	ranked := RankResults(results, "go documentation")
	if len(ranked) != 2 {
		t.Fatalf("expected dedupe to 2 results, got %d: %+v", len(ranked), ranked)
	}
	if ranked[0].Domain != "go.dev" {
		t.Fatalf("expected official docs first, got %+v", ranked)
	}
}

func TestNormalizeQueryStripsFillerAndAddsRecency(t *testing.T) {
	got := NormalizeQuery("Can you please tell me what is the latest Go release?")
	if strings.Contains(strings.ToLower(got), "can you") || strings.Contains(strings.ToLower(got), "please") {
		t.Fatalf("filler not removed: %q", got)
	}
	if !strings.Contains(got, time.Now().Format("2006")) {
		t.Fatalf("expected current-year hint, got %q", got)
	}
}

func TestTokenBudgetAndExcerptSelection(t *testing.T) {
	long := strings.Repeat("unrelated filler sentence. ", 2000) + "\nThe unique needle is sqlite WAL mode.\n" + strings.Repeat("tail. ", 500)
	out := SelectExcerpt(long, "sqlite wal mode", 120)
	if EstimateTokens(out) > 121 {
		t.Fatalf("excerpt exceeded budget: %d", EstimateTokens(out))
	}
	if !strings.Contains(out, "needle") {
		t.Fatalf("query-relevant passage missing:\n%s", out)
	}
	trunc := TruncateToTokens(long, 50)
	if EstimateTokens(trunc) > 51 {
		t.Fatalf("truncate exceeded budget: %d", EstimateTokens(trunc))
	}
}

func TestCacheTTL(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(dir, 50*time.Millisecond)
	c.PutJSON("k", cachedSearch{Backend: "x", Results: []SearchResult{{Title: "t"}}})
	var got cachedSearch
	if !c.GetJSON("k", &got) || got.Backend != "x" {
		t.Fatal("expected cache hit")
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "k.json"), old, old); err != nil {
		t.Fatal(err)
	}
	if c.GetJSON("k", &got) {
		t.Fatal("expired cache entry must miss")
	}
}

type fakeSearcher struct {
	name    string
	results []SearchResult
	err     error
	delay   time.Duration
	calls   int
}

func (f *fakeSearcher) Name() string { return f.name }

func (f *fakeSearcher) Search(ctx context.Context, query string, max int) ([]SearchResult, error) {
	f.calls++
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return f.results, f.err
}

func TestChainFallsThroughAndReportsAttempts(t *testing.T) {
	bad := &fakeSearcher{name: "bad", err: errors.New("boom")}
	empty := &fakeSearcher{name: "empty"}
	good := &fakeSearcher{name: "good", results: []SearchResult{{Title: "Go", URL: "https://go.dev/doc/"}}}
	chain := NewKeylessChain(ChainOptions{
		Backends: []Searcher{bad, empty, good},
		Cache:    NewCache(t.TempDir(), SearchCacheTTL),
		Timeout:  2 * time.Second,
	})
	report := chain.Search(context.Background(), "go docs", 3)
	if report.Err != nil || report.Backend != "good" || len(report.Results) != 1 {
		t.Fatalf("unexpected report: %+v", report)
	}
	if len(report.Attempts) < 3 {
		t.Fatalf("expected attempts for each backend, got %+v", report.Attempts)
	}
	// The second identical query should hit the cache and not call backends.
	before := good.calls
	again := chain.Search(context.Background(), "go docs", 3)
	if !again.FromCache || good.calls != before {
		t.Fatalf("expected cache hit, got cached=%v calls=%d", again.FromCache, good.calls)
	}
}

func TestChainTimeoutEndsWithError(t *testing.T) {
	slow := &fakeSearcher{name: "slow", delay: time.Second, results: []SearchResult{{Title: "late", URL: "https://example.com"}}}
	chain := NewKeylessChain(ChainOptions{
		Backends: []Searcher{slow},
		Cache:    NewCache(t.TempDir(), SearchCacheTTL),
		Timeout:  40 * time.Millisecond,
	})
	start := time.Now()
	report := chain.Search(context.Background(), "slow query", 3)
	if report.Err == nil {
		t.Fatal("expected a timeout error")
	}
	if time.Since(start) > 500*time.Millisecond {
		t.Fatalf("search did not honor timeout: %s", time.Since(start))
	}
}
