package web

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/MdHussain121/alpaca/internal/config"
)

type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// SearchProvider abstracts key-based REST search APIs and SearXNG instances.
type SearchProvider interface {
	Name() string
	Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error)
	Secrets() []string
}

type SearchConfigOptions struct {
	Web        config.WebConfig
	ClientOpts ClientOptions
}

// ResolveSearchProvider selects an active web search provider from config.toml or env vars.
// If nothing is configured, returns (nil, reason) so callers hide web_search from the model cleanly.
func ResolveSearchProvider(opts SearchConfigOptions) (SearchProvider, string) {
	prov := strings.ToLower(strings.TrimSpace(firstNonEmpty(
		opts.Web.SearchProvider,
		os.Getenv("ALPACA_WEB_SEARCH_PROVIDER"),
		os.Getenv("LLM_WEB_SEARCH_PROVIDER"),
	)))
	apiKey := strings.TrimSpace(firstNonEmpty(
		opts.Web.SearchAPIKey,
		os.Getenv("ALPACA_WEB_SEARCH_KEY"),
		os.Getenv("LLM_WEB_SEARCH_KEY"),
	))
	searxURL := strings.TrimSpace(firstNonEmpty(
		opts.Web.SearXNGURL,
		os.Getenv("SEARXNG_URL"),
		os.Getenv("ALPACA_SEARXNG_URL"),
		os.Getenv("LLM_SEARXNG_URL"),
	))
	braveEnv := strings.TrimSpace(os.Getenv("BRAVE_API_KEY"))
	tavilyEnv := strings.TrimSpace(os.Getenv("TAVILY_API_KEY"))

	if prov == "" {
		switch {
		case searxURL != "":
			prov = "searxng"
		case braveEnv != "" || (apiKey != "" && prov == ""):
			if braveEnv != "" {
				prov = "brave"
				apiKey = braveEnv
			} else if tavilyEnv != "" {
				prov = "tavily"
				apiKey = tavilyEnv
			} else {
				prov = "brave"
			}
		case tavilyEnv != "":
			prov = "tavily"
			apiKey = tavilyEnv
		}
	}

	switch prov {
	case "brave":
		key := firstNonEmpty(apiKey, braveEnv)
		if key == "" {
			return nil, "web_search provider 'brave' needs search_api_key in config.toml or BRAVE_API_KEY"
		}
		return &braveProvider{apiKey: key, httpClient: NewSafeHTTPClient(opts.ClientOpts)}, ""
	case "tavily":
		key := firstNonEmpty(apiKey, tavilyEnv)
		if key == "" {
			return nil, "web_search provider 'tavily' needs search_api_key in config.toml or TAVILY_API_KEY"
		}
		return &tavilyProvider{apiKey: key, httpClient: NewSafeHTTPClient(opts.ClientOpts)}, ""
	case "searxng":
		if searxURL == "" {
			return nil, "web_search provider 'searxng' needs searxng_url in config.toml or SEARXNG_URL"
		}
		u, err := url.Parse(searxURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return nil, fmt.Sprintf("invalid searxng_url %q", searxURL)
		}
		// User explicitly configures SearXNG URL in config.toml/env (which may be a self-hosted local instance).
		return &searxngProvider{baseURL: strings.TrimRight(searxURL, "/"), httpClient: &http.Client{Timeout: DefaultTimeout}}, ""
	case "":
		return nil, "no search provider configured (set [web] search_provider + search_api_key or searxng_url in config.toml, or BRAVE_API_KEY/TAVILY_API_KEY/SEARXNG_URL)"
	default:
		return nil, fmt.Sprintf("unsupported search_provider %q (supported: brave, tavily, searxng)", prov)
	}
}

// RedactSecrets replaces any non-empty secret strings inside s with [REDACTED].
func RedactSecrets(s string, secrets []string) string {
	out := s
	for _, sec := range secrets {
		sec = strings.TrimSpace(sec)
		if len(sec) >= 4 {
			out = strings.ReplaceAll(out, sec, "[REDACTED]")
		}
	}
	// Also scrub common env API keys defensively even if passed from another error source.
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

// --- Brave Search API ---

type braveProvider struct {
	apiKey     string
	baseURL    string // optional test override; defaults to https://api.search.brave.com
	httpClient *http.Client
}

func (b *braveProvider) Name() string      { return "brave" }
func (b *braveProvider) Secrets() []string { return []string{b.apiKey} }

func (b *braveProvider) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	if maxResults <= 0 || maxResults > 10 {
		maxResults = 5
	}
	base := b.baseURL
	if base == "" {
		base = "https://api.search.brave.com"
	}
	endpoint := fmt.Sprintf("%s/res/v1/web/search?q=%s&count=%d", strings.TrimRight(base, "/"), url.QueryEscape(query), maxResults)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", b.apiKey)

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, errors.New(RedactSecrets(err.Error(), b.Secrets()))
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("brave HTTP %d: %s", resp.StatusCode, RedactSecrets(strings.TrimSpace(string(raw)), b.Secrets()))
	}
	var payload struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Web.Results))
	for _, r := range payload.Web.Results {
		out = append(out, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Description})
	}
	return out, nil
}

// --- Tavily Search API ---

type tavilyProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func (t *tavilyProvider) Name() string      { return "tavily" }
func (t *tavilyProvider) Secrets() []string { return []string{t.apiKey} }

func (t *tavilyProvider) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	if maxResults <= 0 || maxResults > 10 {
		maxResults = 5
	}
	base := t.baseURL
	if base == "" {
		base = "https://api.tavily.com"
	}
	reqBody, _ := json.Marshal(map[string]any{
		"api_key":     t.apiKey,
		"query":       query,
		"max_results": maxResults,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(base, "/")+"/search", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil, errors.New(RedactSecrets(err.Error(), t.Secrets()))
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("tavily HTTP %d: %s", resp.StatusCode, RedactSecrets(strings.TrimSpace(string(raw)), t.Secrets()))
	}
	var payload struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Results))
	for _, r := range payload.Results {
		out = append(out, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Content})
	}
	return out, nil
}

// --- SearXNG JSON API ---

type searxngProvider struct {
	baseURL    string
	httpClient *http.Client
}

func (s *searxngProvider) Name() string      { return "searxng" }
func (s *searxngProvider) Secrets() []string { return nil }

func (s *searxngProvider) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	if maxResults <= 0 || maxResults > 10 {
		maxResults = 5
	}
	endpoint := fmt.Sprintf("%s/search?q=%s&format=json", s.baseURL, url.QueryEscape(query))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("searxng HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var payload struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if len(payload.Results) > maxResults {
		payload.Results = payload.Results[:maxResults]
	}
	out := make([]SearchResult, 0, len(payload.Results))
	for _, r := range payload.Results {
		out = append(out, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Content})
	}
	return out, nil
}
