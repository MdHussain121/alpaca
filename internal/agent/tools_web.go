package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/web"
)

type WebToolsConfig struct {
	Web        config.WebConfig
	ClientOpts web.ClientOptions
	CacheDir   string
	Chain      *web.Chain
	Session    *web.Session
}

// RegisterWebTools installs keyless web_search, web_fetch, and web_research.
// Session state lives on the Registry so the allowlist survives config reloads.
func RegisterWebTools(r *Registry, cfg WebToolsConfig) {
	if r == nil {
		return
	}
	if cfg.Session != nil {
		r.webSession = cfg.Session
	} else if r.webSession == nil {
		r.webSession = web.NewSession()
	}
	session := r.webSession

	if cfg.Chain != nil {
		r.webChain = cfg.Chain
	} else {
		r.webChain = web.NewKeylessChain(web.ChainOptions{
			ClientOpts: cfg.ClientOpts,
			CacheDir:   cfg.CacheDir,
			SearXNGURL: cfg.Web.SearXNGURL,
		})
	}
	chain := r.webChain
	cache := web.NewCache(cfg.CacheDir, web.PageCacheTTL)
	backends := strings.Join(chain.Backends(), ", ")
	r.SetWebSearchStatus("web_search active (keyless: " + backends + ")")

	fetchTool := ToolSpec{
		Name:        "web_fetch",
		Group:       "web",
		Class:       ClassNet,
		Description: "Fetch one numbered search result (preferred) or a URL the user provided. Never guess URLs. Example: {\"result\":1}",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"result":     map[string]any{"type": "integer", "description": "1-based result number from the latest web_search or web_research"},
				"url":        map[string]any{"type": "string", "description": "Only a URL from a search result or the user's message"},
				"max_tokens": map[string]any{"type": "integer", "description": "Excerpt budget, default 1500 (200-4000)"},
				"offset":     map[string]any{"type": "integer", "description": "Skip this many tokens from the start for a follow-up excerpt"},
			},
		},
		InspectPerm: func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
			target, summary, err := resolveFetchTarget(session, args)
			if err != nil {
				return PermissionInput{}, "", "", 0, 0, err
			}
			return PermissionInput{
				Tool:          "web_fetch",
				CanonicalArgs: map[string]any{"result": intArg(args, "result", 0), "url": target},
				Class:         ClassNet,
				InWorkspace:   false,
				Tainted:       true,
				Summary:       summary,
			}, "", "", 0, 0, nil
		},
		Execute: func(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
			target, summary, err := resolveFetchTarget(session, args)
			if err != nil {
				return ToolExecResult{Output: "Error: " + err.Error(), Summary: "invalid fetch", IsError: true}
			}
			session.BeginTurn(tctx.TurnNumber)
			if err := session.CheckFetch(target); err != nil {
				return ToolExecResult{Output: "Error: " + err.Error(), Summary: "fetch budget", IsError: true}
			}
			start := time.Now()
			res, err := web.Fetch(ctx, web.FetchOptions{
				URL:          target,
				Query:        session.Query(),
				MaxTokens:    clampInt(intArg(args, "max_tokens", 0), 0, 4000),
				OffsetTokens: clampInt(intArg(args, "offset", 0), 0, 200000),
				MaxBytes:     cfg.Web.MaxFetchBytes,
				ClientOpts:   cfg.ClientOpts,
				Cache:        cache,
			})
			session.RecordFetch(target, err)
			if err != nil {
				return ToolExecResult{
					Output:  "Error: " + web.ShortError(err.Error(), 300) + "\nNext: use another result or answer from what you already have.",
					Summary: summary + " · failed",
					IsError: true,
				}
			}
			session.AllowLinks(res.FinalURL, res.Links)
			return ToolExecResult{
				Output:  web.RedactSecrets(res.Wrapped, nil),
				Summary: fmt.Sprintf("%s · %s tok · %s", web.DomainOf(res.FinalURL), compactTokens(res.TokenEstimate), shortDuration(time.Since(start))),
				Tainted: true,
			}
		},
	}
	r.Register(fetchTool)

	r.Register(ToolSpec{
		Name:        "web_search",
		Group:       "web",
		Class:       ClassNet,
		Description: "Search the public web keylessly and return numbered results. Fetch one with web_fetch result=<n>. Example: {\"query\":\"go 1.27 release notes\"}",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":       map[string]any{"type": "string", "description": "Short keywords, not a full sentence"},
				"max_results": map[string]any{"type": "integer", "description": "1-10 results, default 5"},
			},
			"required": []string{"query"},
		},
		InspectPerm: func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
			q := strings.TrimSpace(strArg(args, "query"))
			if q == "" {
				return PermissionInput{}, "", "", 0, 0, fmt.Errorf("query cannot be empty")
			}
			return PermissionInput{
				Tool:          "web_search",
				CanonicalArgs: map[string]any{"query": q},
				Class:         ClassNet,
				InWorkspace:   false,
				Tainted:       true,
				Summary:       q,
			}, "", "", 0, 0, nil
		},
		Execute: func(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
			q := strings.TrimSpace(strArg(args, "query"))
			maxR := clampInt(intArg(args, "max_results", 5), 1, 10)
			report := chain.Search(ctx, q, maxR)
			if report.Err != nil || len(report.Results) == 0 {
				msg := "no results"
				if report.Err != nil {
					msg = report.Err.Error()
				}
				return ToolExecResult{
					Output:  "Error: " + web.ShortError(msg, 300) + "\nNext: try a shorter query or answer from what you know, saying you could not find it.",
					Summary: fmt.Sprintf("%q · no results", q),
					IsError: true,
				}
			}
			session.RememberSearch(q, report.Results)
			var sb strings.Builder
			fmt.Fprintf(&sb, "Search \"%s\" via %s (%d results):\n", q, report.Backend, len(report.Results))
			if note := backendNote(report); note != "" {
				sb.WriteString(note + "\n")
			}
			for i, r := range report.Results {
				fmt.Fprintf(&sb, "[%d] %s | %s | %s\n   URL: %s\n", i+1, r.Title, web.DomainOf(r.URL), web.ShortError(r.Snippet, 220), r.URL)
			}
			return ToolExecResult{
				Output:  web.WrapUntrustedContent(report.Backend+":search", sb.String()),
				Summary: fmt.Sprintf("%q · %s · %d results", q, report.Backend, len(report.Results)),
				Tainted: true,
			}
		},
	})

	r.Register(ToolSpec{
		Name:        "web_research",
		Group:       "web",
		Class:       ClassNet,
		Description: "Search and read the top 2-3 results in one call, returning short cited excerpts. Prefer this for most questions. Example: {\"query\":\"sqlite wal mode\"}",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":       map[string]any{"type": "string", "description": "Short keywords"},
				"max_results": map[string]any{"type": "integer", "description": "2-3 pages, default 3"},
				"max_tokens":  map[string]any{"type": "integer", "description": "Total excerpt budget, default 2000"},
			},
			"required": []string{"query"},
		},
		InspectPerm: func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
			q := strings.TrimSpace(strArg(args, "query"))
			if q == "" {
				return PermissionInput{}, "", "", 0, 0, fmt.Errorf("query cannot be empty")
			}
			return PermissionInput{
				Tool:          "web_research",
				CanonicalArgs: map[string]any{"query": q},
				Class:         ClassNet,
				InWorkspace:   false,
				Tainted:       true,
				Summary:       q,
			}, "", "", 0, 0, nil
		},
		Execute: func(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
			q := strings.TrimSpace(strArg(args, "query"))
			maxR := clampInt(intArg(args, "max_results", 3), 2, 3)
			session.BeginTurn(tctx.TurnNumber)
			start := time.Now()
			report := chain.Search(ctx, q, maxR+2)
			if report.Err != nil || len(report.Results) == 0 {
				msg := "no results"
				if report.Err != nil {
					msg = report.Err.Error()
				}
				return ToolExecResult{Output: "Error: " + web.ShortError(msg, 300), Summary: fmt.Sprintf("%q · no results", q), IsError: true}
			}
			session.RememberSearch(q, report.Results)
			n := maxR
			if n > len(report.Results) {
				n = len(report.Results)
			}
			totalBudget := intArg(args, "max_tokens", 0)
			if totalBudget <= 0 {
				ctxTokens := tctx.ContextTokens
				if ctxTokens <= 0 {
					ctxTokens = 8192
				}
				totalBudget = ctxTokens / 4
				if totalBudget < 900 {
					totalBudget = 900
				}
				if totalBudget > 4500 {
					totalBudget = 4500
				}
			}
			perPage := totalBudget / n
			if perPage > 1500 {
				perPage = 1500
			}
			if perPage < 200 {
				perPage = 200
			}

			type researchPage struct {
				idx  int
				res  web.FetchResult
				err  error
				dur  time.Duration
				item web.SearchResult
			}
			results := make([]researchPage, n)
			var wg sync.WaitGroup
			for i := 0; i < n; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					item := report.Results[i]
					start := time.Now()
					res, err := web.Fetch(ctx, web.FetchOptions{
						URL:        item.URL,
						Query:      q,
						MaxTokens:  perPage,
						MaxBytes:   cfg.Web.MaxFetchBytes,
						ClientOpts: cfg.ClientOpts,
						Cache:      cache,
					})
					results[i] = researchPage{idx: i, res: res, err: err, dur: time.Since(start), item: item}
					session.AllowLinks(item.URL, res.Links)
					session.RecordFetch(item.URL, err)
				}(i)
			}
			wg.Wait()

			var sb strings.Builder
			if note := backendNote(report); note != "" {
				sb.WriteString(note + "\n")
			}
			okCount := 0
			totalTokens := 0
			for _, page := range results {
				marker := "[unreadable]"
				if page.err == nil {
					marker = "[read]"
					okCount++
					totalTokens += page.res.TokenEstimate
				}
				fmt.Fprintf(&sb, "[%d] %s\n%s %s\nURL: %s\n", page.idx+1, page.item.Title, marker, web.DomainOf(page.item.URL), page.item.URL)
				if page.err != nil {
					fmt.Fprintf(&sb, "  Fetch failed: %s\n", web.ShortError(page.err.Error(), 180))
				} else {
					fmt.Fprintf(&sb, "%s\n", page.res.Body)
				}
				sb.WriteString("\n")
			}
			if okCount == 0 {
				return ToolExecResult{
					Output:  "Error: no result could be read.\n" + web.ShortError(sb.String(), 400) + "\nNext: try web_search again with shorter keywords.",
					Summary: fmt.Sprintf("%q · 0/%d pages", q, n),
					IsError: true,
				}
			}
			return ToolExecResult{
				Output:  web.WrapUntrustedContent(report.Backend+":research", sb.String()),
				Summary: fmt.Sprintf("%q · %s · %d/%d pages · %s tok · %s", q, report.Backend, okCount, n, compactTokens(totalTokens), shortDuration(time.Since(start))),
				Tainted: true,
			}
		},
	})
}

func backendNote(report web.SearchReport) string {
	var failed []string
	for _, a := range report.Attempts {
		if a.Error != "" {
			failed = append(failed, a.Backend+" ("+a.Error+")")
		}
	}
	if len(failed) == 0 {
		return ""
	}
	return "Note: " + strings.Join(failed, ", ") + "; used " + report.Backend + "."
}

func resolveFetchTarget(session *web.Session, args map[string]any) (string, string, error) {
	if n := intArg(args, "result", 0); n > 0 {
		item, ok := session.Result(n)
		if !ok {
			return "", "", fmt.Errorf("result %d is not in the latest search. Call web_search first and use one of its numbers.", n)
		}
		return item.URL, fmt.Sprintf("result %d · %s", n, web.DomainOf(item.URL)), nil
	}
	raw := strings.TrimSpace(strArg(args, "url"))
	if raw == "" {
		return "", "", fmt.Errorf("provide result=<n> from a search, or a URL the user provided")
	}
	if !session.IsAllowed(raw) {
		return "", "", fmt.Errorf("URL was not returned by a search and was not provided by the user. Call web_search first. Never guess URLs.")
	}
	canon, err := web.CanonicalHTTPURL(raw)
	if err != nil {
		return "", "", err
	}
	return canon, web.DomainOf(canon), nil
}

func compactTokens(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}

func shortDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
