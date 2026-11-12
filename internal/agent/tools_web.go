package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/web"
)

type WebToolsConfig struct {
	Web            config.WebConfig
	SearchProvider web.SearchProvider // optional explicit override (e.g. tests); auto-resolved from Web/env if nil
	ClientOpts     web.ClientOptions
}

// RegisterWebTools configures web_fetch (always in group "web") and web_search (only when configured).
// If no search provider is configured, web_search is unregistered so it is never offered to the model,
// and Registry.WebSearchStatus records why for /tools.
func RegisterWebTools(r *Registry, cfg WebToolsConfig) {
	if r == nil {
		return
	}

	var secrets []string
	if strings.TrimSpace(cfg.Web.SearchAPIKey) != "" {
		secrets = append(secrets, cfg.Web.SearchAPIKey)
	}

	prov := cfg.SearchProvider
	disabledReason := ""
	if prov == nil {
		prov, disabledReason = web.ResolveSearchProvider(web.SearchConfigOptions{
			Web:        cfg.Web,
			ClientOpts: cfg.ClientOpts,
		})
	}
	if prov != nil {
		secrets = append(secrets, prov.Secrets()...)
	}

	r.Register(ToolSpec{
		Name:        "web_fetch",
		Group:       "web",
		Class:       ClassNet,
		Description: "Fetch an HTTP/HTTPS web page, JSON, or plain text document with SSRF protection, returning readable Markdown wrapped as untrusted data.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"url":       map[string]any{"type": "string", "description": "HTTP or HTTPS URL to fetch"},
				"max_chars": map[string]any{"type": "integer", "description": "Maximum characters to return (default 20000)"},
			},
			"required": []string{"url"},
		},
		InspectPerm: func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
			u := strings.TrimSpace(strArg(args, "url"))
			if u == "" {
				return PermissionInput{}, "", "", 0, 0, fmt.Errorf("url cannot be empty")
			}
			maxC := intArg(args, "max_chars", 0)
			return PermissionInput{
				Tool:          "web_fetch",
				CanonicalArgs: map[string]any{"url": u, "max_chars": maxC},
				Class:         ClassNet,
				InWorkspace:   false,
				Summary:       u,
			}, "", "", 0, 0, nil
		},
		Execute: func(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
			u := strings.TrimSpace(strArg(args, "url"))
			maxC := intArg(args, "max_chars", 0)
			res, err := web.Fetch(ctx, web.FetchOptions{
				URL:        u,
				MaxChars:   maxC,
				MaxBytes:   cfg.Web.MaxFetchBytes,
				ClientOpts: cfg.ClientOpts,
			})
			if err != nil {
				return ToolExecResult{
					Output:  "Error: " + web.RedactSecrets(err.Error(), secrets),
					Summary: u,
					IsError: true,
				}
			}
			return ToolExecResult{
				Output:  web.RedactSecrets(res.Wrapped, secrets),
				Summary: fmt.Sprintf("%s (%dB)", res.FinalURL, res.ByteLen),
				Tainted: true,
			}
		},
	})

	if prov == nil {
		r.Unregister("web_search")
		r.SetWebSearchStatus("web_search disabled: " + disabledReason)
		return
	}

	r.SetWebSearchStatus(fmt.Sprintf("web_search active (%s)", prov.Name()))
	r.Register(ToolSpec{
		Name:        "web_search",
		Group:       "web",
		Class:       ClassNet,
		Description: "Search the web via the configured search provider, returning titles, URLs, and snippets wrapped as untrusted data.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":       map[string]any{"type": "string", "description": "Search query string"},
				"max_results": map[string]any{"type": "integer", "description": "Maximum number of results (1-10, default 5)"},
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
				Summary:       q,
			}, "", "", 0, 0, nil
		},
		Execute: func(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
			q := strings.TrimSpace(strArg(args, "query"))
			maxR := intArg(args, "max_results", 5)
			hits, err := prov.Search(ctx, q, maxR)
			if err != nil {
				return ToolExecResult{
					Output:  "Error: " + web.RedactSecrets(err.Error(), secrets),
					Summary: q,
					IsError: true,
				}
			}
			var sb strings.Builder
			if len(hits) == 0 {
				sb.WriteString("(no results)")
			} else {
				for i, h := range hits {
					fmt.Fprintf(&sb, "%d. %s\n   URL: %s\n   %s\n", i+1, h.Title, h.URL, strings.TrimSpace(h.Snippet))
				}
			}
			wrapped := web.WrapUntrustedContent(prov.Name()+":search", sb.String())
			return ToolExecResult{
				Output:  web.RedactSecrets(wrapped, secrets),
				Summary: fmt.Sprintf("%q (%d results)", q, len(hits)),
				Tainted: true,
			}
		},
	})
}
