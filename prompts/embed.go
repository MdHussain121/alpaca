package prompts

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/MdHussain121/alpaca/internal/runner"
)

//go:embed system.md system-compact.md
var embeddedFS embed.FS

// CompactThreshold is the context-window cutoff below which the compact prompt is chosen.
const CompactThreshold = 8192

var (
	unclosedTagRe = regexp.MustCompile(`\{\{[^{}]*\}\}`)
	blankLinesRe  = regexp.MustCompile(`\n{3,}`)
)

// RenderOptions holds runtime metadata interpolated into the built-in alpaca system prompt.
type RenderOptions struct {
	AppRoot        string
	Workspace      string
	PermissionMode string
	Date           string
	Tools          []runner.ToolDef
	SkillsList     string
	MCPSummary     string
	AgentsMD       string
	ContextSize    int
	CustomPrompt   string
	DisableDefault bool
}

// RenderResult describes the rendered prompt along with selection metadata.
type RenderResult struct {
	Text         string
	Compact      bool
	OverridePath string
	Disabled     bool
}

// DefaultTemplate returns the raw embedded full or compact template.
func DefaultTemplate(compact bool) string {
	name := "system.md"
	if compact {
		name = "system-compact.md"
	}
	b, err := embeddedFS.ReadFile(name)
	if err != nil {
		return ""
	}
	return string(b)
}

// OverridePath returns the on-disk system prompt override path beneath appRoot.
func OverridePath(appRoot string) string {
	if strings.TrimSpace(appRoot) == "" {
		return ""
	}
	return filepath.Join(appRoot, "prompts", "system.md")
}

// Render builds the complete system prompt from embedded templates (or on-disk override)
// and appends any custom suffix cleanly.
func Render(opts RenderOptions) RenderResult {
	custom := strings.TrimSpace(opts.CustomPrompt)
	skillsText := strings.TrimSpace(opts.SkillsList)
	agentsText := strings.TrimSpace(opts.AgentsMD)

	if opts.DisableDefault {
		var parts []string
		if custom != "" {
			parts = append(parts, custom)
		}
		if agentsText != "" {
			parts = append(parts, agentsText)
		}
		if skillsText != "" {
			parts = append(parts, skillsText)
		}
		return RenderResult{
			Text:     strings.Join(parts, "\n\n"),
			Disabled: true,
		}
	}

	compact := opts.ContextSize > 0 && opts.ContextSize < CompactThreshold
	tmpl, overridePath := loadTemplate(opts.AppRoot, compact)

	ws := strings.TrimSpace(opts.Workspace)
	if ws == "" {
		ws = "."
	}
	perm := strings.TrimSpace(opts.PermissionMode)
	if perm == "" {
		perm = "read-auto"
	}
	dateStr := strings.TrimSpace(opts.Date)
	if dateStr == "" {
		dateStr = time.Now().Format("2006-01-02")
	}
	ctxStr := "unknown"
	if opts.ContextSize > 0 {
		ctxStr = fmt.Sprintf("%d", opts.ContextSize)
	}

	builtinTools, derivedMCP, hasWeb := partitionTools(opts.Tools)
	toolsText := formatToolsList(builtinTools, compact)

	mcpText := strings.TrimSpace(opts.MCPSummary)
	if mcpText == "" {
		mcpText = derivedMCP
	}

	out := applySection(tmpl, "tools_section", toolsText != "")
	out = applySection(out, "web_section", hasWeb)
	out = applySection(out, "skills_section", skillsText != "")
	out = applySection(out, "mcp_section", mcpText != "")
	out = applySection(out, "agents_section", agentsText != "")

	repl := strings.NewReplacer(
		"{{workspace}}", ws,
		"{{permission_mode}}", perm,
		"{{date}}", dateStr,
		"{{tools_list}}", toolsText,
		"{{skills_list}}", skillsText,
		"{{mcp_summary}}", mcpText,
		"{{agents_md}}", agentsText,
		"{{context_size}}", ctxStr,
	)
	out = repl.Replace(out)
	out = unclosedTagRe.ReplaceAllString(out, "")
	out = blankLinesRe.ReplaceAllString(out, "\n\n")
	out = strings.TrimSpace(out)

	if custom != "" {
		if out != "" {
			out += "\n\n" + custom
		} else {
			out = custom
		}
	}

	return RenderResult{
		Text:         out,
		Compact:      compact,
		OverridePath: overridePath,
	}
}

func loadTemplate(appRoot string, compact bool) (string, string) {
	if root := strings.TrimSpace(appRoot); root != "" {
		var candidates []string
		if compact {
			candidates = append(candidates, filepath.Join(root, "prompts", "system-compact.md"))
		}
		candidates = append(candidates, filepath.Join(root, "prompts", "system.md"))
		for _, p := range candidates {
			if b, err := os.ReadFile(p); err == nil {
				if s := strings.TrimSpace(string(b)); s != "" {
					return string(b), p
				}
			}
		}
	}
	return DefaultTemplate(compact), ""
}

func applySection(src, tag string, keep bool) string {
	openTag := "{{#" + tag + "}}"
	closeTag := "{{/" + tag + "}}"
	for {
		start := strings.Index(src, openTag)
		if start < 0 {
			return src
		}
		end := strings.Index(src[start+len(openTag):], closeTag)
		if end < 0 {
			return strings.ReplaceAll(src, openTag, "")
		}
		bodyStart := start + len(openTag)
		bodyEnd := bodyStart + end
		closeEnd := bodyEnd + len(closeTag)
		if keep {
			src = src[:start] + src[bodyStart:bodyEnd] + src[closeEnd:]
		} else {
			src = src[:start] + src[closeEnd:]
		}
	}
}

func partitionTools(tools []runner.ToolDef) (builtins []runner.ToolDef, mcpSummary string, hasWeb bool) {
	mcpByServer := map[string][]string{}
	for _, td := range tools {
		name := strings.TrimSpace(td.Function.Name)
		if name == "" {
			continue
		}
		if name == "web_search" || name == "web_fetch" {
			hasWeb = true
		}
		if srv, _, ok := strings.Cut(name, "."); ok && srv != "" {
			mcpByServer[srv] = append(mcpByServer[srv], name)
			continue
		}
		builtins = append(builtins, td)
	}
	sort.Slice(builtins, func(i, j int) bool {
		return builtins[i].Function.Name < builtins[j].Function.Name
	})

	if len(mcpByServer) > 0 {
		servers := make([]string, 0, len(mcpByServer))
		for srv := range mcpByServer {
			servers = append(servers, srv)
		}
		sort.Strings(servers)
		parts := make([]string, 0, len(servers))
		for _, srv := range servers {
			names := mcpByServer[srv]
			sort.Strings(names)
			parts = append(parts, fmt.Sprintf("%s (%s)", srv, strings.Join(names, ", ")))
		}
		mcpSummary = strings.Join(parts, "; ")
	}
	return builtins, mcpSummary, hasWeb
}

func formatToolsList(tools []runner.ToolDef, compact bool) string {
	if len(tools) == 0 {
		return ""
	}
	if compact {
		names := make([]string, 0, len(tools))
		for _, td := range tools {
			names = append(names, td.Function.Name)
		}
		return strings.Join(names, ", ")
	}
	lines := make([]string, 0, len(tools))
	for _, td := range tools {
		desc := oneLineSummary(td.Function.Description, 140)
		if desc == "" {
			lines = append(lines, "- "+td.Function.Name)
		} else {
			lines = append(lines, fmt.Sprintf("- %s: %s", td.Function.Name, desc))
		}
	}
	return strings.Join(lines, "\n")
}

func oneLineSummary(s string, maxRunes int) string {
	s = strings.Join(strings.Fields(s), " ")
	if s == "" {
		return ""
	}
	if idx := strings.Index(s, ". "); idx > 0 && idx < maxRunes {
		s = s[:idx+1]
	}
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return strings.TrimSpace(string(runes[:maxRunes-3])) + "..."
}
