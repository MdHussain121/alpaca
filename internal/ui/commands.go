package ui

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/session"
)

// slashCommand defines a chat slash command in the shared command registry.
type slashCommand struct {
	Name          string
	Aliases       []string
	Description   string
	ArgsHint      string
	RequiresArg   bool
	Enabled       func(m *Model) (bool, string)
	ArgCandidates func(m *Model, argPrefix string) []popupItem
	Run           func(m *Model, arg string) (tea.Model, tea.Cmd)
}

func (m *Model) commandRegistry() []slashCommand {
	thinkArgs := "[show|hide]"
	if m.thinkToggleSupported() {
		thinkArgs = "[on|off]"
	}

	return []slashCommand{
		{
			Name:        "help",
			Aliases:     []string{"?"},
			Description: "Show available commands and shortcuts",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				m.showChatHelp()
				return m, nil
			},
		},
		{
			Name:        "clear",
			Aliases:     []string{"reset"},
			Description: "Clear chat messages and context meter",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				m.resetChat()
				m.toastf("chat cleared")
				return m, nil
			},
		},
		{
			Name:        "stop",
			Aliases:     []string{"cancel"},
			Description: "Stop active generation or running server",
			Enabled: func(m *Model) (bool, string) {
				if m.chatStreaming || (m.proc != nil && !m.proc.Exited()) {
					return true, ""
				}
				return false, "nothing running"
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				m.stopChatStream()
				if m.proc != nil && !m.proc.Exited() {
					m.status = "STOPPING"
					m.statusDe = "stopping…"
					return m, stopCmd(m.proc, "stop")
				}
				if m.status != "READY" {
					m.status = "STOPPED"
				}
				m.toastf("generation stopped")
				return m, nil
			},
		},
		{
			Name:        "quit",
			Aliases:     []string{"exit", "q"},
			Description: "Stop llama-server and quit Launcher",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.forceQuit()
			},
		},
		{
			Name:        "theme",
			Description: "Switch UI color theme (or cycle without args)",
			ArgsHint:    "[name]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				themes := AllThemes()
				items := make([]popupItem, 0, len(themes))
				for _, th := range themes {
					desc := "theme"
					if strings.EqualFold(th.Name, m.sty.t.Name) {
						desc = "active theme"
					}
					items = append(items, popupItem{
						ID:          th.Name,
						Label:       th.Name,
						Description: desc,
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runThemeCommand(arg)
			},
		},
		{
			Name:        "copy",
			Description: "Copy latest code block, block #N, or last reply",
			ArgsHint:    "[n|reply]",
			Enabled: func(m *Model) (bool, string) {
				if len(m.chatMsgs) == 0 {
					return false, "no messages yet"
				}
				for _, t := range m.chatMsgs {
					if t.Role == "assistant" && strings.TrimSpace(t.Content) != "" {
						return true, ""
					}
				}
				return false, "no assistant output"
			},
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				blocks := m.allCodeBlocks()
				items := make([]popupItem, 0, len(blocks)+1)
				items = append(items, popupItem{
					ID:          "reply",
					Label:       "reply",
					Description: "Copy last assistant reply",
				})
				for i, b := range blocks {
					lang := b.Lang
					if lang == "" {
						lang = "code"
					}
					lineCnt := strings.Count(b.Code, "\n") + 1
					items = append(items, popupItem{
						ID:          strconv.Itoa(i + 1),
						Label:       strconv.Itoa(i + 1),
						Description: fmt.Sprintf("%s block (%d lines)", lang, lineCnt),
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runCopyCommand(arg)
			},
		},
		{
			Name:        "think",
			Aliases:     []string{"thinking"},
			Description: "Toggle thinking trace display or enable_thinking",
			ArgsHint:    thinkArgs,
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				items := make([]popupItem, 0, 4)
				if m.thinkToggleSupported() {
					items = append(items,
						popupItem{ID: "on", Label: "on", Description: "Enable thinking on next turn"},
						popupItem{ID: "off", Label: "off", Description: "Disable thinking on next turn"},
					)
				} else {
					items = append(items,
						popupItem{ID: "on", Label: "on", Description: "Not supported by model template", Disabled: true, DisabledReason: "unsupported"},
						popupItem{ID: "off", Label: "off", Description: "Not supported by model template", Disabled: true, DisabledReason: "unsupported"},
					)
				}
				items = append(items,
					popupItem{ID: "show", Label: "show", Description: "Expand thinking blocks in chat"},
					popupItem{ID: "hide", Label: "hide", Description: "Collapse thinking blocks in chat"},
				)
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runThinkCommand(arg)
			},
		},
		{
			Name:        "model",
			Aliases:     []string{"models"},
			Description: "Switch to another discovered GGUF model",
			ArgsHint:    "<name>",
			RequiresArg: true,
			Enabled: func(m *Model) (bool, string) {
				if len(m.models) == 0 {
					return false, "no models found"
				}
				return true, ""
			},
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				items := make([]popupItem, 0, len(m.models))
				curPath := ""
				if m.meta != nil {
					curPath = m.meta.Path
				}
				for _, e := range m.models {
					name := e.cand.RelPath
					if e.meta != nil && e.meta.DisplayName() != "" {
						name = e.meta.DisplayName()
					}
					desc := fmt.Sprintf("%.1f GB", float64(e.cand.Size)/(1<<30))
					if e.meta != nil && e.meta.Path == curPath {
						desc += " (active)"
					}
					items = append(items, popupItem{
						ID:          e.cand.RelPath,
						Label:       name,
						Description: desc,
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runModelCommand(arg)
			},
		},
		{
			Name:        "preset",
			Aliases:     []string{"presets"},
			Description: "Apply a saved configuration preset",
			ArgsHint:    "<name>",
			RequiresArg: true,
			Enabled: func(m *Model) (bool, string) {
				if len(m.loadPresetsFast()) == 0 {
					return false, "no saved presets"
				}
				return true, ""
			},
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				ps := m.loadPresetsFast()
				items := make([]popupItem, 0, len(ps))
				for _, p := range ps {
					items = append(items, popupItem{
						ID:          p.Name,
						Label:       p.Name,
						Description: fmt.Sprintf("ctx %d · kv %s · temp %.2f", p.Settings.Context, p.Settings.VCache, p.Settings.Temp),
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runPresetCommand(arg)
			},
		},
		{
			Name:        "effort",
			Description: "Set reasoning effort level",
			ArgsHint:    "[level]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				levels := flags.ReasoningEffortOptions()
				if m.modelCaps.Loaded && len(m.modelCaps.ReasoningEffortLevels) > 0 {
					levels = append([]string{"default"}, m.modelCaps.ReasoningEffortLevels...)
				}
				items := make([]popupItem, 0, len(levels)+2)
				for _, lv := range levels {
					items = append(items, popupItem{ID: lv, Label: lv, Description: "reasoning_effort"})
				}
				items = append(items,
					popupItem{ID: "on", Label: "on", Description: "Enable reasoning"},
					popupItem{ID: "off", Label: "off", Description: "Disable reasoning"},
				)
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runEffortCommand(arg)
			},
		},
		{
			Name:        "lines",
			Description: "Toggle line numbers inside code blocks",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				m.codeLineNums = !m.codeLineNums
				state := "on"
				if !m.codeLineNums {
					state = "off"
				}
				m.toastf("code line numbers: %s", state)
				m.refreshChatView(false)
				return m, nil
			},
		},
		{
			Name:        "system",
			Description: "Inspect, toggle, or customize the session system prompt",
			ArgsHint:    "[show|off|on|reset|clear|<prompt>]",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				trimmed := strings.TrimSpace(arg)
				sub := strings.ToLower(trimmed)
				switch sub {
				case "", "show":
					m.ensureAgentReady()
					res := m.effectivePromptRenderResult(true)
					mode := "built-in full"
					if res.Disabled {
						mode = "built-in OFF (disable_system_prompt=true)"
					} else if res.Compact {
						mode = "built-in compact (<8K)"
					}
					if res.OverridePath != "" {
						mode += " [override: " + res.OverridePath + "]"
					}
					custom := strings.TrimSpace(m.settings.SystemPrompt)
					if custom == "" {
						custom = "(none)"
					}
					estTok := (len(res.Text) + 3) / 4
					var b strings.Builder
					b.WriteString(fmt.Sprintf("System prompt: %s · custom: %s · ~%d tokens\n", mode, custom, estTok))
					b.WriteString("Toggle with `/system off` / `/system on`, set suffix `/system <text>`, or `/system reset`.\n\n")
					if res.Text == "" {
						b.WriteString("(empty)")
					} else {
						b.WriteString("```markdown\n" + res.Text + "\n```")
					}
					m.appendInfoTurn(b.String())
					return m, nil
				case "off":
					if m.cfg != nil {
						m.cfg.DisableSystemPrompt = true
						_ = m.cfg.Save()
					}
					m.toastf("built-in system prompt disabled")
				case "on":
					if m.cfg != nil {
						m.cfg.DisableSystemPrompt = false
						_ = m.cfg.Save()
					}
					m.toastf("built-in system prompt enabled")
				case "clear":
					m.settings.SystemPrompt = ""
					m.toastf("custom system prompt cleared")
				case "reset":
					m.settings.SystemPrompt = ""
					if m.cfg != nil {
						m.cfg.DisableSystemPrompt = false
						_ = m.cfg.Save()
					}
					m.toastf("system prompt reset to built-in default")
				default:
					m.settings.SystemPrompt = trimmed
					m.toastf("custom system prompt updated")
				}
				m.refreshChatView(false)
				return m, nil
			},
		},
		{
			Name:        "log",
			Aliases:     []string{"logs"},
			Description: "Switch to the llama-server log tab",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				m.runTab = runTabLog
				m.chatInput.Blur()
				return m, nil
			},
		},
		{
			Name:        "restart",
			Description: "Restart llama-server with current settings",
			Enabled: func(m *Model) (bool, string) {
				if m.meta == nil {
					return false, "no model selected"
				}
				return true, ""
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				m.stopChatStream()
				if m.proc != nil && !m.proc.Exited() {
					m.status = "STOPPING"
					m.statusDe = "restarting…"
					return m, stopCmd(m.proc, "restart")
				}
				return m.launch()
			},
		},
		{
			Name:        "tools",
			Description: "Inspect or toggle tool groups (files, exec, web, mcp)",
			ArgsHint:    "[files|exec|web|skills|mcp [on|off]]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				m.ensureAgentReady()
				var items []popupItem
				for _, g := range m.toolRegistry.GroupsOrdered() {
					st := "off"
					if g.Enabled {
						st = "on"
					}
					items = append(items,
						popupItem{ID: g.Name, Label: g.Name, Description: fmt.Sprintf("Toggle %s group (%d tools, %s)", g.Name, g.Count, st)},
						popupItem{ID: g.Name + " on", Label: g.Name + " on", Description: fmt.Sprintf("Enable %s tools", g.Name)},
						popupItem{ID: g.Name + " off", Label: g.Name + " off", Description: fmt.Sprintf("Disable %s tools", g.Name)},
					)
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runToolsCommand(arg)
			},
		},
		{
			Name:        "mcp",
			Description: "Inspect stdio MCP servers, toggle tools, or reload mcp.toml",
			ArgsHint:    "[reload]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				return []popupItem{
					{ID: "reload", Label: "reload", Description: "Reload mcp.toml and restart MCP servers"},
				}
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runMCPCommand(arg)
			},
		},
		{
			Name:        "permissions",
			Aliases:     []string{"perm", "perms"},
			Description: "Set permission mode, clear allowlist, or clear taint",
			ArgsHint:    "[ask|read-auto|workspace-auto|clear|clear-taint]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				return []popupItem{
					{ID: "ask", Label: "ask", Description: "Prompt before read, write, or exec"},
					{ID: "read-auto", Label: "read-auto", Description: "Auto-allow workspace reads; ask write/exec"},
					{ID: "workspace-auto", Label: "workspace-auto", Description: "Auto-allow workspace read+write; ask exec"},
					{ID: "clear", Label: "clear", Description: "Clear session tool allowlist"},
					{ID: "clear-taint", Label: "clear-taint", Description: "Clear TAINTED web session flag"},
				}
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runPermissionsCommand(arg)
			},
		},
		{
			Name:        "workspace",
			Aliases:     []string{"ws", "cd"},
			Description: "Show or switch the active sandbox workspace directory",
			ArgsHint:    "[path]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				root := m.workspaceDir()
				entries, err := os.ReadDir(root)
				if err != nil {
					return nil
				}
				var items []popupItem
				if parent := filepath.Dir(root); parent != "" && parent != root {
					items = append(items, popupItem{ID: parent, Label: "..", Description: parent})
				}
				for _, de := range entries {
					if de.IsDir() && !strings.HasPrefix(de.Name(), ".") {
						full := filepath.Join(root, de.Name())
						items = append(items, popupItem{ID: full, Label: de.Name() + "/", Description: full})
					}
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runWorkspaceCommand(arg)
			},
		},
		{
			Name:        "undo",
			Description: "Restore files changed in the last agent turn",
			Enabled: func(m *Model) (bool, string) {
				m.ensureAgentReady()
				if m.checkpoints != nil {
					if _, paths, ok := m.checkpoints.PendingUndoSummary(); ok && len(paths) > 0 {
						return true, ""
					}
				}
				return false, "nothing to undo"
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runUndoCommand()
			},
		},
		{
			Name:        "skills",
			Description: "Manage Agent Skills (enable/disable, view, create, rename, delete, stats)",
			ArgsHint:    "[list|reload]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				return []popupItem{
					{ID: "list", Label: "list", Description: "Show summary of valid and invalid skills in chat"},
					{ID: "reload", Label: "reload", Description: "Reload skills/ catalog from disk"},
				}
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runSkillsCommand(arg)
			},
		},
		{
			Name:        "skill",
			Description: "Invoke a skill by name directly in chat",
			ArgsHint:    "<name> [instructions]",
			RequiresArg: true,
			Enabled: func(m *Model) (bool, string) {
				m.ensureAgentReady()
				for _, s := range m.skillsCatalog.Valid {
					if s.Enabled {
						return true, ""
					}
				}
				return false, "no enabled skills"
			},
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				m.ensureAgentReady()
				var items []popupItem
				for _, s := range m.skillsCatalog.Valid {
					desc := s.Description
					if !s.Enabled {
						items = append(items, popupItem{
							ID:             s.Name,
							Label:          s.Name,
							Description:    desc,
							Disabled:       true,
							DisabledReason: "disabled",
						})
						continue
					}
					if s.ModelAuthored {
						desc = "[model] " + desc
					}
					items = append(items, popupItem{
						ID:          s.Name,
						Label:       s.Name,
						Description: desc,
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runUseSkillCommand(arg)
			},
		},
		{
			Name:        "context",
			Aliases:     []string{"ctx"},
			Description: "Inspect token breakdown across system, AGENTS.md, tools, skills, attachments, and history",
			ArgsHint:    "[print|compact]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				return []popupItem{
					{ID: "print", Label: "print", Description: "Print token breakdown into chat"},
					{ID: "compact", Label: "compact", Description: "Run tiered conversation compaction"},
				}
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runContextCommand(arg)
			},
		},
		{
			Name:        "compact",
			Description: "Tiered compaction: stub old tool outputs & summarize older turns",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runCompactCommand(arg)
			},
		},
		{
			Name:        "profile",
			Description: "Switch active sampling profile (coding, precise, creative, custom)",
			ArgsHint:    "<name>",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				if m.cfg == nil {
					return nil
				}
				profiles := m.cfg.ResolvedProfiles()
				var items []popupItem
				for _, name := range m.cfg.ProfileNames() {
					p := profiles[name]
					items = append(items, popupItem{
						ID:          name,
						Label:       name,
						Description: fmt.Sprintf("temp=%.2f top_p=%.2f — %s", p.Temp, p.TopP, p.Description),
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runProfileCommand(arg)
			},
		},
		{
			Name:        "session",
			Aliases:     []string{"sessions"},
			Description: "Manage JSONL sessions (new, list, search, load, rename, delete, export, branch)",
			ArgsHint:    "[new|list|search <q>|load <id>|rename <t>|delete <id>|export|branch]",
			ArgCandidates: func(m *Model, argPrefix string) []popupItem {
				items := []popupItem{
					{ID: "new", Label: "new", Description: "Start a fresh session"},
					{ID: "list", Label: "list", Description: "List saved sessions in chat"},
					{ID: "export", Label: "export", Description: "Export active branch to Markdown"},
					{ID: "branch", Label: "branch", Description: "List or switch conversation branches"},
				}
				sums, _ := session.ListSessions(m.effectiveSessionsDir())
				for i, s := range sums {
					if i >= 6 {
						break
					}
					items = append(items, popupItem{
						ID:          "load " + s.ID,
						Label:       "load " + s.ID,
						Description: fmt.Sprintf("%s (%d turns)", s.Title, s.TurnCount),
					})
				}
				return items
			},
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runSessionCommand(arg)
			},
		},
		{
			Name:        "edit",
			Description: "Edit the last user prompt on a new branch while preserving the old branch",
			ArgsHint:    "<new prompt>",
			RequiresArg: true,
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runEditTurnCommand(arg)
			},
		},
		{
			Name:        "regen",
			Aliases:     []string{"regenerate", "retry"},
			Description: "Regenerate the last assistant reply on a new branch",
			Run: func(m *Model, arg string) (tea.Model, tea.Cmd) {
				return m.runRegenerateCommand()
			},
		},
	}
}

func (m *Model) loadPresetsFast() []config.Preset {
	if len(m.presets) > 0 {
		return m.presets
	}
	if m.cfg == nil {
		return nil
	}
	if ps, err := m.cfg.ListPresets(); err == nil && len(ps) > 0 {
		m.presets = ps
	}
	return m.presets
}

func (m *Model) lookupCommand(name string) (slashCommand, bool) {
	q := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(name), "/"))
	for _, c := range m.commandRegistry() {
		if c.Name == q {
			return c, true
		}
		for _, al := range c.Aliases {
			if al == q {
				return c, true
			}
		}
	}
	return slashCommand{}, false
}

func (m *Model) handleChatSlash(raw string) (tea.Model, tea.Cmd) {
	fields := strings.Fields(raw)
	if len(fields) == 0 {
		return m, nil
	}
	cmdToken := strings.ToLower(fields[0])
	arg := ""
	if len(fields) > 1 {
		arg = strings.TrimSpace(strings.TrimPrefix(raw, fields[0]))
	}

	if cmd, ok := m.lookupCommand(cmdToken); ok {
		if cmd.Enabled != nil {
			if enabled, reason := cmd.Enabled(m); !enabled {
				m.appendInfoTurn(fmt.Sprintf("/%s is currently disabled: %s.", cmd.Name, reason))
				return m, nil
			}
		}
		return cmd.Run(m, arg)
	}

	closest := findClosestCommand(m.commandRegistry(), strings.TrimPrefix(cmdToken, "/"))
	if closest != "" {
		m.appendInfoTurn(fmt.Sprintf("Unknown command %q. Did you mean /%s? Type /help for commands.", cmdToken, closest))
	} else {
		m.appendInfoTurn(fmt.Sprintf("Unknown command %q. Type /help for chat commands.", cmdToken))
	}
	return m, nil
}

func findClosestCommand(cmds []slashCommand, typed string) string {
	typed = strings.ToLower(strings.TrimSpace(typed))
	if typed == "" {
		return ""
	}
	bestName := ""
	bestDist := 999
	for _, c := range cmds {
		if strings.HasPrefix(c.Name, typed) {
			return c.Name
		}
		if d := levenshteinDist(typed, c.Name); d < bestDist {
			bestDist = d
			bestName = c.Name
		}
	}
	if bestDist <= 3 || (len(typed) >= 3 && bestDist <= len(typed)/2+1) {
		return bestName
	}
	return ""
}

func levenshteinDist(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	if len(ra) == 0 {
		return len(rb)
	}
	if len(rb) == 0 {
		return len(ra)
	}
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			m := prev[j] + 1
			if cur[j-1]+1 < m {
				m = cur[j-1] + 1
			}
			if prev[j-1]+cost < m {
				m = prev[j-1] + cost
			}
			cur[j] = m
		}
		prev, cur = cur, prev
	}
	return prev[len(rb)]
}

func (m *Model) showChatHelp() {
	var b strings.Builder
	b.WriteString("Chat commands:")
	for _, c := range m.commandRegistry() {
		if c.Name == "think" && !m.thinkToggleSupported() {
			b.WriteString("\n  /think [show|hide] — Toggle thinking display (per-turn enable_thinking unsupported by model)")
			continue
		}
		line := "  /" + c.Name
		if c.ArgsHint != "" {
			line += " " + c.ArgsHint
		}
		b.WriteString("\n" + line + " — " + c.Description)
	}
	b.WriteString("\nShortcuts: ctrl+p palette · enter send · tab complete/switch · ↑↓ scroll · esc stop/back · ctrl+e effort · ctrl+t thinking · ctrl+l clear · ctrl+c quit.")
	m.appendInfoTurn(b.String())
}

func (m *Model) runThemeCommand(arg string) (tea.Model, tea.Cmd) {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		m.cycleTheme(1)
		m.refreshChatView(false)
		return m, saveConfigCmd(m.cfg.Clone())
	}
	q := strings.ToLower(arg)
	for _, th := range AllThemes() {
		if strings.EqualFold(th.Name, q) || strings.HasPrefix(strings.ToLower(th.Name), q) {
			m.cfg.Theme = th.Name
			m.sty = newStyles(th, m.cfg.ASCII)
			m.spinner.Style = m.sty.accent()
			m.toastf("theme: %s", th.Name)
			m.refreshChatView(false)
			return m, saveConfigCmd(m.cfg.Clone())
		}
	}
	m.appendInfoTurn(fmt.Sprintf("Unknown theme %q. Available: %s", arg, strings.Join(ThemeNames(), ", ")))
	return m, nil
}

func (m *Model) runCopyCommand(arg string) (tea.Model, tea.Cmd) {
	argL := strings.ToLower(strings.TrimSpace(arg))
	if argL == "reply" || argL == "last" {
		for i := len(m.chatMsgs) - 1; i >= 0; i-- {
			if m.chatMsgs[i].Role == "assistant" && strings.TrimSpace(m.chatMsgs[i].Content) != "" {
				return m, copyCmd(m.chatMsgs[i].Content)
			}
		}
		m.toastf("no assistant reply to copy")
		return m, nil
	}
	if n, err := strconv.Atoi(argL); err == nil && argL != "" {
		blocks := m.allCodeBlocks()
		if n < 1 || n > len(blocks) {
			m.toastf("code block %d not found (%d available)", n, len(blocks))
			return m, nil
		}
		return m, copyCmd(blocks[n-1].Code)
	}
	if cmd := m.copyLatestCodeOrReply(); cmd != nil {
		return m, cmd
	}
	return m, nil
}

func (m *Model) runThinkCommand(arg string) (tea.Model, tea.Cmd) {
	argL := strings.ToLower(strings.TrimSpace(arg))
	switch argL {
	case "on", "off":
		if !m.thinkToggleSupported() {
			m.appendInfoTurn("/think on|off is not supported by this model's chat template (thinking cannot be disabled per request).")
			return m, nil
		}
		if argL == "off" {
			m.settings.Reasoning = "off"
			m.toastf("thinking disabled for next prompt")
		} else {
			m.settings.Reasoning = "on"
			m.toastf("thinking enabled for next prompt")
		}
		return m, nil
	case "show", "hide":
		wantExpanded := argL == "show"
		m.settings.HideThinking = !wantExpanded
		for i := range m.chatMsgs {
			t := &m.chatMsgs[i]
			if t.Role == "assistant" && t.Reasoning != "" {
				v := wantExpanded
				t.ThinkExpanded = &v
			}
		}
		m.toastf("thinking display: %s", argL)
		m.refreshChatView(false)
		return m, nil
	default:
		m.toggleThinkingDisplay()
		state := "shown"
		if m.settings.HideThinking {
			state = "collapsed"
		}
		m.toastf("thinking display: %s", state)
		m.refreshChatView(false)
		return m, nil
	}
}

func (m *Model) runEffortCommand(arg string) (tea.Model, tea.Cmd) {
	if arg == "" {
		m.cycleReasoningEffortLive()
		return m, nil
	}
	lvl := strings.ToLower(strings.TrimSpace(arg))
	switch lvl {
	case "off", "disable", "disabled":
		m.settings.Reasoning = "off"
		m.toastf("reasoning disabled for next prompt")
		return m, nil
	case "on", "enable", "enabled":
		m.settings.Reasoning = "on"
		m.toastf("reasoning enabled")
		return m, nil
	case "auto":
		m.settings.Reasoning = "auto"
		m.settings.ReasoningEffort = "default"
		m.toastf("reasoning effort: default")
		return m, nil
	}
	for _, opt := range flags.ReasoningEffortOptions() {
		if opt == lvl {
			m.settings.Reasoning = "auto"
			m.settings.ReasoningEffort = opt
			m.toastf("reasoning effort: %s", opt)
			return m, nil
		}
	}
	m.toastf("unknown effort %q (use: default, minimal, low, medium, high, xhigh, max, off)", arg)
	return m, nil
}

func (m *Model) runModelCommand(arg string) (tea.Model, tea.Cmd) {
	q := strings.ToLower(strings.TrimSpace(arg))
	if q == "" {
		if len(m.models) == 0 {
			m.appendInfoTurn("No GGUF models discovered.")
			return m, nil
		}
		var names []string
		for _, e := range m.models {
			names = append(names, e.cand.RelPath)
		}
		m.appendInfoTurn("Available models: " + strings.Join(names, ", ") + " — run /model <name> to switch.")
		return m, nil
	}

	targetIdx := -1
	for i, e := range m.models {
		rel := strings.ToLower(e.cand.RelPath)
		disp := rel
		if e.meta != nil && e.meta.DisplayName() != "" {
			disp = strings.ToLower(e.meta.DisplayName())
		}
		if rel == q || disp == q || strings.EqualFold(filepath.Base(rel), q) {
			targetIdx = i
			break
		}
	}
	if targetIdx < 0 {
		for i, e := range m.models {
			rel := strings.ToLower(e.cand.RelPath)
			disp := rel
			if e.meta != nil {
				disp = strings.ToLower(e.meta.DisplayName())
			}
			if strings.Contains(rel, q) || strings.Contains(disp, q) {
				targetIdx = i
				break
			}
		}
	}
	if targetIdx < 0 {
		m.appendInfoTurn(fmt.Sprintf("Model %q not found in discovered models.", arg))
		return m, nil
	}

	e := m.models[targetIdx]
	m.filter = ""
	m.applyFilter()
	for visPos, origIdx := range m.visible {
		if origIdx == targetIdx {
			m.cursor = visPos
			break
		}
	}
	m.selectCurrent()

	displayName := e.cand.RelPath
	if m.meta != nil && m.meta.DisplayName() != "" {
		displayName = m.meta.DisplayName()
	}

	wasRunning := m.proc != nil && !m.proc.Exited()
	m.stopChatStream()
	if wasRunning {
		_ = m.proc.StopTimeout(0)
		m.proc = nil
	}
	exe := filepath.Join(m.root, "bin", m.backendDir(), "llama-server.exe")
	if _, err := os.Stat(exe); err == nil && m.meta != nil {
		m.toastf("switching to %s…", displayName)
		return m.launch()
	}
	m.toastf("selected model %s", displayName)
	m.refreshChatView(false)
	return m, nil
}

func (m *Model) runPresetCommand(arg string) (tea.Model, tea.Cmd) {
	ps := m.loadPresetsFast()
	if len(ps) == 0 {
		m.appendInfoTurn("No saved presets found.")
		return m, nil
	}
	q := strings.ToLower(strings.TrimSpace(arg))
	if q == "" {
		var names []string
		for _, p := range ps {
			names = append(names, p.Name)
		}
		m.appendInfoTurn("Saved presets: " + strings.Join(names, ", ") + " — run /preset <name>.")
		return m, nil
	}

	var matched *config.Preset
	for i := range ps {
		if strings.EqualFold(ps[i].Name, q) {
			matched = &ps[i]
			break
		}
	}
	if matched == nil {
		for i := range ps {
			if strings.Contains(strings.ToLower(ps[i].Name), q) {
				matched = &ps[i]
				break
			}
		}
	}
	if matched == nil {
		m.appendInfoTurn(fmt.Sprintf("Preset %q not found.", arg))
		return m, nil
	}

	m.settings = matched.Settings
	m.clampToModel()
	m.recompute()
	m.toastf("loaded preset %q", matched.Name)
	m.refreshChatView(false)
	return m, nil
}

func (m *Model) workspaceDir() string {
	if strings.TrimSpace(m.agentWorkDir) != "" {
		return m.agentWorkDir
	}
	if strings.TrimSpace(m.root) != "" {
		return m.root
	}
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

func (m *Model) runToolsCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	fields := strings.Fields(strings.ToLower(strings.TrimSpace(arg)))
	if len(fields) == 0 {
		m.toolsModal = &toolsModalState{}
		return m, m.measureToolSchemaTokensCmd()
	}
	group := fields[0]
	var next bool
	if len(fields) == 1 {
		next = !m.toolRegistry.GroupEnabled(group)
	} else {
		switch fields[1] {
		case "on", "enable", "true", "1":
			next = true
		case "off", "disable", "false", "0":
			next = false
		default:
			m.appendInfoTurn(fmt.Sprintf("Usage: /tools [files|exec [on|off]] (got %q)", arg))
			return m, nil
		}
	}
	known := false
	for _, g := range m.toolRegistry.GroupsOrdered() {
		if g.Name == group {
			known = true
			break
		}
	}
	if !known {
		m.appendInfoTurn(fmt.Sprintf("Unknown tool group %q (available: files, exec).", group))
		return m, nil
	}
	m.toolRegistry.SetGroupEnabled(group, next)
	if group == "exec" && next {
		m.appendExecHonestyNotice()
	}
	if group == "web" && next {
		m.appendWebNotice()
	}
	state := map[bool]string{true: "on", false: "off"}[next]
	m.toastf("tools %s: %s (~%d schema tok)", group, state, m.estimateToolTokensHeuristic())
	return m, m.measureToolSchemaTokensCmd()
}

func (m *Model) runPermissionsCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	q := strings.ToLower(strings.TrimSpace(arg))
	switch q {
	case "":
		m.permsModal = &permsModalState{}
		return m, nil
	case "ask":
		m.permEngine.SetMode(agent.ModeAsk)
		m.toastf("permissions: ask")
		return m, nil
	case "read-auto", "read":
		m.permEngine.SetMode(agent.ModeReadAuto)
		m.toastf("permissions: read-auto")
		return m, nil
	case "workspace-auto", "workspace":
		m.permEngine.SetMode(agent.ModeWorkspaceAuto)
		m.toastf("permissions: workspace-auto")
		return m, nil
	case "clear":
		m.permEngine.ClearAllowlist()
		m.toastf("session allowlist cleared")
		return m, nil
	case "clear-taint":
		m.sessionTainted = false
		m.toastf("session taint cleared")
		return m, nil
	default:
		m.appendInfoTurn(fmt.Sprintf("Unknown permission mode %q (use ask, read-auto, workspace-auto, clear, or clear-taint).", arg))
		return m, nil
	}
}

func (m *Model) runWorkspaceCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	q := strings.TrimSpace(arg)
	if q == "" {
		m.appendInfoTurn(fmt.Sprintf("Active workspace: %s", m.agentWorkDir))
		return m, nil
	}
	target := q
	if !filepath.IsAbs(target) {
		target = filepath.Join(m.agentWorkDir, target)
	}
	abs, err := filepath.Abs(target)
	if err != nil {
		m.appendInfoTurn(fmt.Sprintf("Invalid workspace path %q: %v", q, err))
		return m, nil
	}
	st, err := os.Stat(abs)
	if err != nil || !st.IsDir() {
		m.appendInfoTurn(fmt.Sprintf("Workspace directory does not exist: %s", abs))
		return m, nil
	}
	m.agentWorkDir = abs
	m.checkpoints = agent.NewCheckpointStore(abs, m.sessionID)
	m.popup.reset()
	m.toastf("workspace: %s", filepath.Base(abs))
	m.appendInfoTurn(fmt.Sprintf("Switched workspace to: %s", abs))
	return m, nil
}

func (m *Model) runUndoCommand() (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	if m.checkpoints == nil {
		m.appendInfoTurn("Nothing to undo.")
		return m, nil
	}
	turn, paths, ok := m.checkpoints.PendingUndoSummary()
	if !ok || len(paths) == 0 {
		m.appendInfoTurn("Nothing to undo.")
		return m, nil
	}
	restored, deleted, err := m.checkpoints.UndoLastTurn()
	if err != nil {
		m.appendInfoTurn(fmt.Sprintf("Undo failed for turn %d: %v", turn, err))
		return m, nil
	}
	var parts []string
	if len(restored) > 0 {
		parts = append(parts, "restored: "+strings.Join(restored, ", "))
	}
	if len(deleted) > 0 {
		parts = append(parts, "deleted: "+strings.Join(deleted, ", "))
	}
	m.toastf("undid turn %d (%d files)", turn, len(restored)+len(deleted))
	m.appendInfoTurn(fmt.Sprintf("Undid file changes from turn %d (%s). Note: any run_command shell side effects were NOT undone.", turn, strings.Join(parts, "; ")))
	return m, nil
}

func listWorkspaceFiles(root string, limit int) []string {
	if root == "" {
		return nil
	}
	var out []string
	var walk func(dir string, depth int)
	walk = func(dir string, depth int) {
		if depth > 4 || len(out) >= limit {
			return
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, de := range entries {
			if len(out) >= limit {
				return
			}
			name := de.Name()
			if name == ".git" || name == ".alpaca" || name == ".llm" || name == "node_modules" || name == "bin" || name == "dist" {
				continue
			}
			full := filepath.Join(dir, name)
			if de.IsDir() {
				walk(full, depth+1)
				continue
			}
			rel, err := filepath.Rel(root, full)
			if err != nil {
				continue
			}
			out = append(out, filepath.ToSlash(rel))
		}
	}
	walk(root, 0)
	sort.Strings(out)
	return out
}
