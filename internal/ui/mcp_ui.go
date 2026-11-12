package ui

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/mcp"
)

type mcpModalState struct {
	cursor   int
	expanded map[string]bool
	note     string
}

type mcpRow struct {
	serverName string
	rawTool    string // empty when row represents a server
	srv        mcp.ServerStatus
	tool       mcp.ToolStatus
}

func (m *Model) effectiveMCPConfigPath() string {
	if strings.TrimSpace(m.mcpConfigOv) != "" {
		return m.mcpConfigOv
	}
	return filepath.Join(m.root, "mcp.toml")
}

func (m *Model) ensureMCPReady() {
	if m.toolRegistry == nil {
		return
	}
	if m.mcpManager == nil {
		m.mcpManager = mcp.NewManager(m.root, m.workspaceDir(), m.effectiveMCPConfigPath(), m.toolRegistry)
	}
	if m.mcpLoaded {
		return
	}
	m.mcpLoaded = true
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	sum, err := m.mcpManager.Reload(ctx)
	if err != nil {
		m.toastf("mcp load error: %v", err)
		return
	}
	if sum.Warning != "" {
		m.appendInfoTurn(sum.Warning)
	}
}

func (m *Model) runMCPCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	sub := strings.ToLower(strings.TrimSpace(arg))
	switch sub {
	case "reload":
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		m.mcpManager.SetWorkDir(m.workspaceDir())
		sum, err := m.mcpManager.Reload(ctx)
		if err != nil {
			m.appendInfoTurn(fmt.Sprintf("MCP reload failed: %v", err))
			return m, nil
		}
		msg := fmt.Sprintf("MCP reloaded: %d/%d servers running, %d tools (~%d schema tok)",
			sum.RunningServers, sum.TotalServers, sum.TotalEnabledTools, sum.TotalSchemaTokens)
		m.toastf("%s", msg)
		if sum.Warning != "" {
			m.appendInfoTurn(sum.Warning)
		}
		if m.mcpModal != nil {
			m.mcpModal.note = msg
		}
		return m, m.measureToolSchemaTokensCmd()
	case "":
		m.mcpModal = &mcpModalState{expanded: make(map[string]bool)}
		return m, m.measureToolSchemaTokensCmd()
	default:
		m.appendInfoTurn("Usage: /mcp [reload]")
		return m, nil
	}
}

func (m *Model) buildMCPRows() (mcp.Summary, []mcpRow) {
	if m.mcpManager == nil {
		return mcp.Summary{ConfigPath: m.effectiveMCPConfigPath()}, nil
	}
	sum := m.mcpManager.Summary()
	var rows []mcpRow
	for _, srv := range sum.Servers {
		rows = append(rows, mcpRow{serverName: srv.Name, srv: srv})
		if m.mcpModal != nil && m.mcpModal.expanded[srv.Name] {
			for _, t := range srv.Tools {
				rows = append(rows, mcpRow{
					serverName: srv.Name,
					rawTool:    t.RawName,
					srv:        srv,
					tool:       t,
				})
			}
		}
	}
	return sum, rows
}

func (m *Model) handleMCPModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	mm := m.mcpModal
	if mm == nil {
		return m, nil
	}
	_, rows := m.buildMCPRows()
	switch msg.String() {
	case "esc", "q":
		m.mcpModal = nil
		return m, nil
	case "r":
		return m.runMCPCommand("reload")
	case "up", "k":
		if mm.cursor > 0 {
			mm.cursor--
		}
	case "down", "j":
		if mm.cursor < len(rows)-1 {
			mm.cursor++
		}
	case "enter", "tab":
		if mm.cursor >= 0 && mm.cursor < len(rows) {
			row := rows[mm.cursor]
			if mm.expanded == nil {
				mm.expanded = make(map[string]bool)
			}
			mm.expanded[row.serverName] = !mm.expanded[row.serverName]
		}
	case " ", "t":
		if mm.cursor >= 0 && mm.cursor < len(rows) {
			row := rows[mm.cursor]
			if row.rawTool == "" {
				ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
				defer cancel()
				on, err := m.mcpManager.ToggleServer(ctx, row.serverName)
				if err != nil {
					mm.note = fmt.Sprintf("Toggle %s error: %v", row.serverName, err)
				} else {
					st := "disabled"
					if on {
						st = "enabled"
					}
					mm.note = fmt.Sprintf("Server %s %s", row.serverName, st)
				}
			} else {
				on, err := m.mcpManager.ToggleTool(row.serverName, row.rawTool)
				if err != nil {
					mm.note = fmt.Sprintf("Toggle %s error: %v", row.tool.FullName, err)
				} else {
					st := "disabled"
					if on {
						st = "enabled"
					}
					mm.note = fmt.Sprintf("Tool %s %s", row.tool.FullName, st)
				}
			}
			return m, m.measureToolSchemaTokensCmd()
		}
	}
	if mm.cursor >= len(rows) && len(rows) > 0 {
		mm.cursor = len(rows) - 1
	}
	return m, nil
}

func (m *Model) overlayMCPModal(bg string) string {
	mm := m.mcpModal
	if mm == nil {
		return bg
	}
	w, h := m.width, m.height
	boxW := w - 6
	if boxW > 84 {
		boxW = 84
	}
	if boxW < 46 {
		boxW = 46
	}
	innerW := boxW - 4

	sum, rows := m.buildMCPRows()
	if mm.cursor >= len(rows) && len(rows) > 0 {
		mm.cursor = len(rows) - 1
	}

	var lines []string
	lines = append(lines, truncateANSI(m.sty.sMuted.Render("Config: "+m.effectiveMCPConfigPath()+" (stdio only; env secrets hidden)"), innerW))
	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", innerW)))

	if len(rows) == 0 {
		lines = append(lines, m.sty.sMuted.Render("No MCP servers configured in mcp.toml."))
		lines = append(lines, m.sty.sMuted.Render("Add [servers.<name>] with command, args, env and press [r] reload."))
	} else {
		maxRows := h - 14
		if maxRows < 5 {
			maxRows = 5
		}
		if maxRows > 14 {
			maxRows = 14
		}
		start := 0
		if mm.cursor >= maxRows {
			start = mm.cursor - maxRows + 1
		}
		end := start + maxRows
		if end > len(rows) {
			end = len(rows)
		}
		for i := start; i < end; i++ {
			row := rows[i]
			cur := "  "
			if i == mm.cursor {
				cur = m.sty.sAccentBold.Render("▸ ")
			}
			if row.rawTool == "" {
				s := row.srv
				stateTag := m.sty.sMuted.Render("[off]")
				if s.Enabled {
					if s.Running {
						stateTag = m.sty.sSuccess.Render("[on ]")
					} else {
						stateTag = m.sty.sWarning.Render("[err]")
					}
				}
				expChar := "▸"
				if mm.expanded[s.Name] {
					expChar = "▾"
				}
				envNote := ""
				if s.EnvKeyCount > 0 {
					envNote = fmt.Sprintf(" · %d env hidden", s.EnvKeyCount)
				}
				warnNote := ""
				if s.WarnManyTools {
					warnNote = m.sty.sWarning.Render(" [many tools]")
				}
				txt := fmt.Sprintf("%s%s %s %s (%s · %d/%d tools · ~%d tok · class:%s%s)%s",
					cur, stateTag, expChar,
					m.sty.sText.Bold(true).Render(s.Name),
					s.State, s.EnabledToolCount, s.ToolCount, s.SchemaTokens, s.Class, envNote, warnNote,
				)
				lines = append(lines, truncateANSI(txt, innerW))
				if s.LastError != "" && !s.Running && s.Enabled {
					lines = append(lines, truncateANSI("      "+m.sty.sDanger.Render("err: "+s.LastError), innerW))
				}
			} else {
				t := row.tool
				tag := m.sty.sMuted.Render("[off]")
				if t.Enabled {
					tag = m.sty.sSuccess.Render("[on ]")
				}
				txt := fmt.Sprintf("%s    %s %s (~%d tok)", cur, tag, m.sty.accent2().Render(t.FullName), t.SchemaTokens)
				lines = append(lines, truncateANSI(txt, innerW))
			}
		}
	}

	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", innerW)))
	lines = append(lines, truncateANSI(m.sty.accent2().Render(
		fmt.Sprintf("Active MCP: %d/%d servers · %d tools (~%d schema tok)",
			sum.RunningServers, sum.TotalServers, sum.TotalEnabledTools, sum.TotalSchemaTokens),
	), innerW))
	if sum.Warning != "" {
		lines = append(lines, truncateANSI(m.sty.sWarning.Render(sum.Warning), innerW))
	}
	if mm.note != "" {
		lines = append(lines, truncateANSI(m.sty.sText.Render(mm.note), innerW))
	}
	lines = append(lines, "")
	lines = append(lines, truncateANSI(m.sty.sMuted.Render("↑↓ select · enter expand · space toggle · r reload · esc close"), innerW))

	box := m.sty.panel(boxW, len(lines)+3, "MCP Servers (/mcp)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}
