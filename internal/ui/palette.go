package ui

import (
	"fmt"
	"sort"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

const paletteMaxVisible = 10

type paletteEntry struct {
	Category       string // "cmd", "setting", "preset", "model"
	Label          string
	Description    string
	Disabled       bool
	DisabledReason string
	MatchPositions []int
	Run            func(m *Model) (tea.Model, tea.Cmd)
}

type commandPalette struct {
	open   bool
	input  textinput.Model
	items  []paletteEntry
	idx    int
	scroll int
}

func newCommandPalette() commandPalette {
	ti := textinput.New()
	ti.Placeholder = "Search commands, settings, presets, models…"
	ti.CharLimit = 120
	ti.Prompt = ""
	return commandPalette{input: ti}
}

func (m *Model) openCommandPalette() {
	m.palette.open = true
	m.palette.input.SetValue("")
	m.palette.input.Focus()
	m.palette.idx = 0
	m.palette.scroll = 0
	m.syncPaletteItems()
}

func (m *Model) closeCommandPalette() {
	m.palette.open = false
	m.palette.input.Blur()
	if m.screen == screenRun && m.runTab == runTabChat {
		m.chatInput.Focus()
	}
}

func (m *Model) paletteCatalog() []paletteEntry {
	var out []paletteEntry

	// 1. Registered slash commands.
	for _, c := range m.commandRegistry() {
		cmdCopy := c
		disabled := false
		reason := ""
		if cmdCopy.Enabled != nil {
			if ok, why := cmdCopy.Enabled(m); !ok {
				disabled = true
				reason = why
			}
		}
		out = append(out, paletteEntry{
			Category:       "cmd",
			Label:          "/" + cmdCopy.Name,
			Description:    cmdCopy.Description,
			Disabled:       disabled,
			DisabledReason: reason,
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				if m.screen != screenRun || m.runTab != runTabChat {
					if cmdCopy.Name == "theme" || cmdCopy.Name == "quit" || cmdCopy.Name == "model" || cmdCopy.Name == "preset" {
						return cmdCopy.Run(m, "")
					}
					m.runTab = runTabChat
				}
				return cmdCopy.Run(m, "")
			},
		})
	}

	// 2. Quick settings toggles.
	out = append(out,
		paletteEntry{
			Category:    "setting",
			Label:       "Toggle thinking display",
			Description: "Show or collapse reasoning traces in chat",
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				m.toggleThinkingDisplay()
				m.refreshChatView(false)
				return m, nil
			},
		},
		paletteEntry{
			Category:    "setting",
			Label:       "Toggle code line numbers",
			Description: "Show line numbers in fenced code blocks",
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				m.codeLineNums = !m.codeLineNums
				m.refreshChatView(false)
				return m, nil
			},
		},
		paletteEntry{
			Category:    "setting",
			Label:       "Cycle color theme",
			Description: "Current: " + m.sty.t.Name,
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				return m.runThemeCommand("")
			},
		},
		paletteEntry{
			Category:    "setting",
			Label:       "Cycle reasoning effort",
			Description: "Current: " + m.reasoningBadgeLabel(),
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				m.cycleReasoningEffortLive()
				return m, nil
			},
		},
		paletteEntry{
			Category:    "setting",
			Label:       "Toggle Auto fit",
			Description: fmt.Sprintf("Auto-fit VRAM layers (now %v)", m.settings.AutoFit),
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				m.settings.AutoFit = !m.settings.AutoFit
				m.userEditedAutoFit = true
				m.recompute()
				return m, nil
			},
		},
		paletteEntry{
			Category:    "setting",
			Label:       "Toggle Flash Attention",
			Description: fmt.Sprintf("Flash attention KV optimization (now %v)", m.settings.FlashAttn),
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				m.settings.FlashAttn = !m.settings.FlashAttn
				m.recompute()
				return m, nil
			},
		},
	)

	// 3. Saved presets.
	for _, p := range m.loadPresetsFast() {
		pCopy := p
		out = append(out, paletteEntry{
			Category:    "preset",
			Label:       pCopy.Name,
			Description: fmt.Sprintf("Apply preset (ctx %d · %s)", pCopy.Settings.Context, pCopy.Settings.VCache),
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				return m.runPresetCommand(pCopy.Name)
			},
		})
	}

	// 4. Discovered models.
	for _, e := range m.models {
		rel := e.cand.RelPath
		lbl := rel
		if e.meta != nil && e.meta.DisplayName() != "" {
			lbl = e.meta.DisplayName()
		}
		out = append(out, paletteEntry{
			Category:    "model",
			Label:       lbl,
			Description: "Switch model (" + rel + ")",
			Run: func(m *Model) (tea.Model, tea.Cmd) {
				return m.runModelCommand(rel)
			},
		})
	}

	return out
}

func (m *Model) syncPaletteItems() {
	all := m.paletteCatalog()
	q := strings.ToLower(strings.TrimSpace(m.palette.input.Value()))
	if q == "" {
		m.palette.items = all
		m.palette.idx = 0
		m.palette.scroll = 0
		return
	}

	type scoredPal struct {
		entry paletteEntry
		tier  int
		score int
		orig  int
	}
	var scored []scoredPal
	for i, it := range all {
		tier, sc, pos, ok := matchQuery(q, strings.ToLower(it.Label))
		if !ok {
			if t2, s2, _, ok2 := matchQuery(q, strings.ToLower(it.Category+" "+it.Description)); ok2 {
				tier, sc, pos, ok = t2+1, s2+20, nil, true
			}
		}
		if !ok {
			continue
		}
		it.MatchPositions = pos
		scored = append(scored, scoredPal{entry: it, tier: tier, score: sc, orig: i})
	}
	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].tier != scored[j].tier {
			return scored[i].tier < scored[j].tier
		}
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		return scored[i].orig < scored[j].orig
	})

	filtered := make([]paletteEntry, len(scored))
	for i, s := range scored {
		filtered[i] = s.entry
	}
	m.palette.items = filtered
	m.palette.idx = 0
	m.palette.scroll = 0
}

func (m *Model) updatePaletteKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "ctrl+p":
		m.closeCommandPalette()
		return m, nil
	case "up":
		if n := len(m.palette.items); n > 0 {
			m.palette.idx = ((m.palette.idx-1)%n + n) % n
			if m.palette.idx < m.palette.scroll {
				m.palette.scroll = m.palette.idx
			}
			if m.palette.idx >= m.palette.scroll+paletteMaxVisible {
				m.palette.scroll = m.palette.idx - paletteMaxVisible + 1
			}
		}
		return m, nil
	case "down":
		if n := len(m.palette.items); n > 0 {
			m.palette.idx = (m.palette.idx + 1) % n
			if m.palette.idx < m.palette.scroll {
				m.palette.scroll = m.palette.idx
			}
			if m.palette.idx >= m.palette.scroll+paletteMaxVisible {
				m.palette.scroll = m.palette.idx - paletteMaxVisible + 1
			}
		}
		return m, nil
	case "enter":
		if m.palette.idx >= 0 && m.palette.idx < len(m.palette.items) {
			chosen := m.palette.items[m.palette.idx]
			if chosen.Disabled {
				m.toastf("%s disabled: %s", chosen.Label, chosen.DisabledReason)
				return m, nil
			}
			m.closeCommandPalette()
			if chosen.Run != nil {
				return chosen.Run(m)
			}
		}
		return m, nil
	}

	var cmd tea.Cmd
	m.palette.input, cmd = m.palette.input.Update(msg)
	m.syncPaletteItems()
	return m, cmd
}

func (m *Model) overlayPalette(base string) string {
	w := m.width - 8
	if w > 76 {
		w = 76
	}
	if w < 44 {
		w = 44
	}
	innerW := w - 4
	m.palette.input.Width = innerW - 4

	var b strings.Builder
	b.WriteString(m.sty.accent().Render("Command Palette") + m.sty.muted().Render("  [ctrl+p]"))
	b.WriteString("\n")
	b.WriteString(m.sty.sAccent.Render("› ") + m.palette.input.View())
	b.WriteString("\n" + m.sty.sMuted.Render(strings.Repeat("─", innerW)))

	if len(m.palette.items) == 0 {
		b.WriteString("\n" + m.sty.muted().Render("  no matching items"))
	} else {
		vis := paletteMaxVisible
		if len(m.palette.items) < vis {
			vis = len(m.palette.items)
		}
		start := m.palette.scroll
		if start+vis > len(m.palette.items) {
			start = len(m.palette.items) - vis
		}
		if start < 0 {
			start = 0
		}
		end := start + vis
		for i := start; i < end; i++ {
			it := m.palette.items[i]
			selected := i == m.palette.idx
			marker := "  "
			if selected {
				marker = "▸ "
				if m.sty.ascii {
					marker = "> "
				}
			}
			catTag := m.sty.sMuted.Render(fmt.Sprintf("[%s]", pad(it.Category, 7)))
			lblRaw := truncate(it.Label, 24)
			lbl := renderHighlightedLabel(m.sty, lblRaw, it.MatchPositions, selected, it.Disabled)
			if p := 24 - displayWidth(lblRaw); p > 0 {
				lbl += strings.Repeat(" ", p)
			}
			descW := max(8, innerW-displayWidth(marker)-10-25)
			desc := it.Description
			if it.Disabled && it.DisabledReason != "" {
				desc = "(" + it.DisabledReason + ") " + desc
			}
			descPart := m.sty.sMuted.Render(truncate(desc, descW))
			row := widthClampANSI(marker+catTag+" "+lbl+" "+descPart, innerW)
			b.WriteString("\n" + row)
		}
	}

	b.WriteString("\n" + m.sty.muted().Render("↑↓ move · enter run · esc close"))
	box := m.sty.sInputBox.Width(w).Render(b.String())
	return overlay(base, box)
}
