package ui

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/skills"
	"github.com/MdHussain121/alpaca/prompts"
)

type skillsModalMode int

const (
	skillsModeList skillsModalMode = iota
	skillsModeView
	skillsModeCreateName
	skillsModeCreateDesc
	skillsModeCreateBody
	skillsModeRename
)

type skillsModalState struct {
	cursor  int
	scroll  int
	mode    skillsModalMode
	nameIn  textinput.Model
	descIn  textinput.Model
	bodyIn  textinput.Model
	errText string
}

type skillApproveMode int

const (
	skillApprovePreview skillApproveMode = iota
	skillApproveRejectNote
	skillApproveEditName
	skillApproveEditDesc
	skillApproveEditBody
)

type skillApproveModalState struct {
	gen       uint64
	prompt    agent.SkillStagePrompt
	replyCh   chan<- agent.SkillStageResponse
	ch        <-chan agentLoopMsg
	doneCh    <-chan agent.LoopResult
	mode      skillApproveMode
	scroll    int
	noteInput textinput.Model
	nameInput textinput.Model
	descInput textinput.Model
	editBody  string
	bodyInput textinput.Model
	errText   string
}

func (m *Model) effectiveSkillsDir() string {
	if strings.TrimSpace(m.skillsDirOverride) != "" {
		return m.skillsDirOverride
	}
	return skills.SkillsRoot(m.root, m.agentWorkDir)
}

func (m *Model) effectiveSkillsDataDir() string {
	return skills.DataRoot(m.effectiveSkillsDir(), m.agentWorkDir)
}

func (m *Model) reloadSkills() {
	sDir := m.effectiveSkillsDir()
	dDir := m.effectiveSkillsDataDir()
	if strings.TrimSpace(m.root) != "" || strings.TrimSpace(m.skillsDirOverride) != "" {
		_ = skills.EnsureStarterSkills(sDir, dDir)
	}
	m.skillsCatalog = skills.LoadCatalog(sDir, dDir)
	_, tok, _ := m.skillsCatalog.SystemPromptBlock()
	m.skillListTokens = tok
}

func (m *Model) effectiveMCPSummary() string {
	if m.mcpManager == nil {
		return ""
	}
	sum := m.mcpManager.Summary()
	if len(sum.Servers) == 0 {
		return ""
	}
	var parts []string
	for _, srv := range sum.Servers {
		if !srv.Enabled || !srv.Running || srv.EnabledToolCount == 0 {
			continue
		}
		var names []string
		for _, tl := range srv.Tools {
			if tl.Enabled {
				names = append(names, tl.FullName)
			}
		}
		if len(names) > 0 {
			parts = append(parts, fmt.Sprintf("%s (%s)", srv.Name, strings.Join(names, ", ")))
		}
	}
	return strings.Join(parts, "; ")
}

func (m *Model) effectivePromptRenderResult(includeAgentsAndSkills bool) prompts.RenderResult {
	appRoot := strings.TrimSpace(m.root)
	if appRoot == "" && m.cfg != nil {
		appRoot = strings.TrimSpace(m.cfg.Root())
	}
	permMode := string(agent.ModeReadAuto)
	if m.permEngine != nil {
		permMode = string(m.permEngine.Mode())
	}
	var tools []runner.ToolDef
	if m.toolRegistry != nil {
		tools = m.toolRegistry.ToolDefs()
	}
	var agentsMD, skillsList string
	if includeAgentsAndSkills {
		agentsMD = m.effectiveAgentsMD().CombinedPrompt
		block, tok, _ := m.skillsCatalog.SystemPromptBlock()
		m.skillListTokens = tok
		skillsList = block
	}
	disabled := m.cfg != nil && m.cfg.DisableSystemPrompt
	return prompts.Render(prompts.RenderOptions{
		AppRoot:        appRoot,
		Workspace:      m.agentWorkDir,
		PermissionMode: permMode,
		Tools:          tools,
		SkillsList:     skillsList,
		MCPSummary:     m.effectiveMCPSummary(),
		AgentsMD:       agentsMD,
		ContextSize:    m.effectiveCtxTotal(),
		CustomPrompt:   m.settings.SystemPrompt,
		DisableDefault: disabled,
	})
}

func (m *Model) effectiveSystemPrompt() string {
	return m.effectivePromptRenderResult(true).Text
}

func (m *Model) effectiveSystemBaseForBreakdown() string {
	return m.effectivePromptRenderResult(false).Text
}

// --- /skills and /skill commands ---

func (m *Model) runSkillsCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	m.reloadSkills()

	sub := strings.ToLower(strings.TrimSpace(arg))
	switch sub {
	case "reload":
		m.appendInfoTurn(fmt.Sprintf("Reloaded skills: %d valid, %d invalid (prompt list ~%d tok).",
			len(m.skillsCatalog.Valid), len(m.skillsCatalog.Invalid), m.skillListTokens))
		return m, nil
	case "list":
		var b strings.Builder
		b.WriteString(fmt.Sprintf("Skills in %s (%d valid, %d invalid, ~%d prompt tok):",
			m.skillsCatalog.SkillsDir, len(m.skillsCatalog.Valid), len(m.skillsCatalog.Invalid), m.skillListTokens))
		for _, s := range m.skillsCatalog.Valid {
			st := "on"
			if !s.Enabled {
				st = "off"
			}
			badge := ""
			if s.ModelAuthored {
				badge = " [model-authored]"
			}
			b.WriteString(fmt.Sprintf("\n  - %s (%s%s, uses:%d): %s", s.Name, st, badge, s.UseCount, s.Description))
		}
		for _, inv := range m.skillsCatalog.Invalid {
			b.WriteString(fmt.Sprintf("\n  ! INVALID %s: %s", inv.Name, inv.Reason))
		}
		m.appendInfoTurn(b.String())
		return m, nil
	default:
		m.skillsModal = &skillsModalState{}
		return m, nil
	}
}

func (m *Model) runUseSkillCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	m.reloadSkills()

	fields := strings.Fields(strings.TrimSpace(arg))
	if len(fields) == 0 {
		m.appendInfoTurn("Usage: /skill <name> [optional instructions]")
		return m, nil
	}
	name := strings.ToLower(fields[0])
	extra := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(arg), fields[0]))

	sk, ok := m.skillsCatalog.Lookup(name, false)
	if !ok {
		if dis, exists := m.skillsCatalog.Lookup(name, true); exists && !dis.Enabled {
			m.appendInfoTurn(fmt.Sprintf("Skill %q is disabled. Enable it in /skills first.", name))
			return m, nil
		}
		m.appendInfoTurn(fmt.Sprintf("Skill %q not found. Run /skills to inspect available skills.", name))
		return m, nil
	}

	if m.status == "READY" || m.chatBaseURL != "" {
		prompt := fmt.Sprintf("Apply skill %q now. First load it using `use_skill` with name %q, then follow its instructions safely within active permissions.", sk.Name, sk.Name)
		if extra != "" {
			prompt += "\n\nTask details: " + extra
		}
		return m.startAgentCompletion(prompt)
	}

	_ = skills.RecordUse(m.effectiveSkillsDataDir(), sk.Name)
	m.reloadSkills()
	m.appendInfoTurn(fmt.Sprintf("Loaded skill %q (%s):\n\n%s", sk.Name, sk.Description, sk.Body))
	return m, nil
}

// --- /skills modal ---

type skillRow struct {
	valid   *skills.Skill
	invalid *skills.InvalidSkill
}

func (m *Model) skillsRows() []skillRow {
	rows := make([]skillRow, 0, len(m.skillsCatalog.Valid)+len(m.skillsCatalog.Invalid))
	for i := range m.skillsCatalog.Valid {
		rows = append(rows, skillRow{valid: &m.skillsCatalog.Valid[i]})
	}
	for i := range m.skillsCatalog.Invalid {
		rows = append(rows, skillRow{invalid: &m.skillsCatalog.Invalid[i]})
	}
	return rows
}

func (m *Model) handleSkillsModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	sm := m.skillsModal
	if sm == nil {
		return m, nil
	}
	rows := m.skillsRows()
	if sm.cursor >= len(rows) && len(rows) > 0 {
		sm.cursor = len(rows) - 1
	}
	if sm.cursor < 0 {
		sm.cursor = 0
	}

	switch sm.mode {
	case skillsModeView:
		switch msg.String() {
		case "esc", "enter", "q":
			sm.mode = skillsModeList
			sm.scroll = 0
		case "up", "k":
			if sm.scroll > 0 {
				sm.scroll--
			}
		case "down", "j":
			sm.scroll++
		case "o", "e":
			if len(rows) > 0 && rows[sm.cursor].valid != nil {
				_ = openSkillInEditor(rows[sm.cursor].valid.SkillFile)
				m.toastf("opened %s in default editor", rows[sm.cursor].valid.Name)
			}
		}
		return m, nil

	case skillsModeCreateName:
		switch msg.Type {
		case tea.KeyEsc:
			sm.mode = skillsModeList
			sm.errText = ""
			return m, nil
		case tea.KeyEnter:
			clean, err := skills.ValidateSkillName(sm.nameIn.Value())
			if err != nil {
				sm.errText = err.Error()
				return m, nil
			}
			if _, exists := m.skillsCatalog.Lookup(clean, true); exists {
				sm.errText = fmt.Sprintf("skill %q already exists", clean)
				return m, nil
			}
			sm.nameIn.SetValue(clean)
			sm.errText = ""
			sm.mode = skillsModeCreateDesc
			sm.descIn.Focus()
			return m, nil
		}
		var c tea.Cmd
		sm.nameIn, c = sm.nameIn.Update(msg)
		return m, c

	case skillsModeCreateDesc:
		switch msg.Type {
		case tea.KeyEsc:
			sm.mode = skillsModeList
			sm.errText = ""
			return m, nil
		case tea.KeyEnter:
			desc, err := skills.SanitizeDescription(sm.descIn.Value())
			if err != nil {
				sm.errText = err.Error()
				return m, nil
			}
			sm.descIn.SetValue(desc)
			sm.errText = ""
			sm.mode = skillsModeCreateBody
			sm.bodyIn.Focus()
			return m, nil
		}
		var c tea.Cmd
		sm.descIn, c = sm.descIn.Update(msg)
		return m, c

	case skillsModeCreateBody:
		switch msg.Type {
		case tea.KeyEsc:
			sm.mode = skillsModeList
			sm.errText = ""
			return m, nil
		case tea.KeyEnter:
			body := strings.TrimSpace(sm.bodyIn.Value())
			if err := skills.CreateSkill(m.effectiveSkillsDir(), sm.nameIn.Value(), sm.descIn.Value(), body); err != nil {
				sm.errText = err.Error()
				return m, nil
			}
			created := sm.nameIn.Value()
			sm.mode = skillsModeList
			sm.errText = ""
			m.reloadSkills()
			m.toastf("created skill %s", created)
			return m, nil
		}
		var c tea.Cmd
		sm.bodyIn, c = sm.bodyIn.Update(msg)
		return m, c

	case skillsModeRename:
		switch msg.Type {
		case tea.KeyEsc:
			sm.mode = skillsModeList
			sm.errText = ""
			return m, nil
		case tea.KeyEnter:
			if len(rows) == 0 || rows[sm.cursor].valid == nil {
				sm.mode = skillsModeList
				return m, nil
			}
			oldName := rows[sm.cursor].valid.Name
			newName := strings.TrimSpace(sm.nameIn.Value())
			if err := skills.RenameSkill(m.effectiveSkillsDir(), m.effectiveSkillsDataDir(), oldName, newName); err != nil {
				sm.errText = err.Error()
				return m, nil
			}
			sm.mode = skillsModeList
			sm.errText = ""
			m.reloadSkills()
			m.toastf("renamed %s -> %s", oldName, newName)
			return m, nil
		}
		var c tea.Cmd
		sm.nameIn, c = sm.nameIn.Update(msg)
		return m, c
	}

	// List mode
	switch msg.String() {
	case "esc", "q":
		m.skillsModal = nil
		return m, nil
	case "up", "k":
		if sm.cursor > 0 {
			sm.cursor--
		}
		return m, nil
	case "down", "j":
		if sm.cursor < len(rows)-1 {
			sm.cursor++
		}
		return m, nil
	case " ", "t":
		if len(rows) > 0 && rows[sm.cursor].valid != nil {
			sk := rows[sm.cursor].valid
			_ = skills.SetEnabled(m.effectiveSkillsDataDir(), sk.Name, !sk.Enabled)
			m.reloadSkills()
			state := "enabled"
			if sk.Enabled {
				state = "disabled"
			}
			m.toastf("skill %s %s", sk.Name, state)
		}
		return m, nil
	case "enter", "v":
		if len(rows) > 0 {
			sm.mode = skillsModeView
			sm.scroll = 0
		}
		return m, nil
	case "n":
		ni := textinput.New()
		ni.Prompt = ""
		ni.Placeholder = "lowercase-hyphen-name"
		ni.CharLimit = 40
		ni.Width = 36
		ni.Focus()

		di := textinput.New()
		di.Prompt = ""
		di.Placeholder = "One-line description of when to use this skill"
		di.CharLimit = 240
		di.Width = 54

		bi := textinput.New()
		bi.Prompt = ""
		bi.Placeholder = "Initial Markdown instructions (you can expand in editor with [o])"
		bi.CharLimit = 4096
		bi.Width = 54

		sm.nameIn = ni
		sm.descIn = di
		sm.bodyIn = bi
		sm.errText = ""
		sm.mode = skillsModeCreateName
		return m, nil
	case "r":
		if len(rows) > 0 && rows[sm.cursor].valid != nil {
			ni := textinput.New()
			ni.Prompt = ""
			ni.CharLimit = 40
			ni.Width = 36
			ni.SetValue(rows[sm.cursor].valid.Name)
			ni.Focus()
			sm.nameIn = ni
			sm.errText = ""
			sm.mode = skillsModeRename
		}
		return m, nil
	case "d", "x":
		if len(rows) > 0 {
			target := ""
			if rows[sm.cursor].valid != nil {
				target = rows[sm.cursor].valid.Name
			} else if rows[sm.cursor].invalid != nil {
				target = rows[sm.cursor].invalid.Name
			}
			if target != "" {
				if trashPath, err := skills.DeleteSkill(m.effectiveSkillsDir(), m.effectiveSkillsDataDir(), target); err != nil {
					m.toastf("delete error: %v", err)
				} else {
					m.reloadSkills()
					if sm.cursor > 0 && sm.cursor >= len(m.skillsRows()) {
						sm.cursor--
					}
					m.toastf("moved %s to %s", target, filepath.Base(trashPath))
				}
			}
		}
		return m, nil
	case "o", "e":
		if len(rows) > 0 {
			p := ""
			name := ""
			if rows[sm.cursor].valid != nil {
				p = rows[sm.cursor].valid.SkillFile
				name = rows[sm.cursor].valid.Name
			} else if rows[sm.cursor].invalid != nil {
				p = filepath.Join(rows[sm.cursor].invalid.Dir, "SKILL.md")
				name = rows[sm.cursor].invalid.Name
			}
			if p != "" {
				_ = openSkillInEditor(p)
				m.toastf("opened %s in default editor", name)
			}
		}
		return m, nil
	case "u":
		if len(rows) > 0 && rows[sm.cursor].valid != nil && rows[sm.cursor].valid.Enabled {
			name := rows[sm.cursor].valid.Name
			m.skillsModal = nil
			return m.runUseSkillCommand(name)
		}
		return m, nil
	}
	return m, nil
}

func (m *Model) overlaySkillsModal(bg string) string {
	sm := m.skillsModal
	if sm == nil {
		return bg
	}
	m.ensureAgentReady()
	w, h := m.width, m.height
	boxW := w - 8
	if boxW > 88 {
		boxW = 88
	}
	if boxW < 48 {
		boxW = 48
	}
	innerW := boxW - 4
	rows := m.skillsRows()

	var lines []string
	hdr := fmt.Sprintf("Skills: %d valid, %d invalid · system prompt ~%d tok",
		len(m.skillsCatalog.Valid), len(m.skillsCatalog.Invalid), m.skillListTokens)
	lines = append(lines, truncateANSI(m.sty.sAccentBold.Render(hdr), innerW))
	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", innerW)))

	switch sm.mode {
	case skillsModeView:
		if len(rows) > 0 {
			r := rows[sm.cursor]
			var viewLines []string
			if r.valid != nil {
				sk := r.valid
				metaLine := fmt.Sprintf("Name: %s · uses: %d · last used: %s · %d B",
					sk.Name, sk.UseCount, formatLastUsed(sk.LastUsed), sk.TotalBytes)
				if sk.ModelAuthored {
					metaLine += " · model-authored"
				}
				viewLines = append(viewLines, m.sty.accent2().Render(truncate(metaLine, innerW)))
				viewLines = append(viewLines, m.sty.sMuted.Render("Description: "+truncate(sk.Description, innerW-13)))
				if len(sk.ExtraFiles) > 0 {
					viewLines = append(viewLines, m.sty.sMuted.Render("Files: "+truncate(strings.Join(sk.ExtraFiles, ", "), innerW-7)))
				}
				viewLines = append(viewLines, "")
				viewLines = append(viewLines, renderMarkdownLines(m.sty, innerW, sk.Body, m.codeLineNums, false, nil)...)
			} else if r.invalid != nil {
				inv := r.invalid
				viewLines = append(viewLines, m.sty.sDanger.Render("Invalid Skill: "+inv.Name))
				viewLines = append(viewLines, m.sty.sMuted.Render("Path: "+inv.Dir))
				viewLines = append(viewLines, m.sty.sWarning.Render("Reason: "+inv.Reason))
			}
			maxV := h - 12
			if maxV < 6 {
				maxV = 6
			}
			if maxV > 18 {
				maxV = 18
			}
			start := sm.scroll
			if start > len(viewLines)-maxV {
				start = len(viewLines) - maxV
			}
			if start < 0 {
				start = 0
			}
			sm.scroll = start
			end := start + maxV
			if end > len(viewLines) {
				end = len(viewLines)
			}
			lines = append(lines, viewLines[start:end]...)
		}
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("↑↓ scroll · o open in editor · esc back"))

	case skillsModeCreateName, skillsModeCreateDesc, skillsModeCreateBody:
		lines = append(lines, m.sty.sAccent2Bold.Render("Create New Skill"))
		lines = append(lines, "Name: "+sm.nameIn.View())
		if sm.mode >= skillsModeCreateDesc {
			lines = append(lines, "Desc: "+sm.descIn.View())
		}
		if sm.mode == skillsModeCreateBody {
			lines = append(lines, "Body: "+sm.bodyIn.View())
		}
		if sm.errText != "" {
			lines = append(lines, m.sty.sDanger.Render("Error: "+sm.errText))
		}
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("enter next/save · esc cancel"))

	case skillsModeRename:
		lines = append(lines, m.sty.sAccent2Bold.Render("Rename Skill"))
		lines = append(lines, "New name: "+sm.nameIn.View())
		if sm.errText != "" {
			lines = append(lines, m.sty.sDanger.Render("Error: "+sm.errText))
		}
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("enter rename · esc cancel"))

	default:
		if len(rows) == 0 {
			lines = append(lines, m.sty.sMuted.Render("No skills found in "+m.effectiveSkillsDir()+". Press [n] to create one."))
		} else {
			maxRows := h - 12
			if maxRows < 4 {
				maxRows = 4
			}
			if maxRows > 14 {
				maxRows = 14
			}
			start := 0
			if sm.cursor >= maxRows {
				start = sm.cursor - maxRows + 1
			}
			end := start + maxRows
			if end > len(rows) {
				end = len(rows)
			}
			for i := start; i < end; i++ {
				cur := "  "
				if i == sm.cursor {
					cur = m.sty.sAccentBold.Render("▸ ")
				}
				r := rows[i]
				if r.valid != nil {
					sk := r.valid
					st := m.sty.sSuccess.Render("[on ]")
					if !sk.Enabled {
						st = m.sty.sMuted.Render("[off]")
					}
					badge := ""
					if sk.ModelAuthored {
						badge = " " + m.sty.badge("model", m.sty.t.Accent2)
					}
					usage := fmt.Sprintf("uses:%d", sk.UseCount)
					if !sk.LastUsed.IsZero() {
						usage += " " + formatLastUsed(sk.LastUsed)
					}
					rowLine := fmt.Sprintf("%s%s %s%s %s — %s",
						cur, st, m.sty.sText.Bold(true).Render(sk.Name), badge,
						m.sty.sMuted.Render("("+usage+")"),
						m.sty.sMuted.Render(sk.Description))
					lines = append(lines, truncateANSI(rowLine, innerW))
				} else if r.invalid != nil {
					inv := r.invalid
					rowLine := fmt.Sprintf("%s%s %s — %s",
						cur, m.sty.badge("ERR", m.sty.t.Error),
						m.sty.sDanger.Render(inv.Name),
						m.sty.sWarning.Render(inv.Reason))
					lines = append(lines, truncateANSI(rowLine, innerW))
				}
			}
		}
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("↑↓ select · space toggle · enter view · n new · r rename · d trash · o editor · u use · esc close"))
	}

	box := m.sty.panel(boxW, len(lines)+3, "Agent Skills (/skills)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}

// --- save_skill approval preview modal ---

func (m *Model) handleSkillApproveModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	sam := m.skillApproveModal
	if sam == nil {
		return m, nil
	}
	respond := func(resp agent.SkillStageResponse) (tea.Model, tea.Cmd) {
		sam.replyCh <- resp
		gen, ch, doneCh := sam.gen, sam.ch, sam.doneCh
		m.skillApproveModal = nil
		return m, readAgentEvtCmd(gen, ch, doneCh)
	}

	switch sam.mode {
	case skillApproveRejectNote:
		switch msg.Type {
		case tea.KeyEnter:
			return respond(agent.SkillStageResponse{
				Approved: false,
				Reason:   strings.TrimSpace(sam.noteInput.Value()),
			})
		case tea.KeyEsc:
			sam.mode = skillApprovePreview
			sam.noteInput.Blur()
			return m, nil
		}
		var c tea.Cmd
		sam.noteInput, c = sam.noteInput.Update(msg)
		return m, c

	case skillApproveEditName:
		switch msg.Type {
		case tea.KeyEsc:
			sam.mode = skillApprovePreview
			sam.errText = ""
			return m, nil
		case tea.KeyEnter:
			clean, err := skills.ValidateSkillName(sam.nameInput.Value())
			if err != nil {
				sam.errText = err.Error()
				return m, nil
			}
			sam.nameInput.SetValue(clean)
			sam.errText = ""
			sam.mode = skillApproveEditDesc
			sam.descInput.Focus()
			return m, nil
		}
		var c tea.Cmd
		sam.nameInput, c = sam.nameInput.Update(msg)
		return m, c

	case skillApproveEditDesc:
		switch msg.Type {
		case tea.KeyEsc:
			sam.mode = skillApprovePreview
			sam.errText = ""
			return m, nil
		case tea.KeyEnter:
			desc, err := skills.SanitizeDescription(sam.descInput.Value())
			if err != nil {
				sam.errText = err.Error()
				return m, nil
			}
			sam.descInput.SetValue(desc)
			sam.errText = ""
			bi := textinput.New()
			bi.Prompt = ""
			bi.CharLimit = skills.MaxBodyBytes
			bi.Width = 54
			bi.SetValue(sam.editBody)
			bi.Focus()
			sam.bodyInput = bi
			sam.mode = skillApproveEditBody
			return m, nil
		}
		var c tea.Cmd
		sam.descInput, c = sam.descInput.Update(msg)
		return m, c

	case skillApproveEditBody:
		switch msg.Type {
		case tea.KeyEsc:
			sam.mode = skillApprovePreview
			sam.errText = ""
			return m, nil
		case tea.KeyEnter:
			if strings.TrimSpace(sam.bodyInput.Value()) != "" {
				sam.editBody = sam.bodyInput.Value()
			}
			sam.mode = skillApprovePreview
			sam.errText = ""
			return m, nil
		}
		var c tea.Cmd
		sam.bodyInput, c = sam.bodyInput.Update(msg)
		return m, c
	}

	switch msg.String() {
	case "y", "a", "enter":
		return respond(agent.SkillStageResponse{
			Approved: true,
			Name:     strings.TrimSpace(sam.nameInput.Value()),
			Desc:     strings.TrimSpace(sam.descInput.Value()),
			Body:     sam.editBody,
		})
	case "n", "esc":
		return respond(agent.SkillStageResponse{
			Approved: false,
			Reason:   "rejected by user",
		})
	case "d":
		sam.mode = skillApproveRejectNote
		sam.noteInput.Focus()
		return m, nil
	case "e":
		sam.errText = ""
		sam.mode = skillApproveEditName
		sam.nameInput.Focus()
		return m, nil
	case "o":
		_ = openSkillInEditor(filepath.Join(sam.prompt.StagingDir, "SKILL.md"))
		m.toastf("opened staged SKILL.md — press [r] to reload edits")
		return m, nil
	case "r":
		// Reload any edits made externally to the staged SKILL.md
		if raw, err := os.ReadFile(filepath.Join(sam.prompt.StagingDir, "SKILL.md")); err == nil {
			cat := skills.LoadCatalog(filepath.Dir(sam.prompt.StagingDir), "")
			if sk, ok := cat.Lookup(sam.prompt.Name, true); ok {
				sam.nameInput.SetValue(sk.Name)
				sam.descInput.SetValue(sk.Description)
				sam.editBody = sk.Body
				m.toastf("reloaded staged SKILL.md (%d B)", len(raw))
			}
		}
		return m, nil
	case "up", "k":
		if sam.scroll > 0 {
			sam.scroll--
		}
		return m, nil
	case "down", "j":
		sam.scroll++
		return m, nil
	}
	return m, nil
}

func (m *Model) overlaySkillApproveModal(bg string) string {
	sam := m.skillApproveModal
	if sam == nil {
		return bg
	}
	w, h := m.width, m.height
	boxW := w - 8
	if boxW > 88 {
		boxW = 88
	}
	if boxW < 48 {
		boxW = 48
	}
	innerW := boxW - 4

	var lines []string
	action := "NEW SKILL"
	if sam.prompt.Overwrites {
		action = "OVERWRITE (archives prior version)"
	}
	hdr := m.sty.sAccentBold.Render("Skill Approval Preview: "+sam.nameInput.Value()) + "  " +
		m.sty.badge(" model-authored ", m.sty.t.Accent2) + " " +
		m.sty.sMuted.Render(fmt.Sprintf("%s · %d B", action, sam.prompt.TotalBytes))
	lines = append(lines, truncateANSI(hdr, innerW))
	if sam.prompt.Tainted {
		lines = append(lines, truncateANSI(m.sty.sWarning.Render("⚠ SESSION TAINTED: web content entered this turn — inspect carefully!"), innerW))
	}
	lines = append(lines, truncateANSI(m.sty.sText.Render("Description: "+sam.descInput.Value()), innerW))
	if len(sam.prompt.Files) > 0 {
		var fnames []string
		for k := range sam.prompt.Files {
			fnames = append(fnames, k)
		}
		lines = append(lines, truncateANSI(m.sty.sMuted.Render("Extra files: "+strings.Join(fnames, ", ")), innerW))
	}
	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", innerW)))

	switch sam.mode {
	case skillApproveEditName, skillApproveEditDesc, skillApproveEditBody:
		lines = append(lines, m.sty.sAccent2Bold.Render("Edit Staged Skill (enter advances, esc returns to preview):"))
		lines = append(lines, "Name: "+sam.nameInput.View())
		if sam.mode >= skillApproveEditDesc {
			lines = append(lines, "Desc: "+sam.descInput.View())
		}
		if sam.mode == skillApproveEditBody {
			lines = append(lines, "Body: "+sam.bodyInput.View())
		}
		if sam.errText != "" {
			lines = append(lines, m.sty.sDanger.Render("Error: "+sam.errText))
		}
	case skillApproveRejectNote:
		lines = append(lines, m.sty.sWarning.Render("Reject skill with feedback for model:"))
		lines = append(lines, "  "+sam.noteInput.View())
		lines = append(lines, m.sty.sMuted.Render("enter send rejection · esc back"))
	default:
		var previewLines []string
		if sam.prompt.Diff != "" {
			previewLines = append(previewLines, m.sty.accent2().Render(fmt.Sprintf("Diff against existing skill (+%d/-%d lines):", sam.prompt.DiffAdded, sam.prompt.DiffRemoved)))
			for _, dl := range strings.Split(strings.TrimRight(sam.prompt.Diff, "\n"), "\n") {
				styled := m.sty.sMuted.Render(truncate(dl, innerW))
				if strings.HasPrefix(dl, "+") && !strings.HasPrefix(dl, "+++") {
					styled = m.sty.sSuccess.Render(truncate(dl, innerW))
				} else if strings.HasPrefix(dl, "-") && !strings.HasPrefix(dl, "---") {
					styled = m.sty.sDanger.Render(truncate(dl, innerW))
				} else if strings.HasPrefix(dl, "@@") {
					styled = m.sty.accent2().Render(truncate(dl, innerW))
				}
				previewLines = append(previewLines, styled)
			}
			previewLines = append(previewLines, "")
		}
		previewLines = append(previewLines, m.sty.sMuted.Render("Markdown body preview:"))
		previewLines = append(previewLines, renderMarkdownLines(m.sty, innerW, sam.editBody, m.codeLineNums, false, nil)...)

		maxP := h - 13
		if maxP < 4 {
			maxP = 4
		}
		if maxP > 16 {
			maxP = 16
		}
		start := sam.scroll
		if start > len(previewLines)-maxP {
			start = len(previewLines) - maxP
		}
		if start < 0 {
			start = 0
		}
		sam.scroll = start
		end := start + maxP
		if end > len(previewLines) {
			end = len(previewLines)
		}
		lines = append(lines, previewLines[start:end]...)
		if len(previewLines) > maxP {
			lines = append(lines, m.sty.sMuted.Render(fmt.Sprintf("[↑/↓ scroll preview %d-%d of %d]", start+1, end, len(previewLines))))
		}
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("[y] approve & save · [e] edit inline · [o] open staged · [r] reload · [n] reject · [d] reject w/ reason"))
	}

	box := m.sty.panel(boxW, len(lines)+3, "Stage Skill (.alpaca/skills-staging)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}

func openSkillInEditor(path string) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	ed := strings.TrimSpace(os.Getenv("ALPACA_SKILL_EDITOR"))
	if ed == "" {
		ed = strings.TrimSpace(os.Getenv("LLM_SKILL_EDITOR"))
	}
	if ed != "" {
		cmd := exec.Command(ed, abs)
		return cmd.Start()
	}
	if runtime.GOOS == "windows" {
		cmd := exec.Command("cmd.exe", "/c", "start", "", abs)
		return cmd.Start()
	}
	return exec.Command("xdg-open", abs).Start()
}

func formatLastUsed(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := time.Since(t)
	if d < time.Minute {
		return "just now"
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	}
	if d < 24*time.Hour {
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	}
	return t.UTC().Format("2006-01-02")
}
