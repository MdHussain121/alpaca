package ui

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/session"
)

const (
	atLargeFileBytes = 8 * 1024
	atMaxFileBytes   = 64 * 1024
	atDefaultEndLine = 180
)

var atMentionRe = regexp.MustCompile(`(?:^|\s)@([^\s:@]+(?:/[^\s:@]+)*)(?::(\d+)-(\d+))?`)

type contextModalState struct {
	bd session.ContextBreakdown
}

type pendingAttachFile struct {
	RawToken  string
	RelPath   string
	AbsPath   string
	ByteLen   int
	TotalLn   int
	LineRange string
	StartLine int
	EndLine   int
	Content   string
	Tokens    int
	NeedsConf bool
}

type attachModalState struct {
	promptRaw   string
	attachments []pendingAttachFile
	cursor      int
	enteringRng bool
	rangeInput  textinput.Model
	errText     string
}

// --- AGENTS.md ---

func (m *Model) effectiveAgentsMD() session.AgentsMDResult {
	return session.LoadAgentsMD(m.root, m.agentWorkDir)
}

// --- Sampling profiles (/profile) ---

func (m *Model) effectiveProfileLabel() string {
	if strings.TrimSpace(m.activeProfile) != "" {
		return m.activeProfile
	}
	if m.cfg != nil && strings.TrimSpace(m.cfg.ActiveProfile) != "" {
		return strings.ToLower(strings.TrimSpace(m.cfg.ActiveProfile))
	}
	return "coding"
}

func (m *Model) runProfileCommand(arg string) (tea.Model, tea.Cmd) {
	if m.cfg == nil {
		m.cfg = config.Default()
	}
	profiles := m.cfg.ResolvedProfiles()
	q := strings.ToLower(strings.TrimSpace(arg))
	if q == "" {
		var b strings.Builder
		b.WriteString(fmt.Sprintf("Active profile: %s\nAvailable profiles:", m.effectiveProfileLabel()))
		for _, name := range m.cfg.ProfileNames() {
			p := profiles[name]
			mark := " "
			if name == m.effectiveProfileLabel() {
				mark = "*"
			}
			b.WriteString(fmt.Sprintf("\n %s %s — temp=%.2f top_p=%.2f top_k=%d min_p=%.2f (%s)",
				mark, name, p.Temp, p.TopP, p.TopK, p.MinP, p.Description))
		}
		m.appendInfoTurn(b.String())
		return m, nil
	}

	p, ok := profiles[q]
	if !ok {
		m.appendInfoTurn(fmt.Sprintf("Unknown profile %q. Available: %s", arg, strings.Join(m.cfg.ProfileNames(), ", ")))
		return m, nil
	}
	m.activeProfile = q
	m.settings.Temp = p.Temp
	m.settings.TopP = p.TopP
	m.settings.TopK = p.TopK
	m.settings.MinP = p.MinP
	m.cfg.ActiveProfile = q
	_ = m.cfg.Save()
	m.recompute()
	m.toastf("profile: %s (temp %.2f)", q, p.Temp)
	m.appendInfoTurn(fmt.Sprintf("Switched sampling profile to %q (temp=%.2f, top_p=%.2f, top_k=%d, min_p=%.2f) for next requests.",
		q, p.Temp, p.TopP, p.TopK, p.MinP))
	return m, nil
}

// --- @file inline attachment parsing & large-file confirmation ---

func attachmentTokensOf(atts []session.AttachmentRecord) int {
	total := 0
	for _, a := range atts {
		total += a.Tokens
	}
	return total
}

func (m *Model) submitPromptWithAttachments(raw string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	matches := atMentionRe.FindAllStringSubmatchIndex(raw, -1)
	if len(matches) == 0 {
		return m.startAgentCompletion(raw)
	}

	seen := make(map[string]bool)
	var pending []pendingAttachFile
	needModal := false

	for _, loc := range matches {
		pathRaw := raw[loc[2]:loc[3]]
		sLn, eLn := 0, 0
		if loc[4] >= 0 && loc[6] >= 0 {
			sLn, _ = strconv.Atoi(raw[loc[4]:loc[5]])
			eLn, _ = strconv.Atoi(raw[loc[6]:loc[7]])
		}
		key := fmt.Sprintf("%s:%d-%d", strings.ToLower(pathRaw), sLn, eLn)
		if seen[key] {
			continue
		}
		seen[key] = true

		att, err := loadMentionAttachment(m.agentWorkDir, pathRaw, sLn, eLn)
		if err != nil {
			m.toastf("@%s: %v", pathRaw, err)
			m.appendInfoTurn(fmt.Sprintf("Attachment skipped for @%s: %v", pathRaw, err))
			m.chatInput.SetValue(raw)
			return m, nil
		}
		if att.NeedsConf {
			needModal = true
		}
		pending = append(pending, att)
	}

	if needModal {
		ri := textinput.New()
		ri.Prompt = ""
		ri.Placeholder = "1-180"
		ri.CharLimit = 24
		ri.Width = 20
		m.attachModal = &attachModalState{
			promptRaw:   raw,
			attachments: pending,
			rangeInput:  ri,
		}
		return m, nil
	}

	userTurn := buildUserTurnWithAttachments(raw, pending)
	return m.startAgentTurnWithUser(userTurn, true)
}

func loadMentionAttachment(wsRoot, relInput string, startLine, endLine int) (pendingAttachFile, error) {
	abs, rel, err := agent.ResolveWorkspacePath(wsRoot, relInput, agent.AccessRead)
	if err != nil {
		return pendingAttachFile{}, err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return pendingAttachFile{}, err
	}
	if st.IsDir() {
		return pendingAttachFile{}, fmt.Errorf("%s is a directory", rel)
	}
	if st.Size() > 2*1024*1024 {
		return pendingAttachFile{}, fmt.Errorf("%s is too large (%d KB); specify a line range or read via read_file", rel, st.Size()/1024)
	}
	raw, err := os.ReadFile(abs)
	if err != nil {
		return pendingAttachFile{}, err
	}
	head := raw
	if len(head) > 8192 {
		head = head[:8192]
	}
	if bytes.IndexByte(head, 0) >= 0 || !utf8.Valid(raw) {
		return pendingAttachFile{}, fmt.Errorf("%s appears to be binary", rel)
	}

	norm := strings.ReplaceAll(string(raw), "\r\n", "\n")
	lines := strings.Split(norm, "\n")
	totalLines := len(lines)

	var content string
	var lineRange string
	needsConf := false

	if startLine > 0 && endLine >= startLine {
		if endLine > totalLines {
			endLine = totalLines
		}
		if startLine > totalLines {
			return pendingAttachFile{}, fmt.Errorf("line %d exceeds total %d lines in %s", startLine, totalLines, rel)
		}
		slice := lines[startLine-1 : endLine]
		content = strings.Join(slice, "\n")
		lineRange = fmt.Sprintf("%d-%d", startLine, endLine)
	} else {
		content = norm
		if len(raw) > atLargeFileBytes || totalLines > 220 {
			needsConf = true
			if len(content) > atMaxFileBytes {
				content = content[:atMaxFileBytes] + "\n… [capped at 64 KB]"
			}
		}
	}

	return pendingAttachFile{
		RelPath:   rel,
		AbsPath:   abs,
		ByteLen:   len(raw),
		TotalLn:   totalLines,
		LineRange: lineRange,
		StartLine: startLine,
		EndLine:   endLine,
		Content:   content,
		Tokens:    session.EstimateTokensText(content),
		NeedsConf: needsConf,
	}, nil
}

func buildUserTurnWithAttachments(rawPrompt string, atts []pendingAttachFile) chatTurn {
	if len(atts) == 0 {
		return chatTurn{Role: "user", Content: rawPrompt}
	}
	recs := make([]session.AttachmentRecord, 0, len(atts))
	var apiBlock strings.Builder
	apiBlock.WriteString(rawPrompt)
	apiBlock.WriteString("\n\n<attached_files>")
	for _, a := range atts {
		recs = append(recs, session.AttachmentRecord{
			Path:      a.RelPath,
			LineRange: a.LineRange,
			Bytes:     len(a.Content),
			Tokens:    a.Tokens,
			Content:   a.Content,
		})
		lrAttr := ""
		if a.LineRange != "" {
			lrAttr = fmt.Sprintf(` lines=%q`, a.LineRange)
		}
		apiBlock.WriteString(fmt.Sprintf("\n<file path=%q%s tokens=\"%d\">\n%s\n</file>",
			a.RelPath, lrAttr, a.Tokens, a.Content))
	}
	apiBlock.WriteString("\n</attached_files>")

	return chatTurn{
		Role:        "user",
		Content:     rawPrompt,
		Attachments: recs,
		APIMessages: []runner.APIMessage{{Role: "user", Content: apiBlock.String()}},
	}
}

func (m *Model) handleAttachModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	am := m.attachModal
	if am == nil {
		return m, nil
	}
	if am.enteringRng {
		switch msg.Type {
		case tea.KeyEsc:
			am.enteringRng = false
			am.errText = ""
			return m, nil
		case tea.KeyEnter:
			parts := strings.Split(strings.TrimSpace(am.rangeInput.Value()), "-")
			if len(parts) != 2 {
				am.errText = "use start-end format, e.g. 1-120"
				return m, nil
			}
			s, e1 := strconv.Atoi(strings.TrimSpace(parts[0]))
			e, e2 := strconv.Atoi(strings.TrimSpace(parts[1]))
			if e1 != nil || e2 != nil || s < 1 || e < s {
				am.errText = "invalid line range"
				return m, nil
			}
			cur := &am.attachments[am.cursor]
			updated, err := loadMentionAttachment(m.agentWorkDir, cur.RelPath, s, e)
			if err != nil {
				am.errText = err.Error()
				return m, nil
			}
			updated.NeedsConf = false
			am.attachments[am.cursor] = updated
			am.enteringRng = false
			am.errText = ""
			m.attachModal = nil
			userTurn := buildUserTurnWithAttachments(am.promptRaw, am.attachments)
			return m.startAgentTurnWithUser(userTurn, true)
		}
		var c tea.Cmd
		am.rangeInput, c = am.rangeInput.Update(msg)
		return m, c
	}

	switch msg.String() {
	case "esc":
		m.chatInput.SetValue(am.promptRaw)
		m.attachModal = nil
		return m, nil
	case "a", "enter":
		am.attachments[am.cursor].NeedsConf = false
		m.attachModal = nil
		userTurn := buildUserTurnWithAttachments(am.promptRaw, am.attachments)
		return m.startAgentTurnWithUser(userTurn, true)
	case "r":
		am.enteringRng = true
		endDef := atDefaultEndLine
		if am.attachments[am.cursor].TotalLn > 0 && am.attachments[am.cursor].TotalLn < endDef {
			endDef = am.attachments[am.cursor].TotalLn
		}
		am.rangeInput.SetValue(fmt.Sprintf("1-%d", endDef))
		am.rangeInput.Focus()
		am.errText = ""
		return m, nil
	case "s":
		// Skip large attachment and send prompt as-is
		raw := am.promptRaw
		m.attachModal = nil
		return m.startAgentCompletion(raw)
	}
	return m, nil
}

func (m *Model) overlayAttachModal(bg string) string {
	am := m.attachModal
	if am == nil || am.cursor >= len(am.attachments) {
		return bg
	}
	att := am.attachments[am.cursor]
	w, h := m.width, m.height
	boxW := 74
	if boxW > w-6 {
		boxW = w - 6
	}
	if boxW < 42 {
		boxW = 42
	}
	innerW := boxW - 4

	lines := []string{
		truncateANSI(m.sty.sWarning.Bold(true).Render(fmt.Sprintf("Large File Attachment: @%s", att.RelPath)), innerW),
		truncateANSI(m.sty.sText.Render(fmt.Sprintf("Size: %.1f KB · %d lines · ~%d tokens", float64(att.ByteLen)/1024, att.TotalLn, att.Tokens)), innerW),
		m.sty.sMuted.Render(strings.Repeat("─", innerW)),
	}
	if am.enteringRng {
		lines = append(lines, "Enter line range (start-end): "+am.rangeInput.View())
		if am.errText != "" {
			lines = append(lines, m.sty.sDanger.Render("Error: "+am.errText))
		}
		lines = append(lines, m.sty.sMuted.Render("enter apply range · esc back"))
	} else {
		lines = append(lines, m.sty.sMuted.Render("Large files consume context window budget quickly."))
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("[a] attach full · [r] attach line range · [s] skip file · [esc] cancel"))
	}
	box := m.sty.panel(boxW, len(lines)+3, "Confirm @file Attachment", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}

// --- Context Breakdown (/context) & Tiered Compaction (/compact) ---

func (m *Model) computeContextBreakdown() session.ContextBreakdown {
	m.ensureAgentReady()
	if m.breakdownCache == nil {
		m.breakdownCache = session.NewBreakdownCache()
	}
	agentsRes := m.effectiveAgentsMD()
	skillBlock, _, _ := m.skillsCatalog.SystemPromptBlock()
	var tools []runner.ToolDef
	if m.toolRegistry != nil {
		tools = m.toolRegistry.ToolDefs()
	}
	apiMsgs := m.buildAPIMessagesForLoop()

	liveURL := ""
	if m.status == "READY" || m.chatBaseURL != "" {
		liveURL = m.effectiveBaseURL()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	bd := session.MeasureBreakdown(
		ctx,
		liveURL,
		m.breakdownCache,
		m.effectiveSystemBaseForBreakdown(),
		agentsRes.CombinedPrompt,
		skillBlock,
		tools,
		apiMsgs,
		m.attachmentTokens,
		m.effectiveCtxTotal(),
		m.cachedPrefixTokens,
	)
	if bd.TotalPromptTokens > 0 {
		m.chatCtxUsed = bd.TotalPromptTokens
	}
	return bd
}

func (m *Model) runContextCommand(arg string) (tea.Model, tea.Cmd) {
	sub := strings.ToLower(strings.TrimSpace(arg))
	if sub == "compact" {
		return m.runCompactCommand("")
	}
	bd := m.computeContextBreakdown()
	if sub == "print" || sub == "text" {
		m.appendInfoTurn(formatBreakdownText(bd))
		return m, nil
	}
	m.contextModal = &contextModalState{bd: bd}
	return m, nil
}

func formatBreakdownText(bd session.ContextBreakdown) string {
	src := "estimated"
	if bd.LiveMeasured {
		src = "measured via /input_tokens"
	}
	pct := 0
	if bd.ContextLimit > 0 {
		pct = bd.TotalPromptTokens * 100 / bd.ContextLimit
	}
	return fmt.Sprintf(
		"Context Breakdown (%s):\n  System prompt: %d tok\n  AGENTS.md: %d tok\n  Tool definitions: %d tok\n  Skill list: %d tok\n  Attachments (@file): %d tok\n  Conversation history: %d tok\n  Total prompt: %d / %d tok (%d%%) · Cached prefix: %d tok",
		src, bd.SystemTokens, bd.AgentsMDTokens, bd.ToolDefsTokens, bd.SkillListTokens,
		bd.AttachmentsTokens, bd.HistoryTokens, bd.TotalPromptTokens, bd.ContextLimit, pct, bd.CachedTokens,
	)
}

func (m *Model) handleContextModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q", "enter":
		m.contextModal = nil
	case "c":
		m.contextModal = nil
		return m.runCompactCommand("")
	case "p":
		bd := m.contextModal.bd
		m.contextModal = nil
		m.appendInfoTurn(formatBreakdownText(bd))
	}
	return m, nil
}

func (m *Model) overlayContextModal(bg string) string {
	cm := m.contextModal
	if cm == nil {
		return bg
	}
	bd := cm.bd
	w, h := m.width, m.height
	boxW := 72
	if boxW > w-6 {
		boxW = w - 6
	}
	if boxW < 42 {
		boxW = 42
	}
	innerW := boxW - 4

	limit := bd.ContextLimit
	if limit <= 0 {
		limit = 8192
	}
	pct := float64(bd.TotalPromptTokens) * 100 / float64(limit)
	if pct > 100 {
		pct = 100
	}
	srcTag := "heuristic"
	if bd.LiveMeasured {
		srcTag = "live /input_tokens"
	}

	row := func(label string, tok int) string {
		r := float64(tok) / float64(limit)
		if r < 0 {
			r = 0
		}
		if r > 1 {
			r = 1
		}
		bar := m.sty.gauge(14, r)
		return truncateANSI(fmt.Sprintf("%-20s %s  %6d tok", label, bar, tok), innerW)
	}

	thresh := 80
	if m.cfg != nil {
		thresh = m.cfg.EffectiveCompactionPct()
	}

	lines := []string{
		truncateANSI(m.sty.sAccentBold.Render(fmt.Sprintf("Prompt Usage: %d / %d tok (%.0f%%) · %s", bd.TotalPromptTokens, limit, pct, srcTag)), innerW),
		m.sty.sMuted.Render(strings.Repeat("─", innerW)),
		row("Base System", bd.SystemTokens),
		row("AGENTS.md", bd.AgentsMDTokens),
		row("Tool Definitions", bd.ToolDefsTokens),
		row("Skill List", bd.SkillListTokens),
		row("Attachments (@file)", bd.AttachmentsTokens),
		row("Chat History", bd.HistoryTokens),
		m.sty.sMuted.Render(strings.Repeat("─", innerW)),
		truncateANSI(m.sty.accent2().Render(fmt.Sprintf("KV Prompt Cache Hits: %d tok · Auto-compact threshold: %d%%", bd.CachedTokens, thresh)), innerW),
		"",
		m.sty.sMuted.Render("[c] compact now · [p] print to chat · [esc] close"),
	}
	box := m.sty.panel(boxW, len(lines)+3, "Context Breakdown (/context)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}

func (m *Model) maybeShouldCompactNow() bool {
	if len(m.chatMsgs) < 6 {
		return false
	}
	limit := m.effectiveCtxTotal()
	if limit <= 0 {
		return false
	}
	threshPct := 80
	if m.cfg != nil {
		threshPct = m.cfg.EffectiveCompactionPct()
	}
	apiMsgs := m.buildAPIMessagesForLoop()
	used := m.chatCtxUsed
	heur := session.EstimateTokensText("")
	for _, msg := range apiMsgs {
		heur += 4 + session.EstimateTokensText(msg.Content)
	}
	if heur > used {
		used = heur
	}
	return used*100 >= limit*threshPct
}

func (m *Model) runCompactCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	applied, msg := m.applyCompaction(true)
	if !applied {
		m.appendInfoTurn(msg)
		return m, nil
	}
	m.saveCurrentSession()
	m.toastf("compaction complete")
	return m, nil
}

func (m *Model) applyCompaction(force bool) (bool, string) {
	apiMsgs := m.buildAPIMessagesForLoop()
	if len(apiMsgs) < 4 {
		return false, "Conversation is still short; nothing to compact yet."
	}
	baseURL := ""
	if m.status == "READY" || m.chatBaseURL != "" {
		baseURL = m.effectiveBaseURL()
	}
	keepTurns := 3
	if len(apiMsgs) <= 6 {
		keepTurns = 1
	}
	limit := m.effectiveCtxTotal()
	targetMax := (limit * 65) / 100
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	res := session.CompactHistory(ctx, baseURL, apiMsgs, keepTurns, force, targetMax)
	if !res.Applied {
		return false, "Unable to compact further without dropping recent turns."
	}

	// Preserve older turns visibly in chat scrollback while marking them CompactedOut for API history,
	// keeping recent turns intact after a collapsible compaction card.
	keepRecentCount := keepTurns * 2
	if keepRecentCount > len(m.chatMsgs) {
		keepRecentCount = len(m.chatMsgs)
	}
	cutUI := findLastNUsersUI(m.chatMsgs, keepTurns)

	if res.Tier2Summarized && cutUI > 0 {
		var newTurns []chatTurn
		for i := 0; i < cutUI; i++ {
			cp := m.chatMsgs[i]
			cp.CompactedOut = true
			newTurns = append(newTurns, cp)
		}
		expanded := false
		card := chatTurn{
			Role:                "compaction",
			Content:             res.CardTitle,
			IsCompactionSummary: true,
			CompactionDetail:    res.CardBody,
			ThinkExpanded:       &expanded,
			APIMessages:         res.CompactedMsgs[:1],
		}
		newTurns = append(newTurns, card)
		newTurns = append(newTurns, m.chatMsgs[cutUI:]...)
		m.chatMsgs = newTurns
	} else {
		// Tier 1 only: replace older APIMessages in pre-cut turns cleanly and insert info card
		for i := 0; i < cutUI && i < len(m.chatMsgs); i++ {
			for j := range m.chatMsgs[i].APIMessages {
				am := &m.chatMsgs[i].APIMessages[j]
				if am.Role == "tool" && len(am.Content) > 180 && !strings.HasPrefix(am.Content, "[output omitted, ") {
					am.Content = fmt.Sprintf("[output omitted, %d bytes]", len(am.Content))
				}
			}
		}
		expanded := false
		card := chatTurn{
			Role:                "compaction",
			Content:             res.CardTitle,
			IsCompactionSummary: true,
			CompactionDetail:    res.CardBody,
			ThinkExpanded:       &expanded,
		}
		var newTurns []chatTurn
		newTurns = append(newTurns, m.chatMsgs[:cutUI]...)
		newTurns = append(newTurns, card)
		newTurns = append(newTurns, m.chatMsgs[cutUI:]...)
		m.chatMsgs = newTurns
	}

	// Recompute context usage estimate cleanly so UI badge updates right away
	m.chatCtxUsed = res.AfterTokensEst + m.toolSchemaTokens + m.skillListTokens + m.effectiveAgentsMD().TokensEst
	m.cachedPrefixTokens = 0 // prefix restarted cleanly
	m.refreshChatView(true)
	return true, res.CardTitle
}

func findLastNUsersUI(turns []chatTurn, n int) int {
	seen := 0
	for i := len(turns) - 1; i >= 0; i-- {
		if turns[i].Role == "user" {
			seen++
			if seen >= n {
				return i
			}
		}
	}
	return 0
}
