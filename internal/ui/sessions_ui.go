package ui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/session"
)

type sessionsModalMode int

const (
	sessionsModeList sessionsModalMode = iota
	sessionsModeSearch
	sessionsModeRename
)

type sessionsModalState struct {
	items   []session.SessionSummary
	cursor  int
	mode    sessionsModalMode
	query   string
	input   textinput.Model
	errText string
}

func (m *Model) effectiveSessionsDir() string {
	if strings.TrimSpace(m.sessionsDirOv) != "" {
		return m.sessionsDirOv
	}
	return session.SessionsRoot(m.root, m.agentWorkDir)
}

func (m *Model) ensureSessionDoc() *session.SessionDoc {
	m.ensureAgentReady()
	if m.sessionDoc == nil {
		modelName := ""
		if m.meta != nil {
			modelName = m.meta.DisplayName()
		}
		presetName := ""
		if m.presetIdx >= 0 && m.presetIdx < len(m.presets) {
			presetName = m.presets[m.presetIdx].Name
		}
		m.sessionDoc = session.NewSessionDoc(m.sessionID, m.agentWorkDir, modelName, presetName, m.effectiveProfileLabel())
	}
	return m.sessionDoc
}

func chatTurnsToBranchTurns(turns []chatTurn, branchID string) []session.BranchTurn {
	out := make([]session.BranchTurn, 0, len(turns))
	for i, t := range turns {
		tcs := make([]session.ToolCallRecord, 0, len(t.ToolCalls))
		for _, tc := range t.ToolCalls {
			tcs = append(tcs, session.ToolCallRecord{
				CallID:      tc.CallID,
				Name:        tc.Name,
				Summary:     tc.Summary,
				ArgsPretty:  tc.ArgsPretty,
				Output:      tc.Output,
				Diff:        tc.Diff,
				DiffAdded:   tc.DiffAdded,
				DiffRemoved: tc.DiffRemoved,
				Status:      tc.Status,
				DurationMs:  tc.Duration.Milliseconds(),
			})
		}
		out = append(out, session.BranchTurn{
			TurnID:              fmt.Sprintf("%s-t%d", branchID, i+1),
			BranchID:            branchID,
			Role:                t.Role,
			Reasoning:           t.Reasoning,
			Content:             t.Content,
			Attachments:         append([]session.AttachmentRecord(nil), t.Attachments...),
			ToolCalls:           tcs,
			APIMessages:         append([]runner.APIMessage(nil), t.APIMessages...),
			Tokens:              t.Tokens,
			ThinkTokens:         t.ThinkTokens,
			GenTokSec:           t.GenTokSec,
			Aborted:             t.Aborted,
			Err:                 t.Err,
			IsCompactionSummary: t.IsCompactionSummary,
			CompactionDetail:    t.CompactionDetail,
			Timestamp:           time.Now().UTC(),
		})
	}
	return out
}

func branchTurnsToChatTurns(turns []session.BranchTurn) []chatTurn {
	out := make([]chatTurn, 0, len(turns))
	for _, bt := range turns {
		tcs := make([]toolCard, 0, len(bt.ToolCalls))
		for _, tc := range bt.ToolCalls {
			tcs = append(tcs, toolCard{
				CallID:      tc.CallID,
				Name:        tc.Name,
				Summary:     tc.Summary,
				ArgsPretty:  tc.ArgsPretty,
				Output:      tc.Output,
				Diff:        tc.Diff,
				DiffAdded:   tc.DiffAdded,
				DiffRemoved: tc.DiffRemoved,
				Status:      tc.Status,
				Duration:    time.Duration(tc.DurationMs) * time.Millisecond,
			})
		}
		ct := chatTurn{
			Role:                bt.Role,
			Reasoning:           bt.Reasoning,
			Content:             bt.Content,
			Attachments:         append([]session.AttachmentRecord(nil), bt.Attachments...),
			ToolCalls:           tcs,
			APIMessages:         append([]runner.APIMessage(nil), bt.APIMessages...),
			Tokens:              bt.Tokens,
			ThinkTokens:         bt.ThinkTokens,
			GenTokSec:           bt.GenTokSec,
			ThinkDone:           true,
			Aborted:             bt.Aborted,
			Err:                 bt.Err,
			IsCompactionSummary: bt.IsCompactionSummary,
			CompactionDetail:    bt.CompactionDetail,
		}
		if ct.Role == "compaction" {
			exp := false
			ct.ThinkExpanded = &exp
		}
		out = append(out, ct)
	}
	// Mark turns before latest compaction summary card CompactedOut for API history rebuild.
	lastCompactIdx := -1
	for i := range out {
		if out[i].Role == "compaction" && len(out[i].APIMessages) > 0 {
			lastCompactIdx = i
		}
	}
	for i := 0; i < lastCompactIdx; i++ {
		out[i].CompactedOut = true
	}
	return out
}

func (m *Model) saveCurrentSession() {
	if len(m.chatMsgs) == 0 {
		return
	}
	doc := m.ensureSessionDoc()
	doc.Meta.Workspace = m.agentWorkDir
	doc.Meta.Profile = m.effectiveProfileLabel()
	doc.Meta.Tainted = m.sessionTainted
	if m.meta != nil {
		doc.Meta.Model = m.meta.DisplayName()
	}
	doc.ReplaceActiveTurns(chatTurnsToBranchTurns(m.chatMsgs, doc.Meta.ActiveBranch))
	_ = session.SaveSession(m.effectiveSessionsDir(), doc)
}

func (m *Model) loadSessionIntoUI(doc *session.SessionDoc) {
	m.stopChatStream()
	m.sessionDoc = doc
	m.sessionID = doc.Meta.ID
	m.sessionTainted = doc.Meta.Tainted
	if doc.Meta.Workspace != "" {
		if st, err := os.Stat(doc.Meta.Workspace); err == nil && st.IsDir() {
			m.agentWorkDir = doc.Meta.Workspace
			m.checkpoints = agent.NewCheckpointStore(m.agentWorkDir, m.sessionID)
		}
	}
	if doc.Meta.Profile != "" {
		m.activeProfile = doc.Meta.Profile
	}
	// Loading a session never auto-runs tools.
	m.chatMsgs = branchTurnsToChatTurns(doc.ActiveTurns())
	m.attachmentTokens = 0
	for _, t := range m.chatMsgs {
		m.attachmentTokens += attachmentTokensOf(t.Attachments)
	}
	// Estimate restored context tokens cleanly
	used := m.toolSchemaTokens + m.skillListTokens + m.effectiveAgentsMD().TokensEst
	for _, msg := range m.buildAPIMessagesForLoop() {
		used += 4 + session.EstimateTokensText(msg.Content)
	}
	m.chatCtxUsed = used
	m.refreshChatView(true)
	m.chatView.GotoBottom()
}

// --- /session command ---

func (m *Model) runSessionCommand(arg string) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	dir := m.effectiveSessionsDir()
	trimmed := strings.TrimSpace(arg)
	if trimmed == "" {
		items, _ := session.ListSessions(dir)
		m.sessionsModal = &sessionsModalState{items: items}
		return m, nil
	}

	fields := strings.Fields(trimmed)
	sub := strings.ToLower(fields[0])
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, fields[0]))

	switch sub {
	case "new", "clear":
		m.saveCurrentSession()
		m.resetChat()
		m.toastf("started new session %s", m.sessionID)
	case "list":
		items, _ := session.ListSessions(dir)
		if len(items) == 0 {
			m.appendInfoTurn("No saved sessions in " + dir)
			return m, nil
		}
		var b strings.Builder
		b.WriteString(fmt.Sprintf("Saved sessions (%d):", len(items)))
		for _, it := range items {
			b.WriteString(fmt.Sprintf("\n  - %s · %s (%d turns, %d br)", it.ID, it.Title, it.TurnCount, it.BranchCount))
		}
		m.appendInfoTurn(b.String())
	case "search":
		hits, _ := session.SearchSessions(dir, rest)
		if len(hits) == 0 {
			m.appendInfoTurn(fmt.Sprintf("No sessions matched %q.", rest))
			return m, nil
		}
		var b strings.Builder
		b.WriteString(fmt.Sprintf("Matches for %q (%d):", rest, len(hits)))
		for _, it := range hits {
			b.WriteString(fmt.Sprintf("\n  - %s · %s — %s", it.ID, it.Title, it.Snippet))
		}
		m.appendInfoTurn(b.String())
	case "load":
		id := strings.TrimSpace(rest)
		if id == "" {
			m.appendInfoTurn("Usage: /session load <session-id>")
			return m, nil
		}
		doc, err := session.LoadSessionByID(dir, id)
		if err != nil {
			m.appendInfoTurn(fmt.Sprintf("Failed to load session %q: %v", id, err))
			return m, nil
		}
		m.loadSessionIntoUI(doc)
		m.toastf("loaded session %s (read-only restore)", doc.Meta.ID)
	case "rename":
		if m.sessionDoc == nil {
			m.saveCurrentSession()
		}
		if err := session.RenameSession(dir, m.sessionID, rest); err != nil {
			m.appendInfoTurn(fmt.Sprintf("Rename failed: %v", err))
			return m, nil
		}
		m.sessionDoc.Meta.Title = rest
		m.toastf("session renamed to %q", rest)
	case "delete":
		id := strings.TrimSpace(rest)
		if id == "" {
			id = m.sessionID
		}
		if err := session.DeleteSession(dir, id); err != nil {
			m.appendInfoTurn(fmt.Sprintf("Delete failed: %v", err))
			return m, nil
		}
		m.toastf("deleted session %s", id)
	case "export":
		doc := m.ensureSessionDoc()
		doc.ReplaceActiveTurns(chatTurnsToBranchTurns(m.chatMsgs, doc.Meta.ActiveBranch))
		md := session.ExportMarkdown(doc)
		outPath := filepath.Join(dir, doc.Meta.ID+".md")
		_ = os.MkdirAll(dir, 0o755)
		if err := os.WriteFile(outPath, []byte(md), 0o644); err != nil {
			m.appendInfoTurn(fmt.Sprintf("Export failed: %v", err))
			return m, nil
		}
		m.appendInfoTurn("Exported Markdown transcript to " + outPath)
	case "branch":
		if m.sessionDoc == nil {
			m.appendInfoTurn("No branched session loaded yet.")
			return m, nil
		}
		if rest == "" {
			var b strings.Builder
			b.WriteString(fmt.Sprintf("Branches (active: %s):", m.sessionDoc.Meta.ActiveBranch))
			for _, br := range m.sessionDoc.Meta.Branches {
				b.WriteString(fmt.Sprintf("\n  - %s (%s)", br.ID, br.Name))
			}
			m.appendInfoTurn(b.String())
			return m, nil
		}
		if err := m.sessionDoc.SwitchBranch(rest); err != nil {
			m.appendInfoTurn(err.Error())
			return m, nil
		}
		m.loadSessionIntoUI(m.sessionDoc)
		m.toastf("switched to branch %s", rest)
	default:
		// Treat `/session <id>` shorthand as load
		if doc, err := session.LoadSessionByID(dir, trimmed); err == nil {
			m.loadSessionIntoUI(doc)
			m.toastf("loaded session %s", doc.Meta.ID)
		} else {
			m.appendInfoTurn("Usage: /session [new|list|search <q>|load <id>|rename <title>|delete <id>|export|branch <id>]")
		}
	}
	return m, nil
}

// --- /edit and /regen branching ---

func (m *Model) runEditTurnCommand(arg string) (tea.Model, tea.Cmd) {
	newPrompt := strings.TrimSpace(arg)
	if newPrompt == "" {
		m.appendInfoTurn("Usage: /edit <new prompt text> — forks a new branch replacing the last user prompt.")
		return m, nil
	}
	if m.chatStreaming {
		m.toastf("wait for current generation or press esc first")
		return m, nil
	}
	lastUserIdx := -1
	for i := len(m.chatMsgs) - 1; i >= 0; i-- {
		if m.chatMsgs[i].Role == "user" {
			lastUserIdx = i
			break
		}
	}
	if lastUserIdx < 0 {
		m.appendInfoTurn("No previous user message to edit.")
		return m, nil
	}
	doc := m.ensureSessionDoc()
	doc.ReplaceActiveTurns(chatTurnsToBranchTurns(m.chatMsgs, doc.Meta.ActiveBranch))
	brID, prefix := doc.ForkBeforeActiveIndex(lastUserIdx, "edit")
	m.chatMsgs = branchTurnsToChatTurns(prefix)
	_ = session.SaveSession(m.effectiveSessionsDir(), doc)
	m.toastf("forked branch %s", brID)
	if m.status == "READY" || m.chatBaseURL != "" {
		return m.submitPromptWithAttachments(newPrompt)
	}
	m.chatMsgs = append(m.chatMsgs, chatTurn{Role: "user", Content: newPrompt})
	m.saveCurrentSession()
	m.refreshChatView(true)
	return m, nil
}

func (m *Model) runRegenerateCommand() (tea.Model, tea.Cmd) {
	if m.chatStreaming {
		m.toastf("wait for current generation or press esc first")
		return m, nil
	}
	lastUserIdx := -1
	for i := len(m.chatMsgs) - 1; i >= 0; i-- {
		if m.chatMsgs[i].Role == "user" {
			lastUserIdx = i
			break
		}
	}
	if lastUserIdx < 0 {
		m.appendInfoTurn("No previous user turn to regenerate.")
		return m, nil
	}
	userTurn := m.chatMsgs[lastUserIdx]
	doc := m.ensureSessionDoc()
	doc.ReplaceActiveTurns(chatTurnsToBranchTurns(m.chatMsgs, doc.Meta.ActiveBranch))
	brID, prefix := doc.ForkBeforeActiveIndex(lastUserIdx+1, "regen")
	m.chatMsgs = branchTurnsToChatTurns(prefix)
	_ = session.SaveSession(m.effectiveSessionsDir(), doc)
	m.toastf("regenerating on branch %s", brID)
	if m.status == "READY" || m.chatBaseURL != "" {
		// Re-run last user turn (already in prefix -> appendUserTurn=false)
		_ = userTurn
		return m.startAgentTurnWithUser(chatTurn{}, false)
	}
	m.refreshChatView(true)
	return m, nil
}

// --- /sessions interactive modal ---

func (m *Model) handleSessionsModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	sm := m.sessionsModal
	if sm == nil {
		return m, nil
	}
	switch sm.mode {
	case sessionsModeSearch:
		switch msg.Type {
		case tea.KeyEsc:
			sm.mode = sessionsModeList
			return m, nil
		case tea.KeyEnter:
			sm.query = strings.TrimSpace(sm.input.Value())
			sm.items, _ = session.SearchSessions(m.effectiveSessionsDir(), sm.query)
			sm.cursor = 0
			sm.mode = sessionsModeList
			return m, nil
		}
		var c tea.Cmd
		sm.input, c = sm.input.Update(msg)
		return m, c

	case sessionsModeRename:
		switch msg.Type {
		case tea.KeyEsc:
			sm.mode = sessionsModeList
			sm.errText = ""
			return m, nil
		case tea.KeyEnter:
			if sm.cursor < len(sm.items) {
				target := sm.items[sm.cursor]
				if err := session.RenameSession(m.effectiveSessionsDir(), target.ID, sm.input.Value()); err != nil {
					sm.errText = err.Error()
					return m, nil
				}
				sm.items, _ = session.SearchSessions(m.effectiveSessionsDir(), sm.query)
			}
			sm.mode = sessionsModeList
			sm.errText = ""
			return m, nil
		}
		var c tea.Cmd
		sm.input, c = sm.input.Update(msg)
		return m, c
	}

	switch msg.String() {
	case "esc", "q":
		m.sessionsModal = nil
		return m, nil
	case "up", "k":
		if sm.cursor > 0 {
			sm.cursor--
		}
	case "down", "j":
		if sm.cursor < len(sm.items)-1 {
			sm.cursor++
		}
	case "/":
		ti := textinput.New()
		ti.Prompt = ""
		ti.Placeholder = "search title or message text..."
		ti.Width = 44
		ti.SetValue(sm.query)
		ti.Focus()
		sm.input = ti
		sm.mode = sessionsModeSearch
	case "enter", "l":
		if sm.cursor < len(sm.items) {
			doc, err := session.LoadSession(sm.items[sm.cursor].Path)
			if err != nil {
				m.toastf("load error: %v", err)
			} else {
				m.loadSessionIntoUI(doc)
				m.sessionsModal = nil
				m.toastf("loaded %s", doc.Meta.ID)
			}
		}
	case "n":
		m.saveCurrentSession()
		m.resetChat()
		m.sessionsModal = nil
		m.toastf("started new session")
	case "r":
		if sm.cursor < len(sm.items) {
			ti := textinput.New()
			ti.Prompt = ""
			ti.Width = 44
			ti.SetValue(sm.items[sm.cursor].Title)
			ti.Focus()
			sm.input = ti
			sm.mode = sessionsModeRename
		}
	case "d", "x":
		if sm.cursor < len(sm.items) {
			_ = session.DeleteSession(m.effectiveSessionsDir(), sm.items[sm.cursor].ID)
			sm.items, _ = session.SearchSessions(m.effectiveSessionsDir(), sm.query)
			if sm.cursor > 0 && sm.cursor >= len(sm.items) {
				sm.cursor--
			}
			m.toastf("deleted session")
		}
	case "e":
		if sm.cursor < len(sm.items) {
			doc, err := session.LoadSession(sm.items[sm.cursor].Path)
			if err == nil {
				outPath := filepath.Join(m.effectiveSessionsDir(), doc.Meta.ID+".md")
				_ = os.WriteFile(outPath, []byte(session.ExportMarkdown(doc)), 0o644)
				m.toastf("exported %s", filepath.Base(outPath))
			}
		}
	}
	return m, nil
}

func (m *Model) overlaySessionsModal(bg string) string {
	sm := m.sessionsModal
	if sm == nil {
		return bg
	}
	w, h := m.width, m.height
	boxW := w - 8
	if boxW > 86 {
		boxW = 86
	}
	if boxW < 46 {
		boxW = 46
	}
	innerW := boxW - 4

	var lines []string
	qTag := ""
	if sm.query != "" {
		qTag = fmt.Sprintf(" · filter:%q", sm.query)
	}
	lines = append(lines, truncateANSI(m.sty.sAccentBold.Render(fmt.Sprintf("Saved Sessions (%d)%s", len(sm.items), qTag)), innerW))
	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", innerW)))

	switch sm.mode {
	case sessionsModeSearch:
		lines = append(lines, "Search: "+sm.input.View())
		lines = append(lines, m.sty.sMuted.Render("enter filter · esc back"))
	case sessionsModeRename:
		lines = append(lines, "Rename title: "+sm.input.View())
		if sm.errText != "" {
			lines = append(lines, m.sty.sDanger.Render("Error: "+sm.errText))
		}
		lines = append(lines, m.sty.sMuted.Render("enter save · esc back"))
	default:
		if len(sm.items) == 0 {
			lines = append(lines, m.sty.sMuted.Render("No saved sessions yet."))
		} else {
			maxRows := h - 12
			if maxRows < 4 {
				maxRows = 4
			}
			if maxRows > 12 {
				maxRows = 12
			}
			start := 0
			if sm.cursor >= maxRows {
				start = sm.cursor - maxRows + 1
			}
			end := start + maxRows
			if end > len(sm.items) {
				end = len(sm.items)
			}
			for i := start; i < end; i++ {
				it := sm.items[i]
				cur := "  "
				if i == sm.cursor {
					cur = m.sty.sAccentBold.Render("▸ ")
				}
				activeTag := ""
				if it.ID == m.sessionID {
					activeTag = " [active]"
				}
				brInfo := ""
				if it.BranchCount > 1 {
					brInfo = fmt.Sprintf(", %d br", it.BranchCount)
				}
				row := fmt.Sprintf("%s%s%s (%d turns%s) — %s",
					cur, m.sty.sText.Bold(true).Render(it.Title),
					m.sty.accent2().Render(activeTag),
					it.TurnCount, brInfo, m.sty.sMuted.Render(it.ID))
				lines = append(lines, truncateANSI(row, innerW))
			}
		}
		lines = append(lines, "")
		lines = append(lines, m.sty.sMuted.Render("↑↓ select · enter load · / search · n new · r rename · d delete · e export md · esc close"))
	}

	box := m.sty.panel(boxW, len(lines)+3, "Sessions (/session)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}
