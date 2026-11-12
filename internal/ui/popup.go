package ui

import (
	"fmt"
	"sort"
	"strings"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"
)

const popupMaxVisible = 8

type popupKind int

const (
	popupNone popupKind = iota
	popupSlashCmd
	popupSlashArg
	popupAtFile
)

// popupItem is a single selectable row in the shared slash / argument / @file popup.
type popupItem struct {
	ID             string
	Label          string
	Description    string
	ArgsHint       string
	Disabled       bool
	DisabledReason string
	MatchPositions []int
}

type scoredPopupItem struct {
	item   popupItem
	tier   int // 0 exact, 1 prefix, 2 fuzzy
	score  int
	origIx int
}

type suggestionPopup struct {
	open         bool
	kind         popupKind
	title        string
	cmdName      string
	tokenStart   int // byte offset for @token replacement
	tokenEnd     int
	items        []popupItem
	idx          int
	scroll       int
	dismissedFor string
}

func (p *suggestionPopup) closeKeepDismissed(inputVal string) {
	p.open = false
	p.kind = popupNone
	p.items = nil
	p.idx = 0
	p.scroll = 0
	p.dismissedFor = inputVal
}

func (p *suggestionPopup) reset() {
	p.open = false
	p.kind = popupNone
	p.title = ""
	p.cmdName = ""
	p.items = nil
	p.idx = 0
	p.scroll = 0
	p.dismissedFor = ""
}

func (p *suggestionPopup) move(delta int) {
	if len(p.items) == 0 {
		return
	}
	n := len(p.items)
	p.idx = ((p.idx+delta)%n + n) % n
	p.ensureVisible()
}

func (p *suggestionPopup) ensureVisible() {
	if p.idx < p.scroll {
		p.scroll = p.idx
	}
	if p.idx >= p.scroll+popupMaxVisible {
		p.scroll = p.idx - popupMaxVisible + 1
	}
	maxScroll := len(p.items) - popupMaxVisible
	if maxScroll < 0 {
		maxScroll = 0
	}
	if p.scroll > maxScroll {
		p.scroll = maxScroll
	}
	if p.scroll < 0 {
		p.scroll = 0
	}
}

func (p *suggestionPopup) selected() (popupItem, bool) {
	if !p.open || p.idx < 0 || p.idx >= len(p.items) {
		return popupItem{}, false
	}
	return p.items[p.idx], true
}

// rankPopupItems filters and ranks candidates: exact -> prefix -> fuzzy subsequence.
func rankPopupItems(query string, raw []popupItem) []popupItem {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		out := make([]popupItem, len(raw))
		copy(out, raw)
		return out
	}

	var scored []scoredPopupItem
	for i, it := range raw {
		tier, sc, pos, ok := matchQuery(q, strings.ToLower(it.Label))
		if !ok && it.ID != it.Label {
			if t2, s2, _, ok2 := matchQuery(q, strings.ToLower(it.ID)); ok2 {
				tier, sc, pos, ok = t2, s2+10, nil, true
			}
		}
		if !ok {
			continue
		}
		cp := it
		cp.MatchPositions = pos
		scored = append(scored, scoredPopupItem{
			item:   cp,
			tier:   tier,
			score:  sc,
			origIx: i,
		})
	}

	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].tier != scored[j].tier {
			return scored[i].tier < scored[j].tier
		}
		if scored[i].item.Disabled != scored[j].item.Disabled {
			return !scored[i].item.Disabled
		}
		if scored[i].score != scored[j].score {
			return scored[i].score < scored[j].score
		}
		return scored[i].origIx < scored[j].origIx
	})

	out := make([]popupItem, len(scored))
	for i, s := range scored {
		out[i] = s.item
	}
	return out
}

func matchQuery(q, target string) (tier int, score int, pos []int, ok bool) {
	if q == "" {
		return 0, 0, nil, true
	}
	qr := []rune(q)
	tr := []rune(target)
	if len(qr) > len(tr) {
		return 0, 0, nil, false
	}
	if target == q {
		pos = make([]int, len(qr))
		for i := range qr {
			pos[i] = i
		}
		return 0, 0, pos, true
	}
	if strings.HasPrefix(target, q) {
		pos = make([]int, len(qr))
		for i := range qr {
			pos[i] = i
		}
		return 1, len(tr) - len(qr), pos, true
	}

	// Fuzzy subsequence match with word-boundary bonus.
	pos = make([]int, 0, len(qr))
	qi := 0
	lastMatch := -1
	penalty := 0
	for ti, ch := range tr {
		if qi < len(qr) && ch == qr[qi] {
			pos = append(pos, ti)
			if lastMatch >= 0 {
				penalty += ti - lastMatch - 1
			} else if ti > 0 && unicode.IsLetter(tr[ti-1]) {
				penalty += ti
			}
			lastMatch = ti
			qi++
		}
	}
	if qi != len(qr) {
		return 0, 0, nil, false
	}
	return 2, penalty*4 + len(tr), pos, true
}

// syncChatPopup updates popup suggestions whenever chatInput text changes.
func (m *Model) syncChatPopup() {
	val := m.chatInput.Value()
	if val == "" {
		m.popup.reset()
		return
	}
	if m.popup.dismissedFor != "" {
		if val == m.popup.dismissedFor {
			m.popup.open = false
			return
		}
		m.popup.dismissedFor = ""
	}

	// 1. Slash command / argument completion (when input begins with '/').
	if strings.HasPrefix(val, "/") {
		spaceIdx := strings.IndexAny(val, " \t")
		if spaceIdx < 0 {
			query := strings.TrimPrefix(val, "/")
			cmds := m.commandRegistry()
			raw := make([]popupItem, 0, len(cmds))
			for _, c := range cmds {
				if c.Name == "think" && !m.thinkToggleSupported() {
					// Still list /think for display show/hide, just narrow hint.
				}
				disabled := false
				reason := ""
				if c.Enabled != nil {
					if ok, why := c.Enabled(m); !ok {
						disabled = true
						reason = why
					}
				}
				raw = append(raw, popupItem{
					ID:             c.Name,
					Label:          "/" + c.Name,
					Description:    c.Description,
					ArgsHint:       c.ArgsHint,
					Disabled:       disabled,
					DisabledReason: reason,
				})
			}
			// Match query against command names cleanly (shift match positions +1 for leading '/').
			ranked := rankPopupItems(query, shiftSlashMatchQuery(raw, query))
			m.setPopupItems(popupSlashCmd, "Commands", "", 0, len(val), ranked)
			return
		}

		cmdName := strings.ToLower(strings.TrimPrefix(val[:spaceIdx], "/"))
		argPrefix := strings.TrimLeft(val[spaceIdx+1:], " \t")
		if cmd, ok := m.lookupCommand(cmdName); ok && cmd.ArgCandidates != nil {
			cands := cmd.ArgCandidates(m, argPrefix)
			ranked := rankPopupItems(argPrefix, cands)
			m.setPopupItems(popupSlashArg, "/"+cmd.Name+" "+cmd.ArgsHint, cmd.Name, spaceIdx+1, len(val), ranked)
			return
		}
		m.popup.open = false
		return
	}

	// 2. @file mention anywhere in the message.
	atPos := lastAtTokenStart(val)
	if atPos >= 0 {
		prefix := val[atPos+1:]
		if !strings.ContainsAny(prefix, " \t\n") {
			files := listWorkspaceFiles(m.workspaceDir(), 200)
			raw := make([]popupItem, 0, len(files))
			for _, f := range files {
				raw = append(raw, popupItem{
					ID:          f,
					Label:       f,
					Description: "file",
				})
			}
			ranked := rankPopupItems(prefix, raw)
			m.setPopupItems(popupAtFile, "@ Workspace files", "", atPos, len(val), ranked)
			return
		}
	}

	m.popup.open = false
}

func shiftSlashMatchQuery(raw []popupItem, query string) []popupItem {
	if query == "" {
		return raw
	}
	// rankPopupItems matches query against Label ("/help"): stripping leading '/' for matching
	// and shifting indices keeps '/' unhighlighted while highlighting exact cmd characters!
	out := make([]popupItem, len(raw))
	for i, it := range raw {
		cp := it
		cp.Label = strings.TrimPrefix(it.Label, "/")
		out[i] = cp
	}
	ranked := rankPopupItems(query, out)
	for i := range ranked {
		ranked[i].Label = "/" + ranked[i].Label
		for j := range ranked[i].MatchPositions {
			ranked[i].MatchPositions[j]++
		}
	}
	return ranked
}

func (p *suggestionPopup) set(kind popupKind, title, cmdName string, start, end int, items []popupItem) {
	if len(items) == 0 {
		p.open = false
		p.items = nil
		return
	}
	prevID := ""
	if p.open && p.idx >= 0 && p.idx < len(p.items) {
		prevID = p.items[p.idx].ID
	}
	p.open = true
	p.kind = kind
	p.title = title
	p.cmdName = cmdName
	p.tokenStart = start
	p.tokenEnd = end
	p.items = items
	p.idx = 0
	if prevID != "" {
		for i, it := range items {
			if it.ID == prevID {
				p.idx = i
				break
			}
		}
	}
	p.ensureVisible()
}

func (m *Model) setPopupItems(kind popupKind, title, cmdName string, start, end int, items []popupItem) {
	if kind == popupSlashCmd && strings.TrimSpace(strings.TrimPrefix(m.chatInput.Value(), "/")) != "" {
		// rankPopupItems was already called on shifted labels cleanly inside shiftSlashMatchQuery
		m.popup.set(kind, title, cmdName, start, end, items)
		return
	}
	m.popup.set(kind, title, cmdName, start, end, items)
}

func lastAtTokenStart(val string) int {
	for i := len(val) - 1; i >= 0; i-- {
		if val[i] == '@' {
			if i == 0 || val[i-1] == ' ' || val[i-1] == '\t' || val[i-1] == '\n' {
				return i
			}
			return -1
		}
		if val[i] == ' ' || val[i] == '\t' || val[i] == '\n' {
			return -1
		}
	}
	return -1
}

// handlePopupKey handles navigation/completion/run/esc while suggestion popup is open.
func (m *Model) handlePopupKey(msg tea.KeyMsg) (handled bool, model tea.Model, cmd tea.Cmd) {
	if !m.popup.open {
		return false, m, nil
	}
	switch msg.String() {
	case "up":
		m.popup.move(-1)
		return true, m, nil
	case "down":
		m.popup.move(1)
		return true, m, nil
	case "esc":
		m.popup.closeKeepDismissed(m.chatInput.Value())
		return true, m, nil
	case "tab":
		it, ok := m.popup.selected()
		if !ok {
			return true, m, nil
		}
		if it.Disabled {
			m.toastf("/%s disabled: %s", it.ID, it.DisabledReason)
			return true, m, nil
		}
		m.applyPopupCompletion(it, false)
		return true, m, nil
	case "enter":
		it, ok := m.popup.selected()
		if !ok {
			return false, m, nil
		}
		switch m.popup.kind {
		case popupAtFile:
			m.applyPopupCompletion(it, false)
			return true, m, nil
		case popupSlashArg:
			if it.Disabled {
				m.toastf("%s disabled: %s", it.Label, it.DisabledReason)
				return true, m, nil
			}
			line := "/" + m.popup.cmdName + " " + it.ID
			m.chatInput.SetValue("")
			m.popup.reset()
			mdl, c := m.handleChatSlash(line)
			return true, mdl, c
		case popupSlashCmd:
			if it.Disabled {
				m.toastf("/%s disabled: %s", it.ID, it.DisabledReason)
				return true, m, nil
			}
			// If user typed a partial command for one that requires an arg (e.g. /mo -> /model),
			// expand into argument candidates when it isn't an exact match yet; otherwise execute cleanly!
			c, hasCmd := m.lookupCommand(it.ID)
			exactTyped := strings.EqualFold(strings.TrimSpace(m.chatInput.Value()), "/"+it.ID)
			if hasCmd && c.RequiresArg && !exactTyped {
				m.applyPopupCompletion(it, false)
				return true, m, nil
			}
			m.chatInput.SetValue("")
			m.popup.reset()
			mdl, runCmd := m.handleChatSlash("/" + it.ID)
			return true, mdl, runCmd
		}
	}
	return false, m, nil
}

func (m *Model) applyPopupCompletion(it popupItem, trimSpace bool) {
	switch m.popup.kind {
	case popupSlashCmd:
		suffix := ""
		if c, ok := m.lookupCommand(it.ID); ok && c.ArgCandidates != nil {
			suffix = " "
		}
		if trimSpace {
			suffix = ""
		}
		m.chatInput.SetValue("/" + it.ID + suffix)
		m.chatInput.CursorEnd()
		m.popup.dismissedFor = ""
		m.syncChatPopup()
	case popupSlashArg:
		m.chatInput.SetValue("/" + m.popup.cmdName + " " + it.ID)
		m.chatInput.CursorEnd()
		m.popup.open = false
		m.popup.dismissedFor = m.chatInput.Value()
	case popupAtFile:
		val := m.chatInput.Value()
		before := val[:m.popup.tokenStart]
		m.chatInput.SetValue(before + "@" + it.ID + " ")
		m.chatInput.CursorEnd()
		m.popup.open = false
	}
}

// renderPopupBox renders up to 8 rows directly inside innerW x maxH without changing panel height.
func renderPopupBox(sty styleSet, p suggestionPopup, maxW, maxH int) []string {
	if !p.open || len(p.items) == 0 || maxH < 4 || maxW < 24 {
		return nil
	}

	tl, tr, bl, br, hl, vl := "╭", "╮", "╰", "╯", "─", "│"
	upMark, dnMark := "↑", "↓"
	if sty.ascii {
		tl, tr, bl, br, hl, vl = "+", "+", "+", "+", "-", "|"
		upMark, dnMark = "^", "v"
	}

	visRows := popupMaxVisible
	if len(p.items) < visRows {
		visRows = len(p.items)
	}
	if visRows > maxH-2 {
		visRows = maxH - 2
	}
	if visRows < 1 {
		return nil
	}

	scroll := p.scroll
	if p.idx < scroll {
		scroll = p.idx
	}
	if p.idx >= scroll+visRows {
		scroll = p.idx - visRows + 1
	}
	if scroll+visRows > len(p.items) {
		scroll = len(p.items) - visRows
	}
	if scroll < 0 {
		scroll = 0
	}

	boxW := maxW
	innerW := boxW - 2

	posIndicator := fmt.Sprintf("%d/%d", p.idx+1, len(p.items))
	if scroll > 0 {
		posIndicator = upMark + " " + posIndicator
	}
	if scroll+visRows < len(p.items) {
		posIndicator = posIndicator + " " + dnMark
	}

	titleLeft := hl + " " + p.title + " "
	titleRight := " " + posIndicator + " " + hl
	fill := innerW - displayWidth(titleLeft) - displayWidth(titleRight)
	var topLine string
	if fill >= 1 {
		topLine = sty.sAccent.Render(tl+titleLeft) + sty.sMuted.Render(strings.Repeat(hl, fill)+titleRight+tr)
	} else {
		topLine = sty.sAccent.Render(tl + truncate(titleLeft, innerW) + tr)
	}

	out := make([]string, 0, visRows+2)
	out = append(out, topLine)

	labelColW := 18
	if innerW < 48 {
		labelColW = 14
	} else if innerW >= 96 {
		labelColW = 24
	}
	argColW := 16
	if innerW < 60 {
		argColW = 0
	}

	end := scroll + visRows
	if end > len(p.items) {
		end = len(p.items)
	}
	for i := scroll; i < end; i++ {
		it := p.items[i]
		selected := i == p.idx

		marker := "  "
		if selected {
			marker = "▸ "
			if sty.ascii {
				marker = "> "
			}
		}

		rawLbl := truncate(it.Label, labelColW-1)
		lblStyled := renderHighlightedLabel(sty, rawLbl, it.MatchPositions, selected, it.Disabled)
		if pad := labelColW - displayWidth(rawLbl); pad > 0 {
			lblStyled += strings.Repeat(" ", pad)
		}

		argPart := ""
		if argColW > 0 {
			a := truncate(it.ArgsHint, argColW-1)
			argPart = sty.sMuted.Render(pad(a, argColW))
		}

		descW := innerW - 2 - displayWidth(marker) - labelColW - argColW
		descTxt := it.Description
		if it.Disabled && it.DisabledReason != "" {
			descTxt = "(" + it.DisabledReason + ") " + descTxt
		}
		if descW < 4 {
			descW = 4
		}
		descTrunc := truncate(descTxt, descW)
		descStyled := sty.sMuted.Render(descTrunc)
		if !it.Disabled && selected {
			descStyled = sty.sText.Render(descTrunc)
		}

		rowContent := marker + lblStyled + argPart + descStyled
		rowContent = widthClampANSI(" "+rowContent+" ", innerW)
		out = append(out, sty.sMuted.Render(vl)+rowContent+sty.sMuted.Render(vl))
	}

	footerHint := " [tab] complete · [enter] run · [esc] close "
	footerFill := innerW - displayWidth(footerHint)
	var bottomLine string
	if footerFill >= 1 {
		bottomLine = sty.sMuted.Render(bl + strings.Repeat(hl, footerFill) + footerHint + br)
	} else {
		bottomLine = sty.sMuted.Render(bl + strings.Repeat(hl, innerW) + br)
	}
	out = append(out, bottomLine)
	return out
}

func renderHighlightedLabel(sty styleSet, label string, positions []int, selected, disabled bool) string {
	posSet := make(map[int]bool, len(positions))
	for _, p := range positions {
		posSet[p] = true
	}
	baseStyle := sty.sText
	if selected {
		baseStyle = sty.sAccentBold
	}
	if disabled {
		baseStyle = sty.sMuted
	}
	matchStyle := sty.sAccentBold.Underline(true)
	if disabled {
		matchStyle = sty.sMuted.Underline(true)
	}

	var b strings.Builder
	for i, r := range []rune(label) {
		ch := string(r)
		if posSet[i] {
			b.WriteString(matchStyle.Render(ch))
		} else {
			b.WriteString(baseStyle.Render(ch))
		}
	}
	return b.String()
}
