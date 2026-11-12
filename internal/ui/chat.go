package ui

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/session"
)

type runTab int

const (
	runTabLog runTab = iota
	runTabChat
)

type chatTurnCache struct {
	width      int
	hideThink  bool
	thinkExp   int // -1 auto, 0 collapsed, 1 expanded
	thinkDone  bool
	lineNums   bool
	streaming  bool
	spinFrame  string
	reasonLen  int
	contentLen int
	tokens     int
	thinkToks  int
	genTokSec  float64
	aborted    bool
	errStr     string
	toolVer    int
	themeName  string
	ascii      bool
	rendered   []string
}

type chatTurn struct {
	Role                string // "user", "assistant", "info", "compaction"
	Reasoning           string
	Content             string
	Attachments         []session.AttachmentRecord
	GenTokSec           float64
	Tokens              int
	ThinkStart          time.Time
	ThinkEnd            time.Time
	ThinkTokens         int
	ThinkDone           bool
	ThinkExpanded       *bool
	ToolCalls           []toolCard
	ToolVer             int
	APIMessages         []runner.APIMessage
	CompactedOut        bool
	IsCompactionSummary bool
	CompactionDetail    string
	Aborted             bool
	Err                 string
	thinkParser         *runner.ThinkStreamParser
	mdCache             markdownRenderCache
	cache               chatTurnCache
}

type chatStreamEvt struct {
	gen   uint64
	delta runner.ChatDelta
	ch    <-chan runner.ChatDelta
	errCh <-chan error
}

type chatDoneMsg struct {
	gen uint64
	err error
}

func (m *Model) stopChatStream() {
	m.permModal = nil
	m.askModal = nil
	if m.chatCancel != nil {
		m.chatCancel()
		m.chatCancel = nil
	}
	if m.chatStreaming {
		m.chatStreaming = false
		m.chatGen++
		if n := len(m.chatMsgs); n > 0 && m.chatMsgs[n-1].Role == "assistant" {
			t := &m.chatMsgs[n-1]
			t.Aborted = true
			t.markThinkDone()
			for i := range t.ToolCalls {
				if t.ToolCalls[i].Status == "running" {
					t.ToolCalls[i].Status = "interrupted"
				}
			}
			t.ToolVer++
		}
		m.refreshChatView(true)
	}
}

func (m *Model) thinkToggleSupported() bool {
	return !m.modelCaps.Loaded || m.modelCaps.SupportsEnableThinking
}

func (m *Model) toggleThinkingDisplay() {
	anyExpanded := false
	hasThinking := false
	for i := range m.chatMsgs {
		t := &m.chatMsgs[i]
		if t.Role == "assistant" && t.Reasoning != "" {
			hasThinking = true
			if t.isThinkExpanded(m.settings.HideThinking, false) {
				anyExpanded = true
			}
		}
	}
	if hasThinking {
		target := !anyExpanded
		for i := range m.chatMsgs {
			t := &m.chatMsgs[i]
			if t.Role == "assistant" && t.Reasoning != "" {
				b := target
				t.ThinkExpanded = &b
			}
		}
		m.settings.HideThinking = !target
		return
	}
	m.settings.HideThinking = !m.settings.HideThinking
}

func (t *chatTurn) markThinkDone() {
	if t.ThinkDone {
		return
	}
	t.ThinkDone = true
	if !t.ThinkStart.IsZero() && t.ThinkEnd.IsZero() {
		t.ThinkEnd = time.Now()
	}
}

func (t *chatTurn) isThinkExpanded(hideThinking, streaming bool) bool {
	if t.ThinkExpanded != nil {
		return *t.ThinkExpanded
	}
	if hideThinking {
		return false
	}
	// Auto-collapse once thinking finishes; during live streaming use 3-line preview.
	return false
}

func (t *chatTurn) thinkDuration() time.Duration {
	if t.ThinkStart.IsZero() {
		return 0
	}
	end := t.ThinkEnd
	if end.IsZero() {
		end = time.Now()
	}
	d := end.Sub(t.ThinkStart)
	if d < 0 {
		return 0
	}
	return d
}

func fetchPropsCmd(port int, baseURL string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		var (
			p   runner.ModelProps
			err error
		)
		if baseURL != "" {
			p, err = runner.FetchPropsURL(ctx, baseURL)
		} else {
			p, err = runner.FetchProps(ctx, port)
		}
		return propsMsg{props: p, err: err}
	}
}

func (m *Model) copyLatestCodeOrReply() tea.Cmd {
	blocks := m.allCodeBlocks()
	if len(blocks) > 0 {
		b := blocks[len(blocks)-1]
		lang := b.Lang
		if lang == "" {
			lang = "code"
		}
		return func() tea.Msg {
			if err := copyToClipboard(b.Code); err != nil {
				return toastMsg("copy failed: " + err.Error())
			}
			return toastMsg("copied " + lang + " code block")
		}
	}
	for i := len(m.chatMsgs) - 1; i >= 0; i-- {
		if m.chatMsgs[i].Role == "assistant" && strings.TrimSpace(m.chatMsgs[i].Content) != "" {
			return copyCmd(m.chatMsgs[i].Content)
		}
	}
	m.toastf("nothing to copy yet")
	return nil
}

func (m *Model) allCodeBlocks() []codeBlock {
	var all []codeBlock
	for _, t := range m.chatMsgs {
		if t.Role == "assistant" && t.Content != "" {
			all = append(all, extractCodeBlocks(t.Content)...)
		}
	}
	return all
}

func (m *Model) resetChat() {
	m.stopChatStream()
	m.chatMsgs = nil
	m.chatCtxUsed = 0
	m.sessionTainted = false
	m.sessionID = fmt.Sprintf("sess-%d", time.Now().UnixNano())
	m.sessionDoc = nil
	m.attachmentTokens = 0
	m.cachedPrefixTokens = 0
	m.checkpoints = nil
	m.popup.reset()
	m.refreshChatView(true)
}

func (m *Model) effectivePort() int {
	if m.activePort > 0 && m.proc != nil && !m.proc.Exited() {
		return m.activePort
	}
	return m.settings.Port
}

func (m *Model) effectiveCtxTotal() int {
	if m.telemetry.NCtx > 0 {
		return m.telemetry.NCtx
	}
	if m.metrics.NCtx > 0 {
		return m.metrics.NCtx
	}
	if m.settings.Context > 0 {
		return m.settings.Context
	}
	return 8192
}

func (m *Model) effectiveCtxUsed() int {
	total := m.effectiveCtxTotal()
	fromKV := 0
	if m.telemetry.KVCacheRatio > 0 && total > 0 {
		fromKV = int(math.Round(m.telemetry.KVCacheRatio * float64(total)))
	}
	if m.chatCtxUsed > fromKV {
		return m.chatCtxUsed
	}
	return fromKV
}

func (m *Model) effectiveGenTokSec() float64 {
	if m.chatStreaming && m.metrics.GenTokPerSec > 0 {
		return m.metrics.GenTokPerSec
	}
	if m.telemetry.GenTokPerSec > 0 {
		return m.telemetry.GenTokPerSec
	}
	return m.metrics.GenTokPerSec
}

func (m *Model) effectivePromptTokSec() float64 {
	if m.chatStreaming && m.metrics.PromptTokSec > 0 {
		return m.metrics.PromptTokSec
	}
	if m.telemetry.PromptTokSec > 0 {
		return m.telemetry.PromptTokSec
	}
	return m.metrics.PromptTokSec
}

func (m *Model) submitChatInput() (tea.Model, tea.Cmd) {
	raw := strings.TrimSpace(m.chatInput.Value())
	if raw == "" {
		return m, nil
	}
	m.chatInput.SetValue("")
	m.popup.reset()

	if strings.HasPrefix(raw, "/") {
		return m.handleChatSlash(raw)
	}

	if m.chatStreaming {
		m.chatInput.SetValue(raw)
		m.toastf("generation running — press esc to stop current reply")
		return m, nil
	}
	if m.status != "READY" && m.chatBaseURL == "" {
		m.chatInput.SetValue(raw)
		m.toastf("model is %s — wait until READY to send", m.status)
		return m, nil
	}

	return m.submitPromptWithAttachments(raw)
}

func (m *Model) appendInfoTurn(msg string) {
	m.chatMsgs = append(m.chatMsgs, chatTurn{Role: "info", Content: msg})
	m.refreshChatView(true)
	m.chatView.GotoBottom()
}

func (m *Model) cycleReasoningEffortLive() {
	if m.settings.Reasoning == "off" {
		m.settings.Reasoning = "auto"
		m.settings.ReasoningEffort = "minimal"
	} else {
		m.settings.ReasoningEffort = cycle(flags.ReasoningEffortOptions(), m.settings.ReasoningEffort, 1)
	}
	m.toastf("reasoning effort: %s", m.reasoningBadgeLabel())
}

func (m *Model) reasoningBadgeLabel() string {
	if m.settings.Reasoning == "off" {
		return "off"
	}
	if m.settings.ReasoningEffort == "" || m.settings.ReasoningEffort == "default" {
		return "default"
	}
	return m.settings.ReasoningEffort
}

func (m *Model) startChatCompletion(prompt string) (tea.Model, tea.Cmd) {
	return m.startAgentCompletion(prompt)
}

func readChatEvtCmd(gen uint64, ch <-chan runner.ChatDelta, errCh <-chan error) tea.Cmd {
	return func() tea.Msg {
		d, ok := <-ch
		if !ok {
			var err error
			if errCh != nil {
				err = <-errCh
			}
			return chatDoneMsg{gen: gen, err: err}
		}
		return chatStreamEvt{gen: gen, delta: d, ch: ch, errCh: errCh}
	}
}

func (m *Model) handleChatDelta(d runner.ChatDelta) {
	if len(m.chatMsgs) == 0 {
		return
	}
	t := &m.chatMsgs[len(m.chatMsgs)-1]
	if t.Role != "assistant" {
		return
	}
	if t.thinkParser == nil {
		t.thinkParser = &runner.ThinkStreamParser{}
	}

	reasonPart := d.Reasoning
	contentPart := d.Content
	if contentPart != "" {
		inlineReason, cleanContent := t.thinkParser.Feed(contentPart)
		reasonPart += inlineReason
		contentPart = cleanContent
	}

	if reasonPart != "" {
		if t.ThinkStart.IsZero() {
			t.ThinkStart = time.Now()
		}
		t.Reasoning += reasonPart
		t.ThinkTokens += estimateThinkTokens(reasonPart)
		if !t.ThinkDone {
			m.spinner, _ = m.spinner.Update(m.spinner.Tick())
		}
	}
	if contentPart != "" {
		if t.Reasoning != "" {
			t.markThinkDone()
		}
		t.Content += contentPart
	} else if t.thinkParser.Closed() && t.Reasoning != "" {
		t.markThinkDone()
	}
	if d.PromptTokens > 0 || d.CompletionTokens > 0 {
		t.Tokens = d.CompletionTokens
		m.chatCtxUsed = d.PromptTokens + d.CompletionTokens
	}
	if d.GenTokPerSec > 0 {
		t.GenTokSec = d.GenTokPerSec
		m.metrics.GenTokPerSec = d.GenTokPerSec
	}
	if d.PromptTokSec > 0 {
		m.metrics.PromptTokSec = d.PromptTokSec
	}
	m.refreshChatView(true)
}

func estimateThinkTokens(s string) int {
	if s == "" {
		return 0
	}
	// Each streamed chunk averages ~1 token; longer coalesced chunks scale at ~4 chars/token.
	n := (len(s) + 3) / 4
	if n < 1 {
		return 1
	}
	return n
}

func (m *Model) finishChatStream(err error) {
	m.chatStreaming = false
	if m.chatCancel != nil {
		m.chatCancel()
		m.chatCancel = nil
	}
	if len(m.chatMsgs) > 0 {
		t := &m.chatMsgs[len(m.chatMsgs)-1]
		if t.Role == "assistant" {
			if t.thinkParser != nil {
				r, c := t.thinkParser.Flush()
				if r != "" {
					if t.ThinkStart.IsZero() {
						t.ThinkStart = time.Now()
					}
					t.Reasoning += r
					t.ThinkTokens += estimateThinkTokens(r)
				}
				if c != "" {
					t.Content += c
				}
			}
			t.markThinkDone()
			if err != nil && !t.Aborted && err != context.Canceled {
				t.Err = err.Error()
			}
		}
	}
	m.refreshChatView(true)
}

func (m *Model) resizeRunViews() {
	m.resizeLogView()
	w := m.width - 4
	if w < 10 {
		w = 10
	}
	// Run screen: title (1) + metrics (1) + help (1) + panel border/title (3) +
	// chat status/dividers/input (4) = 10 lines chrome.
	chatH := m.height - 10
	if chatH < 3 {
		chatH = 3
	}
	m.chatView.Width = w
	m.chatView.Height = chatH
	inputW := w - 4
	if inputW < 10 {
		inputW = 10
	}
	m.chatInput.Width = inputW
	m.refreshChatView(false)
}

func (m *Model) refreshChatView(scrollBottom bool) {
	innerW := m.chatView.Width
	if innerW < 20 {
		innerW = 60
	}

	if len(m.chatMsgs) == 0 {
		name := "model"
		if m.meta != nil {
			name = m.meta.DisplayName()
		}
		lines := []string{
			"",
			m.sty.accent().Render("  TUI Chat ready — ") + m.sty.text().Render(truncate(name, innerW-20)),
			m.sty.muted().Render("  Type a prompt below and press Enter, or /help for commands."),
		}
		if sp := strings.TrimSpace(m.settings.SystemPrompt); sp != "" {
			lines = append(lines, m.sty.muted().Render("  System prompt: "+truncate(sp, innerW-18)))
		}
		if m.status != "READY" {
			lines = append(lines, m.sty.warning().Render("  Waiting for llama-server to finish loading ("+m.status+")…"))
		}
		m.chatView.SetContent(strings.Join(lines, "\n"))
		return
	}

	out := make([]string, 0, len(m.chatMsgs)*4)
	for i := range m.chatMsgs {
		isLast := i == len(m.chatMsgs)-1
		lines := m.renderChatTurn(&m.chatMsgs[i], innerW, isLast)
		out = append(out, lines...)
	}

	wasAtBottom := len(m.chatMsgs) <= 1 || m.chatView.AtBottom()
	m.chatView.SetContent(strings.Join(out, "\n"))
	if scrollBottom && wasAtBottom {
		m.chatView.GotoBottom()
	}
}

func (m *Model) renderChatTurn(t *chatTurn, innerW int, isLast bool) []string {
	streaming := isLast && m.chatStreaming
	thinkExp := -1
	if t.ThinkExpanded != nil {
		if *t.ThinkExpanded {
			thinkExp = 1
		} else {
			thinkExp = 0
		}
	}
	spinFrame := ""
	if streaming && t.Reasoning != "" && !t.ThinkDone {
		spinFrame = m.spinner.View()
	}

	c := &t.cache
	if c.rendered != nil &&
		c.width == innerW &&
		c.hideThink == m.settings.HideThinking &&
		c.thinkExp == thinkExp &&
		c.thinkDone == t.ThinkDone &&
		c.lineNums == m.codeLineNums &&
		c.streaming == streaming &&
		c.spinFrame == spinFrame &&
		c.reasonLen == len(t.Reasoning) &&
		c.contentLen == len(t.Content) &&
		c.tokens == t.Tokens &&
		c.thinkToks == t.ThinkTokens &&
		c.genTokSec == t.GenTokSec &&
		c.aborted == t.Aborted &&
		c.errStr == t.Err &&
		c.toolVer == t.ToolVer &&
		c.themeName == m.sty.t.Name &&
		c.ascii == m.sty.ascii {
		return c.rendered
	}

	var out []string
	switch t.Role {
	case "user":
		out = append(out, m.sty.sAccent2Bold.Render("You"))
		for _, l := range wrapText(t.Content, innerW-2) {
			out = append(out, "  "+m.sty.sText.Render(l))
		}
		for _, a := range t.Attachments {
			lr := ""
			if a.LineRange != "" {
				lr = ":" + a.LineRange
			}
			out = append(out, "  "+m.sty.accent2().Render(fmt.Sprintf("@%s%s (%d tok)", a.Path, lr, a.Tokens)))
		}
		out = append(out, "")
	case "compaction":
		exp := false
		if t.ThinkExpanded != nil {
			exp = *t.ThinkExpanded
		}
		out = append(out, renderCollapsible(m.sty, innerW, "", collapsibleBlock{
			Summary:   t.Content,
			Body:      t.CompactionDetail,
			RightHint: "[ctrl+o]",
			Expanded:  exp,
		})...)
		out = append(out, "")
	case "info":
		for _, l := range wrapText(t.Content, innerW-2) {
			out = append(out, m.sty.sMuted.Render("• "+l))
		}
		out = append(out, "")
	case "assistant":
		hdr := m.sty.sAccentBold.Render("Assistant")
		var meta []string
		if t.GenTokSec > 0 {
			meta = append(meta, fmt.Sprintf("%.1f t/s", t.GenTokSec))
		}
		if t.Tokens > 0 {
			meta = append(meta, fmt.Sprintf("%d tok", t.Tokens))
		}
		if t.Aborted {
			meta = append(meta, "stopped")
		}
		if len(meta) > 0 {
			hdr += "  " + m.sty.sMuted.Render(strings.Join(meta, " · "))
		}
		out = append(out, hdr)

		if t.Reasoning != "" {
			dur := t.thinkDuration()
			thinkTok := t.ThinkTokens
			if thinkTok == 0 {
				thinkTok = estimateThinkTokens(t.Reasoning)
			}
			activeThink := streaming && !t.ThinkDone
			expanded := t.isThinkExpanded(m.settings.HideThinking, streaming)
			summary := fmt.Sprintf("Thought for %.1fs · %d tok", dur.Seconds(), thinkTok)
			preview := 0
			if activeThink {
				summary = fmt.Sprintf("Thinking %.1fs · %d tok", dur.Seconds(), thinkTok)
				if !m.settings.HideThinking {
					preview = 3
				}
			}
			out = append(out, renderCollapsible(m.sty, innerW, spinFrame, collapsibleBlock{
				Summary:      summary,
				Body:         t.Reasoning,
				RightHint:    "[ctrl+t]",
				Expanded:     expanded,
				Streaming:    activeThink,
				PreviewLines: preview,
			})...)
		}

		for _, tc := range t.ToolCalls {
			out = append(out, renderToolCardBlock(m, tc, innerW, streaming, m.spinner.View())...)
		}

		body := t.Content
		if body != "" || (streaming && t.Reasoning == "" && len(t.ToolCalls) == 0) {
			out = append(out, renderMarkdownLines(m.sty, innerW, body, m.codeLineNums, streaming, &t.mdCache)...)
		}
		if t.Err != "" {
			for _, l := range wrapText("error: "+t.Err, innerW-2) {
				out = append(out, "  "+m.sty.sDanger.Render(l))
			}
		}
		out = append(out, "")
	}

	*c = chatTurnCache{
		width:      innerW,
		hideThink:  m.settings.HideThinking,
		thinkExp:   thinkExp,
		thinkDone:  t.ThinkDone,
		lineNums:   m.codeLineNums,
		streaming:  streaming,
		spinFrame:  spinFrame,
		reasonLen:  len(t.Reasoning),
		contentLen: len(t.Content),
		tokens:     t.Tokens,
		thinkToks:  t.ThinkTokens,
		genTokSec:  t.GenTokSec,
		aborted:    t.Aborted,
		errStr:     t.Err,
		toolVer:    t.ToolVer,
		themeName:  m.sty.t.Name,
		ascii:      m.sty.ascii,
		rendered:   out,
	}
	return out
}

const chatCoalesceInterval = 16 * time.Millisecond

// coalesceChatDeltas caps SSE UI redraws at ~60 fps while passing first tokens
// and final Done markers immediately with zero extra latency.
func coalesceChatDeltas(ctx context.Context, raw <-chan runner.ChatDelta, out chan<- runner.ChatDelta) {
	defer close(out)
	var pending runner.ChatDelta
	hasPending := false
	var lastFlush time.Time
	var timer *time.Timer
	var timerC <-chan time.Time

	stopTimer := func() {
		if timer != nil {
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timerC = nil
		}
	}
	defer stopTimer()

	flush := func() bool {
		if !hasPending {
			return true
		}
		d := pending
		pending = runner.ChatDelta{}
		hasPending = false
		lastFlush = time.Now()
		select {
		case out <- d:
			return true
		case <-ctx.Done():
			return false
		}
	}

	for {
		select {
		case d, ok := <-raw:
			if !ok {
				stopTimer()
				flush()
				return
			}
			if d.Done {
				stopTimer()
				if !flush() {
					return
				}
				select {
				case out <- d:
				case <-ctx.Done():
				}
				return
			}
			pending.Reasoning += d.Reasoning
			pending.Content += d.Content
			if d.FinishReason != "" {
				pending.FinishReason = d.FinishReason
			}
			if d.PromptTokens > 0 {
				pending.PromptTokens = d.PromptTokens
			}
			if d.CompletionTokens > 0 {
				pending.CompletionTokens = d.CompletionTokens
			}
			if d.PromptTokSec > 0 {
				pending.PromptTokSec = d.PromptTokSec
			}
			if d.GenTokPerSec > 0 {
				pending.GenTokPerSec = d.GenTokPerSec
			}
			hasPending = true
			if lastFlush.IsZero() || time.Since(lastFlush) >= chatCoalesceInterval {
				stopTimer()
				if !flush() {
					return
				}
			} else if timerC == nil {
				wait := chatCoalesceInterval - time.Since(lastFlush)
				if timer == nil {
					timer = time.NewTimer(wait)
				} else {
					timer.Reset(wait)
				}
				timerC = timer.C
			}
		case <-timerC:
			timerC = nil
			if !flush() {
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

func wrapText(text string, width int) []string {
	if width < 8 {
		width = 8
	}
	rawLines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	var out []string
	for _, raw := range rawLines {
		if raw == "" {
			out = append(out, "")
			continue
		}
		var cur strings.Builder
		curW := 0
		for _, r := range raw {
			rw := runeWidth(r)
			if curW+rw > width && curW > 0 {
				s := cur.String()
				if idx := strings.LastIndexAny(s, " \t"); idx > len(s)/2 {
					out = append(out, strings.TrimRight(s[:idx], " \t"))
					rem := strings.TrimLeft(s[idx+1:], " \t")
					cur.Reset()
					cur.WriteString(rem)
					curW = displayWidth(rem)
				} else {
					out = append(out, s)
					cur.Reset()
					curW = 0
				}
			}
			cur.WriteRune(r)
			curW += rw
		}
		out = append(out, cur.String())
	}
	return out
}
