package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/runner"
)

type toolCard struct {
	CallID      string
	Name        string
	Summary     string
	ArgsPretty  string
	Output      string
	Diff        string
	DiffAdded   int
	DiffRemoved int
	Status      string // "running", "ok", "error", "denied", "interrupted"
	Duration    time.Duration
	Expanded    *bool
}

type askUserReply struct {
	Answer string
	Err    error
}

type permModalState struct {
	gen          uint64
	prompt       agent.PermissionPrompt
	replyCh      chan<- agent.PermissionResponse
	ch           <-chan agentLoopMsg
	doneCh       <-chan agent.LoopResult
	enteringNote bool
	noteInput    textinput.Model
	choice       int // 0 allow once, 1 always allow, 2 deny
}

const (
	permChoiceAllow = iota
	permChoiceAlways
	permChoiceDeny
)

type askModalState struct {
	gen       uint64
	question  string
	options   []string
	cursor    int
	scroll    int
	inputMode bool
	replyCh   chan<- askUserReply
	ch        <-chan agentLoopMsg
	doneCh    <-chan agent.LoopResult
	input     textinput.Model
}

type toolsModalState struct {
	cursor int
}

type permsModalState struct {
	scroll int
}

type agentLoopKind int

const (
	agentEvtDelta agentLoopKind = iota
	agentEvtToolStart
	agentEvtToolEnd
	agentEvtTurnMsgs
	agentEvtPermReq
	agentEvtAskReq
	agentEvtSkillStageReq
	agentEvtHeartbeat
	agentEvtDone
)

type agentLoopMsg struct {
	gen        uint64
	kind       agentLoopKind
	ev         agent.LoopEvent
	perm       agent.PermissionPrompt
	permRep    chan<- agent.PermissionResponse
	question   string
	options    []string
	askRep     chan<- askUserReply
	skillStage agent.SkillStagePrompt
	skillRep   chan<- agent.SkillStageResponse
	res        agent.LoopResult

	ch     <-chan agentLoopMsg
	doneCh <-chan agent.LoopResult
}

type agentToolTokensMsg struct {
	tokens int
	live   bool
}

func (m *Model) ensureAgentReady() {
	if m.agentWorkDir == "" {
		wd := strings.TrimSpace(m.root)
		if wd == "" {
			if gwd, err := os.Getwd(); err == nil && gwd != "" {
				wd = gwd
			} else {
				wd = "."
			}
		}
		if abs, err := filepath.Abs(wd); err == nil {
			wd = abs
		}
		m.agentWorkDir = wd
	}
	if m.sessionID == "" {
		m.sessionID = fmt.Sprintf("sess-%d", time.Now().Unix())
	}
	if m.toolRegistry == nil {
		m.toolRegistry = agent.NewDefaultRegistry()
		m.reloadWebTools()
	}
	if m.permEngine == nil {
		m.permEngine = agent.NewPermissionEngine(agent.ModeReadAuto)
	}
	if m.checkpoints == nil {
		m.checkpoints = agent.NewCheckpointStore(m.agentWorkDir, m.sessionID)
	}
	if m.skillsCatalog.SkillsDir == "" {
		m.reloadSkills()
	}
	m.ensureMCPReady()
}

func (m *Model) reloadWebTools() {
	if m.toolRegistry == nil {
		return
	}
	var wc config.WebConfig
	if m.cfg != nil {
		wc = m.cfg.Web
	}
	agent.RegisterWebTools(m.toolRegistry, agent.WebToolsConfig{
		Web:        wc,
		ClientOpts: m.webClientOpts,
	})
}

func (m *Model) workspaceLabel() string {
	m.ensureAgentReady()
	base := filepath.Base(m.agentWorkDir)
	if base == "" || base == "." || base == string(filepath.Separator) {
		return m.agentWorkDir
	}
	return base
}

func (m *Model) permissionBadgeLabel() string {
	m.ensureAgentReady()
	return string(m.permEngine.Mode())
}

func (m *Model) effectiveBaseURL() string {
	if m.chatBaseURL != "" {
		return m.chatBaseURL
	}
	return fmt.Sprintf("http://127.0.0.1:%d", m.effectivePort())
}

func (m *Model) estimateToolTokensHeuristic() int {
	m.ensureAgentReady()
	defs := m.toolRegistry.ToolDefs()
	if len(defs) == 0 {
		return 0
	}
	b, err := json.Marshal(defs)
	if err != nil {
		return 0
	}
	// OpenAI JSON tool schemas tokenize at roughly ~3.5 chars/token on Qwen/Llama tokenizers.
	return (len(b) + 3) / 4
}

func (m *Model) measureToolSchemaTokensCmd() tea.Cmd {
	m.ensureAgentReady()
	defs := m.toolRegistry.ToolDefs()
	if len(defs) == 0 {
		m.toolSchemaTokens = 0
		return nil
	}
	heur := m.estimateToolTokensHeuristic()
	if m.status != "READY" {
		return func() tea.Msg {
			return agentToolTokensMsg{tokens: heur, live: false}
		}
	}
	baseURL := m.effectiveBaseURL()
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		baseReq := runner.ChatRequest{
			Messages: []runner.APIMessage{{Role: "user", Content: "hi"}},
		}
		withTools := baseReq
		withTools.Tools = defs
		nWithout, err1 := runner.InputTokensURL(ctx, baseURL, baseReq)
		nWith, err2 := runner.InputTokensURL(ctx, baseURL, withTools)
		if err1 == nil && err2 == nil && nWith >= nWithout && nWith > 0 {
			return agentToolTokensMsg{tokens: nWith - nWithout, live: true}
		}
		return agentToolTokensMsg{tokens: heur, live: false}
	}
}

func (m *Model) buildAPIMessagesForLoop() []runner.APIMessage {
	apiMsgs := make([]runner.APIMessage, 0, len(m.chatMsgs)*2)
	for _, t := range m.chatMsgs {
		if t.CompactedOut {
			continue
		}
		switch t.Role {
		case "compaction":
			apiMsgs = append(apiMsgs, t.APIMessages...)
		case "user":
			if len(t.APIMessages) > 0 {
				apiMsgs = append(apiMsgs, t.APIMessages...)
			} else {
				apiMsgs = append(apiMsgs, runner.APIMessage{Role: "user", Content: t.Content})
			}
		case "assistant":
			if len(t.APIMessages) > 0 {
				apiMsgs = append(apiMsgs, t.APIMessages...)
			} else if t.Content != "" {
				apiMsgs = append(apiMsgs, runner.APIMessage{Role: "assistant", Content: t.Content})
			}
		}
	}
	return apiMsgs
}

func (m *Model) startAgentCompletion(prompt string) (tea.Model, tea.Cmd) {
	return m.startAgentTurnWithUser(chatTurn{Role: "user", Content: prompt}, true)
}

func (m *Model) startAgentTurnWithUser(userTurn chatTurn, appendUserTurn bool) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	if m.maybeShouldCompactNow() {
		_, _ = m.applyCompaction(false)
	}
	m.agentTurnCounter++
	turnNum := m.agentTurnCounter
	m.genStartTime = time.Now()

	if appendUserTurn {
		m.chatMsgs = append(m.chatMsgs, userTurn)
		m.attachmentTokens += attachmentTokensOf(userTurn.Attachments)
	}
	history := m.buildAPIMessagesForLoop()

	m.chatMsgs = append(m.chatMsgs, chatTurn{
		Role:        "assistant",
		thinkParser: &runner.ThinkStreamParser{},
	})
	m.chatStreaming = true
	m.chatGen++
	gen := m.chatGen

	ctx, cancel := context.WithCancel(context.Background())
	m.chatCancel = cancel

	var templateKwargs map[string]any
	effort := m.settings.ReasoningEffort
	if m.settings.Reasoning == "off" {
		effort = ""
		if m.thinkToggleSupported() {
			templateKwargs = map[string]any{"enable_thinking": false}
		}
	}

	req := agent.LoopRequest{
		BaseURL:            m.effectiveBaseURL(),
		SystemPrompt:       m.effectiveSystemPrompt(),
		History:            history,
		MaxSteps:           agent.DefaultMaxSteps,
		MaxTokens:          m.settings.MaxTokens,
		Temperature:        m.settings.Temp,
		ToolTemperature:    m.settings.ToolTemp,
		TopP:               m.settings.TopP,
		TopK:               m.settings.TopK,
		MinP:               m.settings.MinP,
		ReasoningEffort:    effort,
		ChatTemplateKwargs: templateKwargs,
		Registry:           m.toolRegistry,
		Permissions:        m.permEngine,
		Tainted:            m.sessionTainted,
		ToolCtx: agent.ToolContext{
			WorkspaceRoot: m.agentWorkDir,
			SkillsDir:     m.effectiveSkillsDir(),
			SkillsDataDir: m.effectiveSkillsDataDir(),
			SessionID:     m.sessionID,
			TurnNumber:    turnNum,
			ContextTokens: m.effectiveCtxTotal(),
			Checkpoints:   m.checkpoints,
			Shell:         agent.ShellCmd,
			AskUser:       nil, // populated below
		},
	}

	outCh := make(chan agentLoopMsg, 32)
	doneCh := make(chan agent.LoopResult, 1)

	// Unified raw action channel keeps text deltas, tool cards, permission modals,
	// and ask_user modals in strict chronological order while coalescing deltas at 60fps.
	type rawAction struct {
		ev         *agent.LoopEvent
		perm       *agent.PermissionPrompt
		permRep    chan<- agent.PermissionResponse
		askQ       string
		askOpts    []string
		askRep     chan<- askUserReply
		isAskReq   bool
		skillStage *agent.SkillStagePrompt
		skillRep   chan<- agent.SkillStageResponse
	}
	actionCh := make(chan rawAction, 64)

	req.ToolCtx.StageSkill = func(pCtx context.Context, pr agent.SkillStagePrompt) (agent.SkillStageResponse, error) {
		rep := make(chan agent.SkillStageResponse, 1)
		cp := pr
		select {
		case actionCh <- rawAction{skillStage: &cp, skillRep: rep}:
		case <-pCtx.Done():
			return agent.SkillStageResponse{Approved: false, Reason: "cancelled"}, pCtx.Err()
		}
		select {
		case ans := <-rep:
			return ans, nil
		case <-pCtx.Done():
			return agent.SkillStageResponse{Approved: false, Reason: "cancelled"}, pCtx.Err()
		}
	}

	req.ToolCtx.AskUser = func(pCtx context.Context, question string, options []string) (string, error) {
		rep := make(chan askUserReply, 1)
		select {
		case actionCh <- rawAction{isAskReq: true, askQ: question, askOpts: append([]string(nil), options...), askRep: rep}:
		case <-pCtx.Done():
			return "", pCtx.Err()
		}
		select {
		case ans := <-rep:
			return ans.Answer, ans.Err
		case <-pCtx.Done():
			return "", pCtx.Err()
		}
	}

	req.AskPermission = func(pCtx context.Context, pr agent.PermissionPrompt) (agent.PermissionResponse, error) {
		rep := make(chan agent.PermissionResponse, 1)
		cp := pr
		select {
		case actionCh <- rawAction{perm: &cp, permRep: rep}:
		case <-pCtx.Done():
			return agent.PermissionResponse{Choice: agent.PermDeny}, pCtx.Err()
		}
		select {
		case ans := <-rep:
			return ans, nil
		case <-pCtx.Done():
			return agent.PermissionResponse{Choice: agent.PermDeny}, pCtx.Err()
		}
	}

	req.OnEvent = func(ev agent.LoopEvent) {
		cp := ev
		select {
		case actionCh <- rawAction{ev: &cp}:
		case <-ctx.Done():
		}
	}

	go func() {
		res := agent.RunTurn(ctx, req)
		close(actionCh)
		doneCh <- res
	}()

	// Coalesce consecutive EvtDelta items at ~60fps while passing tool/modal events immediately in exact order.
	go func() {
		defer close(outCh)
		var pending runner.ChatDelta
		hasPending := false
		var lastFlush time.Time
		var timer *time.Timer
		var timerC <-chan time.Time
		// Heartbeats keep the spinner alive while a tool blocks the loop without
		// producing deltas (for example a slow build or test command).
		heartbeat := time.NewTicker(150 * time.Millisecond)
		defer heartbeat.Stop()
		modalPending := false

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

		flushDelta := func() bool {
			if !hasPending {
				return true
			}
			d := pending
			pending = runner.ChatDelta{}
			hasPending = false
			lastFlush = time.Now()
			select {
			case outCh <- agentLoopMsg{gen: gen, kind: agentEvtDelta, ev: agent.LoopEvent{Kind: agent.EvtDelta, Delta: d}}:
				return true
			case <-ctx.Done():
				return false
			}
		}

		for {
			select {
			case act, ok := <-actionCh:
				if !ok {
					stopTimer()
					flushDelta()
					return
				}
				modalPending = false
				if act.ev != nil && act.ev.Kind == agent.EvtDelta {
					d := act.ev.Delta
					if d.Done {
						stopTimer()
						if !flushDelta() {
							return
						}
						continue
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
					if d.CachedPromptTokens > 0 {
						pending.CachedPromptTokens = d.CachedPromptTokens
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
						if !flushDelta() {
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
					continue
				}

				// Non-delta event: flush any buffered text first so order stays chronological.
				stopTimer()
				if !flushDelta() {
					return
				}

				var msg agentLoopMsg
				msg.gen = gen
				switch {
				case act.perm != nil:
					msg.kind = agentEvtPermReq
					msg.perm = *act.perm
					msg.permRep = act.permRep
				case act.isAskReq:
					msg.kind = agentEvtAskReq
					msg.question = act.askQ
					msg.options = act.askOpts
					msg.askRep = act.askRep
				case act.skillStage != nil:
					msg.kind = agentEvtSkillStageReq
					msg.skillStage = *act.skillStage
					msg.skillRep = act.skillRep
				case act.ev != nil:
					msg.ev = *act.ev
					switch act.ev.Kind {
					case agent.EvtToolStart:
						msg.kind = agentEvtToolStart
					case agent.EvtToolEnd:
						msg.kind = agentEvtToolEnd
					case agent.EvtTurnMessages:
						msg.kind = agentEvtTurnMsgs
					}
				}
				switch msg.kind {
				case agentEvtPermReq, agentEvtAskReq, agentEvtSkillStageReq:
					modalPending = true
				}
				select {
				case outCh <- msg:
				case <-ctx.Done():
					return
				}
			case <-timerC:
				timerC = nil
				if !flushDelta() {
					return
				}
			case <-heartbeat.C:
				if modalPending {
					continue
				}
				select {
				case outCh <- agentLoopMsg{gen: gen, kind: agentEvtHeartbeat}:
				default:
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	m.refreshChatView(true)
	m.chatView.GotoBottom()
	return m, readAgentEvtCmd(gen, outCh, doneCh)
}

func readAgentEvtCmd(gen uint64, ch <-chan agentLoopMsg, doneCh <-chan agent.LoopResult) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			var res agent.LoopResult
			if doneCh != nil {
				res = <-doneCh
			}
			return agentLoopMsg{gen: gen, kind: agentEvtDone, res: res}
		}
		msg.ch = ch
		msg.doneCh = doneCh
		return msg
	}
}

func (m *Model) handleAgentLoopMsg(msg agentLoopMsg) (tea.Model, tea.Cmd) {
	if msg.gen != m.chatGen {
		return m, nil
	}
	switch msg.kind {
	case agentEvtDelta:
		m.handleChatDelta(msg.ev.Delta)
		return m, readAgentEvtCmd(msg.gen, msg.ch, msg.doneCh)

	case agentEvtHeartbeat:
		m.spinner, _ = m.spinner.Update(m.spinner.Tick())
		m.refreshChatView(false)
		return m, readAgentEvtCmd(msg.gen, msg.ch, msg.doneCh)

	case agentEvtToolStart:
		if len(m.chatMsgs) > 0 {
			t := &m.chatMsgs[len(m.chatMsgs)-1]
			if t.Role == "assistant" {
				t.markThinkDone()
				t.ToolCalls = append(t.ToolCalls, toolCard{
					CallID:     msg.ev.CallID,
					Name:       msg.ev.Tool,
					Summary:    msg.ev.Summary,
					ArgsPretty: msg.ev.ArgsPretty,
					Status:     "running",
				})
				t.ToolVer++
				m.spinner, _ = m.spinner.Update(m.spinner.Tick())
				m.refreshChatView(true)
			}
		}
		return m, readAgentEvtCmd(msg.gen, msg.ch, msg.doneCh)

	case agentEvtToolEnd:
		if len(m.chatMsgs) > 0 {
			t := &m.chatMsgs[len(m.chatMsgs)-1]
			if t.Role == "assistant" {
				status := "ok"
				if msg.ev.Result.Interrupted {
					status = "interrupted"
				} else if msg.ev.Result.IsError {
					if strings.Contains(strings.ToLower(msg.ev.Result.Output), "permission denied") {
						status = "denied"
					} else {
						status = "error"
					}
				}
				updated := false
				for i := len(t.ToolCalls) - 1; i >= 0; i-- {
					if t.ToolCalls[i].CallID == msg.ev.CallID {
						tc := &t.ToolCalls[i]
						tc.Status = status
						if msg.ev.Result.Summary != "" {
							tc.Summary = msg.ev.Result.Summary
						}
						tc.Output = msg.ev.Result.Output
						tc.Diff = msg.ev.Result.Diff
						tc.DiffAdded = msg.ev.Result.DiffAdded
						tc.DiffRemoved = msg.ev.Result.DiffRemoved
						tc.Duration = msg.ev.Duration
						updated = true
						break
					}
				}
				if !updated {
					t.ToolCalls = append(t.ToolCalls, toolCard{
						CallID:      msg.ev.CallID,
						Name:        msg.ev.Tool,
						Summary:     msg.ev.Result.Summary,
						Output:      msg.ev.Result.Output,
						Diff:        msg.ev.Result.Diff,
						DiffAdded:   msg.ev.Result.DiffAdded,
						DiffRemoved: msg.ev.Result.DiffRemoved,
						Status:      status,
						Duration:    msg.ev.Duration,
					})
				}
				t.ToolVer++
				m.refreshChatView(true)
			}
		}
		if msg.ev.Result.Tainted {
			m.sessionTainted = true
		}
		if msg.ev.Tool == "save_skill" || msg.ev.Tool == "use_skill" {
			m.reloadSkills()
		}
		return m, readAgentEvtCmd(msg.gen, msg.ch, msg.doneCh)

	case agentEvtTurnMsgs:
		if len(m.chatMsgs) > 0 {
			t := &m.chatMsgs[len(m.chatMsgs)-1]
			if t.Role == "assistant" {
				t.APIMessages = append(t.APIMessages, msg.ev.NewMessages...)
			}
		}
		return m, readAgentEvtCmd(msg.gen, msg.ch, msg.doneCh)

	case agentEvtPermReq:
		ni := textinput.New()
		ni.Prompt = ""
		ni.Placeholder = "reason / instructions for model..."
		ni.CharLimit = 240
		ni.Width = 44
		m.permModal = &permModalState{
			gen:       msg.gen,
			prompt:    msg.perm,
			replyCh:   msg.permRep,
			ch:        msg.ch,
			doneCh:    msg.doneCh,
			noteInput: ni,
			choice:    permChoiceAllow,
		}
		// Show the prompt inline with the chat, never on the logs tab.
		m.runTab = runTabChat
		m.chatInput.Blur()
		m.refreshChatView(true)
		m.chatView.GotoBottom()
		return m, nil // pause event pump until user decides

	case agentEvtAskReq:
		ti := textinput.New()
		ti.Prompt = ""
		ti.Placeholder = "type your answer..."
		ti.CharLimit = 500
		ti.Width = 48
		inputMode := len(msg.options) == 0
		if inputMode {
			ti.Focus()
		}
		m.runTab = runTabChat
		m.chatInput.Blur()
		m.askModal = &askModalState{
			gen:       msg.gen,
			question:  msg.question,
			options:   msg.options,
			replyCh:   msg.askRep,
			ch:        msg.ch,
			doneCh:    msg.doneCh,
			input:     ti,
			inputMode: inputMode,
		}
		m.refreshChatView(false)
		return m, nil // pause event pump until user answers

	case agentEvtSkillStageReq:
		ni := textinput.New()
		ni.Prompt = ""
		ni.Placeholder = "rejection reason for model..."
		ni.CharLimit = 240
		ni.Width = 44

		nameIn := textinput.New()
		nameIn.Prompt = ""
		nameIn.CharLimit = 40
		nameIn.Width = 36
		nameIn.SetValue(msg.skillStage.Name)

		descIn := textinput.New()
		descIn.Prompt = ""
		descIn.CharLimit = 240
		descIn.Width = 54
		descIn.SetValue(msg.skillStage.Description)

		m.skillApproveModal = &skillApproveModalState{
			gen:       msg.gen,
			prompt:    msg.skillStage,
			replyCh:   msg.skillRep,
			ch:        msg.ch,
			doneCh:    msg.doneCh,
			noteInput: ni,
			nameInput: nameIn,
			descInput: descIn,
			editBody:  msg.skillStage.Body,
		}
		return m, nil // pause event pump until user decides

	case agentEvtDone:
		m.permModal = nil
		m.askModal = nil
		m.skillApproveModal = nil
		if msg.res.Tainted {
			m.sessionTainted = true
		}
		if len(m.chatMsgs) > 0 {
			t := &m.chatMsgs[len(m.chatMsgs)-1]
			if t.Role == "assistant" {
				// Persist the full tool-call/tool-result exchange so the next turn
				// sees valid history instead of an empty assistant message.
				if len(msg.res.MessagesAppended) > 0 {
					t.APIMessages = msg.res.MessagesAppended
				}
				if msg.res.PromptTokens > 0 || msg.res.CompletionTokens > 0 {
					t.Tokens = msg.res.CompletionTokens
					m.chatCtxUsed = msg.res.PromptTokens + msg.res.CompletionTokens
				}
				if msg.res.CachedTokens > 0 {
					m.cachedPrefixTokens = msg.res.CachedTokens
				}
				if msg.res.GenTokPerSec > 0 {
					t.GenTokSec = msg.res.GenTokPerSec
					m.metrics.GenTokPerSec = msg.res.GenTokPerSec
				}
				if msg.res.PromptTokSec > 0 {
					m.metrics.PromptTokSec = msg.res.PromptTokSec
				}
				if msg.res.Interrupted || msg.res.Err != nil {
					if msg.res.Interrupted {
						t.Aborted = true
					}
					for i := range t.ToolCalls {
						if t.ToolCalls[i].Status == "running" {
							if msg.res.Interrupted {
								t.ToolCalls[i].Status = "interrupted"
							} else {
								t.ToolCalls[i].Status = "error"
							}
						}
					}
					t.ToolVer++
				}
			}
		}
		if m.cfg != nil && m.cfg.LongGenBell && !m.genStartTime.IsZero() && time.Since(m.genStartTime) >= 20*time.Second {
			_, _ = bellWriter.Write([]byte{'\a'})
		}
		m.genStartTime = time.Time{}
		m.finishChatStream(msg.res.Err)
		if !msg.res.Interrupted && msg.res.Err == nil && m.maybeShouldCompactNow() {
			_, _ = m.applyCompaction(false)
		}
		m.saveCurrentSession()
		return m, nil
	}
	return m, nil
}

func (m *Model) toggleToolCardsDisplay() {
	anyExpanded := false
	hasCards := false
	for i := range m.chatMsgs {
		t := &m.chatMsgs[i]
		if t.Role == "compaction" {
			hasCards = true
			if t.ThinkExpanded != nil && *t.ThinkExpanded {
				anyExpanded = true
			}
		}
		for j := range t.ToolCalls {
			hasCards = true
			tc := &t.ToolCalls[j]
			if tc.Expanded != nil && *tc.Expanded {
				anyExpanded = true
			}
		}
	}
	if !hasCards {
		m.toastf("no collapsible cards yet")
		return
	}
	next := !anyExpanded
	for i := range m.chatMsgs {
		t := &m.chatMsgs[i]
		if t.Role == "compaction" {
			val := next
			t.ThinkExpanded = &val
		}
		for j := range t.ToolCalls {
			val := next
			t.ToolCalls[j].Expanded = &val
		}
		t.ToolVer++
	}
	m.refreshChatView(false)
	if next {
		m.toastf("card details expanded")
	} else {
		m.toastf("card details collapsed")
	}
}

func renderToolCardBlock(m *Model, tc toolCard, innerW int, streaming bool, spinFrame string) []string {
	icon := "✓"
	if m.sty.ascii {
		icon = "[ok]"
	}
	cardSpin := ""
	isRunning := tc.Status == "running" && streaming
	switch tc.Status {
	case "running":
		if streaming {
			cardSpin = spinFrame
			icon = ""
		} else {
			icon = "■"
		}
	case "error":
		icon = "✗"
		if m.sty.ascii {
			icon = "[err]"
		}
	case "denied":
		icon = "⊘"
		if m.sty.ascii {
			icon = "[deny]"
		}
	case "interrupted":
		icon = "■"
		if m.sty.ascii {
			icon = "[stop]"
		}
	}

	title := tc.Name
	if tc.Summary != "" {
		title += " · " + tc.Summary
	}
	if icon != "" {
		title = icon + " " + title
	}
	var extras []string
	if tc.DiffAdded > 0 || tc.DiffRemoved > 0 {
		extras = append(extras, fmt.Sprintf("+%d/-%d", tc.DiffAdded, tc.DiffRemoved))
	}
	if tc.Duration > 0 {
		if tc.Duration < time.Second {
			extras = append(extras, fmt.Sprintf("%dms", tc.Duration.Milliseconds()))
		} else {
			extras = append(extras, fmt.Sprintf("%.1fs", tc.Duration.Seconds()))
		}
	}
	if len(extras) > 0 {
		title += " (" + strings.Join(extras, ", ") + ")"
	}

	expanded := tc.Status == "error" || tc.Status == "denied"
	if tc.Expanded != nil {
		expanded = *tc.Expanded
	}

	var bodyParts []string
	if tc.ArgsPretty != "" {
		bodyParts = append(bodyParts, "Args: "+tc.ArgsPretty)
	}
	if tc.Diff != "" {
		bodyParts = append(bodyParts, "Diff:\n"+tc.Diff)
	}
	if tc.Output != "" {
		outStr := tc.Output
		if !expanded && len(outStr) > 240 {
			outStr = outStr[:240] + "…"
		} else if len(outStr) > 2400 {
			outStr = outStr[:2400] + "\n… [output clipped]"
		}
		bodyParts = append(bodyParts, "Result:\n"+outStr)
	}

	lines := renderCollapsible(m.sty, innerW, cardSpin, collapsibleBlock{
		Summary:   title,
		Body:      strings.Join(bodyParts, "\n"),
		RightHint: "[ctrl+o]",
		Expanded:  expanded,
		Streaming: isRunning,
	})
	if expanded && tc.Diff != "" {
		// Re-tint diff lines inside expanded collapsible card (+ green, - red, @@ accent2).
		for i, ln := range lines {
			plain := strings.TrimPrefix(stripANSI(ln), "  │ ")
			plainASCII := strings.TrimPrefix(plain, "| ")
			if strings.HasPrefix(plain, "+") && !strings.HasPrefix(plain, "+++") {
				lines[i] = m.sty.sSuccess.Render(stripANSI(ln))
			} else if strings.HasPrefix(plain, "-") && !strings.HasPrefix(plain, "---") {
				lines[i] = m.sty.sDanger.Render(stripANSI(ln))
			} else if strings.HasPrefix(plainASCII, "@@") {
				lines[i] = m.sty.accent2().Render(stripANSI(ln))
			}
		}
	}
	return lines
}

// --- Modal key handlers ---

func (m *Model) finishPermission(resp agent.PermissionResponse, resume bool) (tea.Model, tea.Cmd) {
	pm := m.permModal
	if pm == nil {
		return m, nil
	}
	// Buffered reply channel: the agent may already be unwinding after Esc.
	select {
	case pm.replyCh <- resp:
	default:
	}
	gen, ch, doneCh := pm.gen, pm.ch, pm.doneCh
	m.permModal = nil
	m.refreshChatView(true)
	m.chatView.GotoBottom()
	if m.runTab == runTabChat {
		m.chatInput.Focus()
	}
	if !resume {
		return m, nil
	}
	return m, readAgentEvtCmd(gen, ch, doneCh)
}

func (m *Model) abortTurnFromPrompt() (tea.Model, tea.Cmd) {
	pm := m.permModal
	if pm == nil {
		return m, nil
	}
	select {
	case pm.replyCh <- agent.PermissionResponse{Choice: agent.PermDeny}:
	default:
	}
	m.permModal = nil
	m.stopChatStream()
	m.chatInput.Focus()
	m.toastf("turn cancelled")
	return m, nil
}

func (m *Model) movePermissionChoice(delta int) {
	pm := m.permModal
	if pm == nil {
		return
	}
	for i := 0; i < 3; i++ {
		pm.choice = (pm.choice + delta + 3) % 3
		if pm.choice != permChoiceAlways || !pm.prompt.HighRisk {
			break
		}
	}
	m.refreshChatView(false)
}

func (m *Model) choosePermission(choice int) (tea.Model, tea.Cmd) {
	pm := m.permModal
	if pm == nil {
		return m, nil
	}
	if choice == permChoiceAlways && pm.prompt.HighRisk {
		m.toastf("high-risk commands cannot be always-allowed")
		return m, nil
	}
	switch choice {
	case permChoiceAllow:
		return m.finishPermission(agent.PermissionResponse{Choice: agent.PermAllowOnce}, true)
	case permChoiceAlways:
		return m.finishPermission(agent.PermissionResponse{Choice: agent.PermAllowSession}, true)
	default:
		return m.finishPermission(agent.PermissionResponse{Choice: agent.PermDeny}, true)
	}
}

func (m *Model) handlePermModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	pm := m.permModal
	if pm == nil {
		return m, nil
	}

	if pm.enteringNote {
		switch msg.Type {
		case tea.KeyEnter:
			note := strings.TrimSpace(pm.noteInput.Value())
			return m.finishPermission(agent.PermissionResponse{Choice: agent.PermDenyWithNote, Note: note}, true)
		case tea.KeyEsc:
			pm.enteringNote = false
			pm.noteInput.Blur()
			m.refreshChatView(false)
			return m, nil
		}
		var c tea.Cmd
		pm.noteInput, c = pm.noteInput.Update(msg)
		m.refreshChatView(false)
		return m, c
	}

	switch msg.String() {
	case "up", "k":
		m.movePermissionChoice(-1)
		return m, nil
	case "down", "j", "tab":
		m.movePermissionChoice(1)
		return m, nil
	case "1", "y":
		return m.choosePermission(permChoiceAllow)
	case "2", "a":
		return m.choosePermission(permChoiceAlways)
	case "3", "n":
		return m.choosePermission(permChoiceDeny)
	case "enter":
		return m.choosePermission(pm.choice)
	case "esc":
		return m.abortTurnFromPrompt()
	case "d":
		pm.enteringNote = true
		pm.noteInput.Focus()
		m.refreshChatView(false)
		return m, nil
	}
	return m, nil
}

func (m *Model) handleAskModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	am := m.askModal
	if am == nil {
		return m, nil
	}
	respond := func(ans string, err error) (tea.Model, tea.Cmd) {
		select {
		case am.replyCh <- askUserReply{Answer: ans, Err: err}:
		default:
		}
		gen, ch, doneCh := am.gen, am.ch, am.doneCh
		m.askModal = nil
		m.refreshChatView(true)
		m.chatView.GotoBottom()
		if m.runTab == runTabChat {
			m.chatInput.Focus()
		}
		return m, readAgentEvtCmd(gen, ch, doneCh)
	}

	beginTyping := func(seed string) (tea.Model, tea.Cmd) {
		am.inputMode = true
		am.cursor = len(am.options)
		am.input.SetValue(seed)
		am.input.CursorEnd()
		am.input.Focus()
		return m, nil
	}
	backToOptions := func() (tea.Model, tea.Cmd) {
		am.inputMode = false
		am.input.Blur()
		if n := len(am.options); n > 0 && am.cursor >= n {
			am.cursor = n - 1
		}
		return m, nil
	}

	if len(am.options) > 0 && !am.inputMode {
		switch msg.Type {
		case tea.KeyUp:
			if am.cursor > 0 {
				am.cursor--
			}
			return m, nil
		case tea.KeyDown:
			if am.cursor < len(am.options) {
				am.cursor++
			}
			return m, nil
		case tea.KeyEnter:
			if am.cursor < 0 || am.cursor > len(am.options) {
				am.cursor = 0
			}
			if am.cursor == len(am.options) {
				return beginTyping("")
			}
			return respond(am.options[am.cursor], nil)
		case tea.KeyTab:
			return beginTyping("")
		case tea.KeyEsc:
			return respond("", context.Canceled)
		case tea.KeySpace:
			return beginTyping(" ")
		case tea.KeyRunes:
			if len(msg.Runes) == 1 && msg.Runes[0] >= '1' && msg.Runes[0] <= '9' {
				n := int(msg.Runes[0] - '0')
				if n <= len(am.options) {
					return respond(am.options[n-1], nil)
				}
				return m, nil
			}
			// Typing anything else starts a free-form answer.
			return beginTyping(string(msg.Runes))
		}
		return m, nil
	}

	switch msg.Type {
	case tea.KeyEnter:
		ans := strings.TrimSpace(am.input.Value())
		if ans == "" {
			if len(am.options) > 0 {
				return backToOptions()
			}
			return m, nil
		}
		return respond(ans, nil)
	case tea.KeyEsc:
		if len(am.options) > 0 {
			return backToOptions()
		}
		return respond("", context.Canceled)
	}
	var c tea.Cmd
	am.input, c = am.input.Update(msg)
	return m, c
}

func (m *Model) handleToolsModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	groups := m.toolRegistry.GroupsOrdered()
	if len(groups) == 0 {
		m.toolsModal = nil
		return m, nil
	}
	switch msg.String() {
	case "esc", "q":
		m.toolsModal = nil
		return m, nil
	case "up", "k":
		if m.toolsModal.cursor > 0 {
			m.toolsModal.cursor--
		}
		return m, nil
	case "down", "j":
		if m.toolsModal.cursor < len(groups)-1 {
			m.toolsModal.cursor++
		}
		return m, nil
	case " ", "enter":
		idx := m.toolsModal.cursor
		if idx >= 0 && idx < len(groups) {
			g := groups[idx]
			next := !g.Enabled
			m.toolRegistry.SetGroupEnabled(g.Name, next)
			if g.Name == "exec" && next {
				m.appendExecHonestyNotice()
			}
			if g.Name == "web" && next {
				m.appendWebNotice()
			}
			m.toastf("tool group %s: %s", g.Name, map[bool]string{true: "on", false: "off"}[next])
			return m, m.measureToolSchemaTokensCmd()
		}
		return m, nil
	}
	return m, nil
}

func (m *Model) handlePermsModalKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	m.ensureAgentReady()
	switch msg.String() {
	case "esc", "q", "enter":
		m.permsModal = nil
		return m, nil
	case "1":
		m.permEngine.SetMode(agent.ModeAsk)
		m.toastf("permissions: ask")
		return m, nil
	case "2":
		m.permEngine.SetMode(agent.ModeReadAuto)
		m.toastf("permissions: read-auto")
		return m, nil
	case "3":
		m.permEngine.SetMode(agent.ModeWorkspaceAuto)
		m.toastf("permissions: workspace-auto")
		return m, nil
	case "c":
		m.permEngine.ClearAllowlist()
		m.toastf("session allowlist cleared")
		return m, nil
	case "t":
		m.sessionTainted = false
		m.toastf("session taint cleared")
		return m, nil
	case "up", "k":
		if m.permsModal.scroll > 0 {
			m.permsModal.scroll--
		}
		return m, nil
	case "down", "j":
		m.permsModal.scroll++
		return m, nil
	}
	return m, nil
}

func (m *Model) appendExecHonestyNotice() {
	if m.execNoticeShown {
		return
	}
	m.execNoticeShown = true
	m.appendInfoTurn("Warning: run_command executes shell commands with your Windows user permissions. Path sandboxing confines file tools only, and /undo cannot revert shell side effects.")
}

func (m *Model) appendWebNotice() {
	if m.webNoticeShown {
		return
	}
	m.webNoticeShown = true
	m.appendInfoTurn("Web tools query public search engines and pages, with no keys. Queries and fetched URLs leave this machine; fetched text is untrusted data.")
}

// --- Modal overlays ---

// renderPermissionInline draws the pending tool permission as a card inside the chat
// transcript, so the turn never hides the conversation behind a modal overlay.
func (m *Model) renderPermissionInline(innerW int) []string {
	pm := m.permModal
	if pm == nil || innerW < 24 {
		return nil
	}
	cardW := innerW
	if cardW > 100 {
		cardW = 100
	}
	contentW := cardW - 4
	if contentW < 12 {
		contentW = 12
	}

	var lines []string
	hdr := m.sty.sAccentBold.Render("Permission needed · " + pm.prompt.Tool)
	if pm.prompt.HighRisk {
		hdr += "  " + m.sty.badge(" HIGH-RISK ", m.sty.t.Error)
	}
	lines = append(lines, truncateANSI(hdr, contentW))
	// Keep the whole card inside the chat viewport on 80x24 terminals; the
	// tool, reason and choices matter more than a long diff preview.
	maxPreview := m.height - 20
	if maxPreview < 2 {
		maxPreview = 2
	}
	if maxPreview > 14 {
		maxPreview = 14
	}
	if pm.prompt.Summary != "" && pm.prompt.Diff == "" {
		for _, sl := range wrapText(pm.prompt.Summary, contentW) {
			lines = append(lines, m.sty.sText.Render(sl))
		}
	}
	if pm.prompt.Reason != "" {
		for _, rl := range wrapText("Reason: "+pm.prompt.Reason, contentW) {
			lines = append(lines, m.sty.sWarning.Render(rl))
		}
	}
	if pm.prompt.Tool == "run_command" {
		for _, nl := range wrapText("Runs with your own user rights; the sandbox confines file tools only.", contentW) {
			lines = append(lines, m.sty.sMuted.Render(nl))
		}
	}

	if pm.prompt.Diff != "" {
		diffInfo := fmt.Sprintf("Diff (%s): +%d / -%d lines", pm.prompt.DiffPath, pm.prompt.DiffAdded, pm.prompt.DiffRemoved)
		lines = append(lines, truncateANSI(m.sty.accent2().Render(diffInfo), contentW))
		diffLines := strings.Split(strings.TrimRight(pm.prompt.Diff, "\n"), "\n")
		shown := len(diffLines)
		truncated := false
		if shown > maxPreview {
			shown = maxPreview - 1
			if shown < 0 {
				shown = 0
			}
			truncated = true
		}
		for _, dl := range diffLines[:shown] {
			txt := truncate(dl, contentW)
			styled := m.sty.sMuted.Render(txt)
			switch {
			case strings.HasPrefix(dl, "+") && !strings.HasPrefix(dl, "+++"):
				styled = m.sty.sSuccess.Render(txt)
			case strings.HasPrefix(dl, "-") && !strings.HasPrefix(dl, "---"):
				styled = m.sty.sDanger.Render(txt)
			case strings.HasPrefix(dl, "@@"):
				styled = m.sty.accent2().Render(txt)
			}
			lines = append(lines, styled)
		}
		if truncated {
			lines = append(lines, m.sty.sMuted.Render(fmt.Sprintf("… %d more diff lines (expand the tool card after deciding)", len(diffLines)-shown)))
		}
	} else if pm.prompt.ArgsPretty != "" {
		wrapped := wrapText(pm.prompt.ArgsPretty, contentW)
		limit := maxPreview
		if len(wrapped) > limit {
			limit--
			if limit < 0 {
				limit = 0
			}
		}
		for i, al := range wrapped {
			if i >= limit {
				lines = append(lines, m.sty.sMuted.Render(fmt.Sprintf("… %d more lines", len(wrapped)-i)))
				break
			}
			lines = append(lines, m.sty.sText.Render(al))
		}
	}

	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", contentW)))
	if pm.enteringNote {
		lines = append(lines, m.sty.sWarning.Render("Deny note: ")+pm.noteInput.View())
		lines = append(lines, truncateANSI(m.sty.sMuted.Render("enter submit deny+note · esc back"), contentW))
	} else {
		type permOpt struct {
			label string
			hint  string
		}
		opts := []permOpt{
			{"Allow once", "run this call only"},
			{"Always allow " + pm.prompt.Tool, "no more prompts for this tool this session"},
			{"Don't allow", "model receives a denial"},
		}
		if pm.prompt.HighRisk {
			opts[permChoiceAlways] = permOpt{"Always allow", "unavailable for high-risk commands"}
		}
		for i, opt := range opts {
			marker := "  "
			st := m.sty.sText
			if i == pm.choice {
				marker = "▸ "
				st = m.sty.sAccentBold
			}
			if i == permChoiceAlways && pm.prompt.HighRisk {
				st = m.sty.sMuted
			}
			row := fmt.Sprintf("%s%d. %-24s %s", marker, i+1, opt.label, opt.hint)
			lines = append(lines, st.Render(truncate(row, contentW)))
		}
		lines = append(lines, truncateANSI(m.sty.sMuted.Render("↑↓ choose · enter confirm · 1/2/3 · d note · esc cancel"), contentW))
	}

	box := m.sty.panel(cardW, len(lines)+3, "Agent Permission", true, lines)
	return append(strings.Split(box, "\n"), "")
}

func (m *Model) overlayAskModal(bg string) string {
	am := m.askModal
	if am == nil {
		return bg
	}
	w, h := m.width, m.height
	boxW := w - 12
	if boxW > 76 {
		boxW = 76
	}
	if boxW < 40 {
		boxW = 40
	}
	innerW := boxW - 4

	var lines []string
	qLines := wrapText(am.question, innerW)
	// Keep the question from pushing the choices/input off the bottom of the screen.
	maxQLines := h - 10
	if maxQLines < 2 {
		maxQLines = 2
	}
	if len(qLines) > maxQLines {
		qLines = append(qLines[:maxQLines-1], "…")
	}
	for _, ql := range qLines {
		lines = append(lines, m.sty.sText.Bold(true).Render(ql))
	}
	lines = append(lines, "")

	if len(am.options) > 0 && !am.inputMode {
		// Options wrap instead of being cut off at the box edge. Keep the whole
		// modal inside the terminal by windowing very long option lists.
		maxOpts := (h - 9 - len(qLines)) / 3
		if maxOpts < 1 {
			maxOpts = 1
		}
		if maxOpts > len(am.options) {
			maxOpts = len(am.options)
		}
		if am.cursor < len(am.options) {
			if am.cursor < am.scroll {
				am.scroll = am.cursor
			}
			if am.cursor >= am.scroll+maxOpts {
				am.scroll = am.cursor - maxOpts + 1
			}
		}
		maxScroll := len(am.options) - maxOpts
		if maxScroll < 0 {
			maxScroll = 0
		}
		if am.scroll > maxScroll {
			am.scroll = maxScroll
		}
		if am.scroll < 0 {
			am.scroll = 0
		}
		end := am.scroll + maxOpts
		if end > len(am.options) {
			end = len(am.options)
		}
		if am.scroll > 0 {
			lines = append(lines, m.sty.sMuted.Render(fmt.Sprintf("  ↑ %d more", am.scroll)))
		}
		for i := am.scroll; i < end; i++ {
			pref := "  "
			st := m.sty.sText
			if i == am.cursor {
				pref = "▸ "
				st = m.sty.sAccentBold
			}
			head := fmt.Sprintf("%s%d. ", pref, i+1)
			cont := strings.Repeat(" ", displayWidth(head))
			wrapped := wrapText(am.options[i], innerW-displayWidth(head))
			if len(wrapped) > 3 {
				wrapped = wrapped[:3]
				wrapped[2] += "…"
			}
			for j, ol := range wrapped {
				line := head + ol
				if j > 0 {
					line = cont + ol
				}
				lines = append(lines, st.Render(line))
			}
		}
		if end < len(am.options) {
			lines = append(lines, m.sty.sMuted.Render(fmt.Sprintf("  ↓ %d more", len(am.options)-end)))
		}
		sel := am.cursor == len(am.options)
		pref, st := "  ", m.sty.sText
		if sel {
			pref, st = "▸ ", m.sty.sAccentBold
		}
		lines = append(lines, st.Render(pref+"✎ Type a different answer…"))
		lines = append(lines, "")
		lines = append(lines, truncateANSI(m.sty.sMuted.Render("↑↓ choose · 1-9 pick · type to answer · enter confirm · esc cancel"), innerW))
	} else {
		if am.inputMode {
			lines = append(lines, m.sty.sAccent.Render("› ")+am.input.View())
			lines = append(lines, "")
			if len(am.options) > 0 {
				lines = append(lines, m.sty.sMuted.Render("enter submit · esc back to choices"))
			} else {
				lines = append(lines, m.sty.sMuted.Render("enter submit · esc cancel"))
			}
		} else {
			lines = append(lines, m.sty.sAccent.Render("› ")+am.input.View())
			lines = append(lines, "")
			lines = append(lines, m.sty.sMuted.Render("enter submit · esc cancel"))
		}
	}

	box := m.sty.panel(boxW, len(lines)+3, "Agent Question", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}

func (m *Model) overlayToolsModal(bg string) string {
	m.ensureAgentReady()
	w, h := m.width, m.height
	boxW := 66
	if boxW > w-6 {
		boxW = w - 6
	}
	if boxW < 38 {
		boxW = 38
	}
	innerW := boxW - 4

	groups := m.toolRegistry.GroupsOrdered()
	lines := []string{
		m.sty.sMuted.Render("Toggle tool groups active for /v1/chat/completions:"),
		"",
	}
	for i, g := range groups {
		cursor := "  "
		if m.toolsModal != nil && i == m.toolsModal.cursor {
			cursor = "> "
		}
		state := m.sty.sMuted.Render("[off]")
		if g.Enabled {
			state = m.sty.sSuccess.Render("[on ]")
		}
		note := ""
		if g.Name == "exec" {
			note = " · runs with OS user rights"
		}
		lines = append(lines, truncateANSI(cursor+state+" "+m.sty.sText.Bold(true).Render(g.Name)+m.sty.sMuted.Render(fmt.Sprintf(" (%d tools%s)", g.Count, note)), innerW))
	}
	if wsStatus := m.toolRegistry.WebSearchStatus(); wsStatus != "" {
		st := m.sty.sMuted
		if strings.Contains(wsStatus, "disabled") {
			st = m.sty.sWarning
		}
		for _, wl := range wrapText(wsStatus, innerW) {
			lines = append(lines, st.Render(wl))
		}
	}
	if m.mcpManager != nil {
		if sum := m.mcpManager.Summary(); sum.Warning != "" {
			for _, wl := range wrapText(sum.Warning, innerW) {
				lines = append(lines, m.sty.sWarning.Render(wl))
			}
		}
	}
	toks := m.toolSchemaTokens
	tokLabel := "estimated ~"
	if m.toolSchemaLive {
		tokLabel = "measured via /input_tokens: "
	}
	if toks == 0 && len(m.toolRegistry.ToolDefs()) > 0 {
		toks = m.estimateToolTokensHeuristic()
	}
	lines = append(lines, "")
	lines = append(lines, truncateANSI(m.sty.accent2().Render(fmt.Sprintf("Active tool schema cost: %s%d tok", tokLabel, toks)), innerW))
	lines = append(lines, m.sty.sMuted.Render("↑↓ select · space/enter toggle · esc close"))

	box := m.sty.panel(boxW, len(lines)+3, "Agent Tools (/tools)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}

func (m *Model) overlayPermsModal(bg string) string {
	m.ensureAgentReady()
	w, h := m.width, m.height
	boxW := 74
	if boxW > w-6 {
		boxW = w - 6
	}
	if boxW < 42 {
		boxW = 42
	}
	innerW := boxW - 4

	mode := m.permEngine.Mode()
	mark := func(target agent.PermissionMode) string {
		if mode == target {
			return m.sty.sSuccess.Render("● ")
		}
		return m.sty.sMuted.Render("○ ")
	}
	lines := []string{
		truncateANSI(mark(agent.ModeAsk)+"[1] ask             always prompt before read/write/exec", innerW),
		truncateANSI(mark(agent.ModeReadAuto)+"[2] read-auto       auto-allow workspace reads; ask writes/exec (default)", innerW),
		truncateANSI(mark(agent.ModeWorkspaceAuto)+"[3] workspace-auto  auto-allow reads & writes in workspace; always ask exec", innerW),
		m.sty.sMuted.Render(strings.Repeat("─", innerW)),
		truncateANSI(m.sty.sWarning.Render("Security note: path jail confines file tools only; shell commands run as you."), innerW),
		m.sty.sMuted.Render(fmt.Sprintf("Workspace: %s", truncate(m.agentWorkDir, innerW-12))),
	}
	if m.sessionTainted {
		lines = append(lines, truncateANSI(m.sty.sWarning.Render("Session taint: TAINTED — untrusted web content loaded; writes/exec/net ask ([t] clear)"), innerW))
	} else {
		lines = append(lines, m.sty.sMuted.Render("Session taint: CLEAN"))
	}
	lines = append(lines, m.sty.sMuted.Render(strings.Repeat("─", innerW)))
	entries := m.permEngine.Allowlist()
	if len(entries) == 0 {
		lines = append(lines, m.sty.sMuted.Render("Session allowlist: (empty)"))
	} else {
		lines = append(lines, m.sty.sAccent2Bold.Render(fmt.Sprintf("Session allowlist (%d):", len(entries))))
		start := m.permsModal.scroll
		if start > len(entries)-6 {
			start = len(entries) - 6
		}
		if start < 0 {
			start = 0
		}
		end := start + 6
		if end > len(entries) {
			end = len(entries)
		}
		for _, e := range entries[start:end] {
			lines = append(lines, truncateANSI("  • "+e.Tool+": "+e.Summary, innerW))
		}
	}
	lines = append(lines, "")
	lines = append(lines, m.sty.sMuted.Render("1/2/3 mode · c clear allowlist · t clear taint · esc close"))

	box := m.sty.panel(boxW, len(lines)+3, "Agent Permissions (/permissions)", true, lines)
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box,
		lipgloss.WithWhitespaceChars(" "), lipgloss.WithWhitespaceForeground(m.sty.t.Bg))
}
