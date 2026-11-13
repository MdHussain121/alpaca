package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/web"
)

const (
	DefaultMaxSteps        = 25
	DefaultMaxIdentical    = 3
	DefaultMaxToolBytesOut = 192 * 1024
)

// PermissionChoice is returned by the UI dialog for a paused permission check.
type PermissionChoice int

const (
	PermAllowOnce PermissionChoice = iota
	PermAllowSession
	PermDeny
	PermDenyWithNote
)

// PermissionPrompt describes a tool waiting on interactive approval.
type PermissionPrompt struct {
	CallID      string
	Tool        string
	Summary     string
	Reason      string
	Class       ToolClass
	HighRisk    bool
	ArgsPretty  string
	Diff        string
	DiffPath    string
	DiffAdded   int
	DiffRemoved int
	RawArgs     map[string]any
}

// PermissionResponse captures the user's permission dialog decision.
type PermissionResponse struct {
	Choice PermissionChoice
	Note   string
}

// EventKind identifies incremental agent loop notifications to the UI.
type EventKind int

const (
	EvtDelta EventKind = iota
	EvtToolStart
	EvtToolEnd
	EvtTurnMessages // emits assistant tool-call / tool result APIMessages persisted in history
)

// LoopEvent notifies the UI about live streaming text, reasoning, or tool calls.
type LoopEvent struct {
	Kind        EventKind
	Delta       runner.ChatDelta
	CallID      string
	Tool        string
	Summary     string
	ArgsPretty  string
	Result      ToolExecResult
	Duration    time.Duration
	NewMessages []runner.APIMessage
}

// LoopRequest supplies everything required to execute one user prompt turn in agent mode.
type LoopRequest struct {
	BaseURL            string
	SystemPrompt       string
	History            []runner.APIMessage
	MaxSteps           int
	MaxTokens          int
	Temperature        float64
	ToolTemperature    float64
	TopP               float64
	TopK               int
	MinP               float64
	ReasoningEffort    string
	ChatTemplateKwargs map[string]any
	Registry           *Registry
	Permissions        *PermissionEngine
	ToolCtx            ToolContext
	Tainted            bool
	AskPermission      func(ctx context.Context, req PermissionPrompt) (PermissionResponse, error)
	OnEvent            func(LoopEvent)
}

// LoopResult summarizes the finished agent turn for history & telemetry.
type LoopResult struct {
	FinalContent     string
	Steps            int
	PromptTokens     int
	CachedTokens     int
	CompletionTokens int
	GenTokPerSec     float64
	PromptTokSec     float64
	Interrupted      bool
	Tainted          bool
	MessagesAppended []runner.APIMessage
	Err              error
}

func appendRetrievedSources(text string, reg *Registry) (string, string) {
	if reg == nil {
		return text, ""
	}
	session := reg.WebSession()
	urls := session.FetchedSnapshot()
	if len(urls) == 0 && session.ResultCount() == 0 {
		return text, ""
	}

	cleaned, removed := removeUnverifiedURLs(text, session)
	if len(urls) == 0 {
		if !removed {
			return cleaned, ""
		}
		suffix := "\n\nSources: no page was retrieved; an unverified URL was removed."
		return strings.TrimRight(cleaned, " \n") + suffix, suffix
	}
	if !removed && (strings.Contains(strings.ToLower(cleaned), "http://") || strings.Contains(strings.ToLower(cleaned), "https://")) {
		return cleaned, ""
	}
	var b strings.Builder
	b.WriteString("\n\nSources:\n")
	for i, u := range urls {
		if i >= 5 {
			break
		}
		fmt.Fprintf(&b, "- [%d] %s\n", i+1, u)
	}
	suffix := b.String()
	return strings.TrimRight(cleaned, " \n") + suffix, suffix
}

func removeUnverifiedURLs(text string, session *web.Session) (string, bool) {
	removed := false
	out := text
	for _, field := range strings.Fields(text) {
		if !strings.Contains(field, "://") {
			continue
		}
		u := strings.Trim(field, " \t\r\n()[]<>{}\"'.,;:!?")
		if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
			continue
		}
		if session.IsAllowed(u) {
			continue
		}
		out = strings.ReplaceAll(out, u, "")
		removed = true
	}
	return out, removed
}

// RunTurn executes one full multi-step tool loop until the model emits a plain assistant
// response (without tool calls), hits MaxSteps, or is interrupted via ctx.
func RunTurn(ctx context.Context, req LoopRequest) LoopResult {
	maxSteps := req.MaxSteps
	if maxSteps <= 0 {
		maxSteps = DefaultMaxSteps
	}

	var toolDefs []runner.ToolDef
	if req.Registry != nil {
		toolDefs = req.Registry.ToolDefs()
		for i := len(req.History) - 1; i >= 0; i-- {
			if req.History[i].Role == "user" {
				req.Registry.SeedWebSession(req.History[i].Content)
				break
			}
		}
	}

	// Build prompt-cache-stable prefix cleanly: system prompt first, followed by history.
	convo := make([]runner.APIMessage, 0, len(req.History)+8)
	sysPrompt := strings.TrimSpace(req.SystemPrompt)
	if req.Registry != nil && req.Registry.GroupEnabled("web") {
		if sysPrompt != "" {
			sysPrompt += "\n\n"
		}
		sysPrompt += web.UntrustedSystemDirective
	}
	if sysPrompt != "" {
		convo = append(convo, runner.APIMessage{Role: "system", Content: sysPrompt})
	}
	convo = append(convo, req.History...)

	var appended []runner.APIMessage
	identicalCounts := make(map[string]int)
	totalToolBytes := 0

	var res LoopResult
	res.Tainted = req.Tainted
	emit := func(ev LoopEvent) {
		if req.OnEvent != nil {
			req.OnEvent(ev)
		}
	}

	for step := 1; step <= maxSteps; step++ {
		if ctx.Err() != nil {
			res.Interrupted = true
			return res
		}
		res.Steps = step

		cacheOn := true
		temp := req.Temperature
		if step > 1 && req.ToolTemperature > 0 {
			temp = req.ToolTemperature
		}
		chatReq := runner.ChatRequest{
			Messages:           convo,
			Tools:              toolDefs,
			CachePrompt:        &cacheOn,
			MaxTokens:          req.MaxTokens,
			Temperature:        temp,
			TopP:               req.TopP,
			TopK:               req.TopK,
			MinP:               req.MinP,
			ReasoningEffort:    req.ReasoningEffort,
			ChatTemplateKwargs: req.ChatTemplateKwargs,
		}
		if len(toolDefs) > 0 {
			chatReq.ToolChoice = "auto"
		}

		var stepText strings.Builder
		var acc ToolCallAccumulator
		streamErr := runner.StreamChatURL(ctx, req.BaseURL, chatReq, func(d runner.ChatDelta) {
			if d.Content != "" {
				stepText.WriteString(d.Content)
			}
			if len(d.ToolCalls) > 0 {
				acc.Feed(d.ToolCalls)
			}
			if d.PromptTokens > 0 {
				res.PromptTokens = d.PromptTokens
			}
			if d.CachedPromptTokens > 0 {
				res.CachedTokens = d.CachedPromptTokens
			}
			if d.CompletionTokens > 0 {
				res.CompletionTokens = d.CompletionTokens
			}
			if d.GenTokPerSec > 0 {
				res.GenTokPerSec = d.GenTokPerSec
			}
			if d.PromptTokSec > 0 {
				res.PromptTokSec = d.PromptTokSec
			}
			emit(LoopEvent{Kind: EvtDelta, Delta: d})
		})

		if ctx.Err() != nil {
			res.Interrupted = true
			res.FinalContent = stepText.String()
			return res
		}
		if streamErr != nil {
			res.Err = streamErr
			return res
		}

		calls := acc.Results()
		textOut := stepText.String()
		if len(calls) == 0 && strings.TrimSpace(textOut) != "" {
			if fbCalls, cleaned := ExtractFallbackTextToolCalls(textOut); len(fbCalls) > 0 {
				calls = fbCalls
				textOut = cleaned
			}
		}

		if len(calls) == 0 {
			if withSources, suffix := appendRetrievedSources(textOut, req.Registry); suffix != "" {
				textOut = withSources
				if req.OnEvent != nil {
					emit(LoopEvent{Kind: EvtDelta, Delta: runner.ChatDelta{Content: suffix}})
				}
			}
			res.FinalContent = textOut
			if strings.TrimSpace(textOut) != "" {
				appended = append(appended, runner.APIMessage{Role: "assistant", Content: textOut})
			}
			res.MessagesAppended = appended
			return res
		}

		assistantMsg := runner.APIMessage{
			Role:      "assistant",
			Content:   textOut,
			ToolCalls: calls,
		}
		convo = append(convo, assistantMsg)
		appended = append(appended, assistantMsg)

		for _, tc := range calls {
			startT := time.Now()
			callReq := req
			callReq.ToolCtx.Tainted = req.Tainted
			toolMsg, execRes, interrupted := executeSingleCall(ctx, callReq, tc, identicalCounts, &totalToolBytes, emit, startT)
			if execRes.Tainted {
				req.Tainted = true
				res.Tainted = true
			}
			convo = append(convo, toolMsg)
			appended = append(appended, toolMsg)
			emit(LoopEvent{
				Kind:       EvtToolEnd,
				CallID:     tc.ID,
				Tool:       tc.Function.Name,
				Summary:    execRes.Summary,
				ArgsPretty: prettyArgs(tc.Function.Arguments),
				Result:     execRes,
				Duration:   time.Since(startT),
			})
			if interrupted {
				res.Interrupted = true
				res.MessagesAppended = appended
				return res
			}
		}
	}

	res.Err = fmt.Errorf("agent stopped after reaching max steps (%d) per turn", maxSteps)
	res.MessagesAppended = appended
	return res
}

func executeSingleCall(
	ctx context.Context,
	req LoopRequest,
	tc runner.ToolCall,
	identicalCounts map[string]int,
	totalToolBytes *int,
	emit func(LoopEvent),
	startT time.Time,
) (runner.APIMessage, ToolExecResult, bool) {
	toolName := strings.TrimSpace(tc.Function.Name)
	pretty := prettyArgs(tc.Function.Arguments)

	emit(LoopEvent{
		Kind:       EvtToolStart,
		CallID:     tc.ID,
		Tool:       toolName,
		ArgsPretty: pretty,
	})

	if ctx.Err() != nil {
		ir := ToolExecResult{Output: "Tool execution was interrupted by user.", Summary: "interrupted", IsError: true, Interrupted: true}
		return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: ir.Output}, ir, true
	}

	if req.Registry == nil {
		er := ToolExecResult{Output: "Error: no tool registry configured.", Summary: "no registry", IsError: true}
		return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
	}

	spec, ok := req.Registry.Lookup(toolName)
	if !ok {
		er := ToolExecResult{Output: fmt.Sprintf("Error: unknown or disabled tool %q.", toolName), Summary: "unknown tool", IsError: true}
		return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
	}

	args, _, parseErr := DecodeToolArgs(tc.Function.Arguments)
	if parseErr != nil {
		er := ToolExecResult{Output: fmt.Sprintf("Error: %v. Please fix your JSON arguments and retry.", parseErr), Summary: "bad JSON", IsError: true}
		return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
	}

	repeatKey := SignatureKey(toolName, args)
	identicalCounts[repeatKey]++
	if identicalCounts[repeatKey] > DefaultMaxIdentical {
		er := ToolExecResult{
			Output:  fmt.Sprintf("Error: repeated identical call to %s (%d times) blocked to avoid an infinite loop.", toolName, identicalCounts[repeatKey]),
			Summary: "loop guard",
			IsError: true,
		}
		return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
	}

	var permIn PermissionInput
	var diffText, diffPath string
	var diffAdded, diffRemoved int
	if spec.InspectPerm != nil {
		pIn, df, dp, da, dr, err := spec.InspectPerm(req.ToolCtx, args)
		if err != nil {
			er := errRes(err)
			return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
		}
		permIn, diffText, diffPath, diffAdded, diffRemoved = pIn, df, dp, da, dr
	} else {
		permIn = PermissionInput{
			Tool:          toolName,
			CanonicalArgs: canonicalizeMap(args),
			Class:         spec.Class,
			InWorkspace:   true,
			Summary:       toolName,
		}
	}
	permIn.Tainted = req.Tainted

	if req.Permissions != nil {
		dec, reason := req.Permissions.Evaluate(permIn)
		switch dec {
		case DecisionDeny:
			er := ToolExecResult{Output: "Denied: " + reason, Summary: "denied", IsError: true}
			return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
		case DecisionAsk:
			if req.AskPermission == nil {
				er := ToolExecResult{Output: "Denied: interactive approval required (" + reason + ")", Summary: "denied", IsError: true}
				return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
			}
			resp, askErr := req.AskPermission(ctx, PermissionPrompt{
				CallID:      tc.ID,
				Tool:        toolName,
				Summary:     permIn.Summary,
				Reason:      reason,
				Class:       spec.Class,
				HighRisk:    permIn.HighRisk,
				ArgsPretty:  pretty,
				Diff:        diffText,
				DiffPath:    diffPath,
				DiffAdded:   diffAdded,
				DiffRemoved: diffRemoved,
				RawArgs:     permIn.CanonicalArgs,
			})
			if askErr != nil || ctx.Err() != nil {
				ir := ToolExecResult{Output: "Tool execution was interrupted by user.", Summary: "interrupted", IsError: true, Interrupted: true}
				return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: ir.Output}, ir, true
			}
			switch resp.Choice {
			case PermAllowSession:
				req.Permissions.AllowSession(toolName, permIn.CanonicalArgs, permIn.Summary)
			case PermDeny:
				er := ToolExecResult{Output: "Denied by user.", Summary: "denied", IsError: true}
				return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
			case PermDenyWithNote:
				note := strings.TrimSpace(resp.Note)
				if note == "" {
					note = "no reason given"
				}
				er := ToolExecResult{Output: "Denied by user: " + note, Summary: "denied with note", IsError: true}
				return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: er.Output}, er, false
			}
		}
	}

	execRes := spec.Execute(ctx, req.ToolCtx, args)
	if ctx.Err() != nil {
		execRes.Interrupted = true
		if !strings.Contains(execRes.Output, "interrupted") {
			execRes.Output = strings.TrimSpace(execRes.Output + "\nTool execution was interrupted by user.")
		}
	}
	execRes.Output = clampToolOutput(execRes.Output)
	if *totalToolBytes+len(execRes.Output) > DefaultMaxToolBytesOut {
		rem := DefaultMaxToolBytesOut - *totalToolBytes
		if rem < 256 {
			rem = 256
		}
		if len(execRes.Output) > rem {
			execRes.Output = execRes.Output[:rem] + "\n... [turn output cap reached]"
		}
	}
	*totalToolBytes += len(execRes.Output)
	_ = startT
	return runner.APIMessage{Role: "tool", ToolCallID: tc.ID, Name: toolName, Content: execRes.Output}, execRes, execRes.Interrupted
}

func prettyArgs(raw string) string {
	var v any
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &v); err == nil {
		if b, err := json.MarshalIndent(v, "", "  "); err == nil {
			return string(b)
		}
	}
	return raw
}
