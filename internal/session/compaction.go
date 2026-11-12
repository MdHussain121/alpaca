package session

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/MdHussain121/alpaca/internal/runner"
)

// ContextBreakdown holds per-segment prompt token counts for /context and meter UI.
type ContextBreakdown struct {
	SystemTokens      int
	AgentsMDTokens    int
	SkillListTokens   int
	ToolDefsTokens    int
	AttachmentsTokens int
	HistoryTokens     int
	TotalPromptTokens int
	CachedTokens      int
	ContextLimit      int
	LiveMeasured      bool
}

// BreakdownCache caches /v1/chat/completions/input_tokens probes keyed by prompt prefix hashes
// so opening /context does not re-probe unchanged payloads.
type BreakdownCache struct {
	mu      sync.Mutex
	entries map[string]int
}

// NewBreakdownCache creates a bounded prompt-token measurement cache.
func NewBreakdownCache() *BreakdownCache {
	return &BreakdownCache{entries: make(map[string]int)}
}

func (c *BreakdownCache) get(key string) (int, bool) {
	if c == nil {
		return 0, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.entries[key]
	return v, ok
}

func (c *BreakdownCache) set(key string, val int) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.entries) > 64 {
		c.entries = make(map[string]int)
	}
	c.entries[key] = val
}

func EstimateTokensText(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}

func hashPayload(parts ...any) string {
	h := sha256.New()
	for _, p := range parts {
		b, _ := json.Marshal(p)
		h.Write(b)
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func countAPIMessagesTokensHeur(msgs []runner.APIMessage) int {
	total := 0
	for _, m := range msgs {
		total += 4 + EstimateTokensText(m.Role) + EstimateTokensText(m.Content)
		for _, tc := range m.ToolCalls {
			total += EstimateTokensText(tc.Function.Name) + EstimateTokensText(tc.Function.Arguments) + 8
		}
	}
	return total
}

// MeasureBreakdown computes per-category token usage, probing the server's /input_tokens
// endpoint when live and caching prefix measurements cleanly.
func MeasureBreakdown(
	ctx context.Context,
	baseURL string,
	cache *BreakdownCache,
	baseSys string,
	agentsBlock string,
	skillBlock string,
	tools []runner.ToolDef,
	apiMsgs []runner.APIMessage,
	attachmentTokens int,
	ctxLimit int,
	cachedTokens int,
) ContextBreakdown {
	bd := ContextBreakdown{
		SystemTokens:      EstimateTokensText(baseSys),
		AgentsMDTokens:    EstimateTokensText(agentsBlock),
		SkillListTokens:   EstimateTokensText(skillBlock),
		AttachmentsTokens: attachmentTokens,
		HistoryTokens:     countAPIMessagesTokensHeur(apiMsgs),
		ContextLimit:      ctxLimit,
		CachedTokens:      cachedTokens,
	}
	if len(tools) > 0 {
		tb, _ := json.Marshal(tools)
		bd.ToolDefsTokens = (len(tb) + 3) / 4
	}
	if bd.AttachmentsTokens > 0 && bd.HistoryTokens >= bd.AttachmentsTokens {
		bd.HistoryTokens -= bd.AttachmentsTokens
	} else if bd.AttachmentsTokens > bd.HistoryTokens {
		bd.HistoryTokens = 0
	}
	bd.TotalPromptTokens = bd.SystemTokens + bd.AgentsMDTokens + bd.SkillListTokens + bd.ToolDefsTokens + bd.AttachmentsTokens + bd.HistoryTokens

	if strings.TrimSpace(baseURL) == "" {
		return bd
	}

	probe := func(sysText string, tDefs []runner.ToolDef, msgs []runner.APIMessage) (int, bool) {
		key := hashPayload(baseURL, sysText, tDefs, msgs)
		if v, ok := cache.get(key); ok {
			return v, true
		}
		var reqMsgs []runner.APIMessage
		if sysText != "" {
			reqMsgs = append(reqMsgs, runner.APIMessage{Role: "system", Content: sysText})
		}
		reqMsgs = append(reqMsgs, msgs...)
		if len(reqMsgs) == 0 {
			reqMsgs = []runner.APIMessage{{Role: "user", Content: ""}}
		}
		n, err := runner.InputTokensURL(ctx, baseURL, runner.ChatRequest{Messages: reqMsgs, Tools: tDefs})
		if err != nil || n <= 0 {
			return 0, false
		}
		cache.set(key, n)
		return n, true
	}

	fullSys := joinNonEmpty(baseSys, agentsBlock, skillBlock)
	totalLive, okTotal := probe(fullSys, tools, apiMsgs)
	if !okTotal {
		return bd
	}
	bd.LiveMeasured = true

	// Measure incremental layers in deterministic prefix order.
	baseOnly := 0
	if baseSys != "" {
		if n, ok := probe(baseSys, nil, nil); ok {
			baseOnly = n
			bd.SystemTokens = n
		}
	}
	sysPlusAgents := baseOnly
	if agentsBlock != "" {
		if n, ok := probe(joinNonEmpty(baseSys, agentsBlock), nil, nil); ok {
			sysPlusAgents = n
			if n >= baseOnly {
				bd.AgentsMDTokens = n - baseOnly
			}
		}
	}
	sysAll := sysPlusAgents
	if skillBlock != "" {
		if n, ok := probe(fullSys, nil, nil); ok {
			sysAll = n
			if n >= sysPlusAgents {
				bd.SkillListTokens = n - sysPlusAgents
			}
		}
	}
	sysPlusTools := sysAll
	if len(tools) > 0 {
		if n, ok := probe(fullSys, tools, nil); ok {
			sysPlusTools = n
			if n >= sysAll {
				bd.ToolDefsTokens = n - sysAll
			}
		}
	}

	histTotal := totalLive - sysPlusTools
	if histTotal < 0 {
		histTotal = 0
	}
	if bd.AttachmentsTokens > histTotal {
		bd.HistoryTokens = 0
	} else {
		bd.HistoryTokens = histTotal - bd.AttachmentsTokens
	}
	bd.TotalPromptTokens = totalLive
	return bd
}

func joinNonEmpty(parts ...string) string {
	var nonEmpty []string
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			nonEmpty = append(nonEmpty, s)
		}
	}
	return strings.Join(nonEmpty, "\n\n")
}

// CompactResult reports the result of tiered conversation compaction.
type CompactResult struct {
	Applied         bool
	Tier1Stubbed    int
	Tier2Summarized bool
	SummaryText     string
	CardTitle       string
	CardBody        string
	BeforeTokensEst int
	AfterTokensEst  int
	DroppedMsgCount int
	CompactedMsgs   []runner.APIMessage
}

// CompactHistory executes tiered compaction:
//  1. Replace old tool outputs (>180B before the last keepRecentTurns user turns) with stubs.
//  2. Summarize older turns before the kept tail via the active model (with extractive fallback)
//     preserving the goal, modified files/paths, key decisions, and open tasks.
func CompactHistory(
	ctx context.Context,
	baseURL string,
	apiMsgs []runner.APIMessage,
	keepRecentTurns int,
	forceSummarize bool,
	targetMaxTokens int,
) CompactResult {
	res := CompactResult{}
	if len(apiMsgs) == 0 {
		return res
	}
	if keepRecentTurns <= 0 {
		keepRecentTurns = 4
	}
	res.BeforeTokensEst = countAPIMessagesTokensHeur(apiMsgs)

	// Locate cut boundary keeping last keepRecentTurns user turns intact.
	cutIdx := findKeepTailIndex(apiMsgs, keepRecentTurns)
	if cutIdx <= 1 && !forceSummarize {
		return res
	}

	work := make([]runner.APIMessage, len(apiMsgs))
	copy(work, apiMsgs)

	// Tier 1: stub verbose tool outputs before cutIdx.
	for i := 0; i < cutIdx; i++ {
		if work[i].Role == "tool" && len(work[i].Content) > 180 && !strings.HasPrefix(work[i].Content, "[output omitted, ") {
			origLen := len(work[i].Content)
			work[i].Content = fmt.Sprintf("[output omitted, %d bytes]", origLen)
			res.Tier1Stubbed++
		}
	}

	afterTier1 := countAPIMessagesTokensHeur(work)
	needTier2 := forceSummarize || (targetMaxTokens > 0 && afterTier1 > targetMaxTokens) || res.Tier1Stubbed == 0
	if !needTier2 || cutIdx <= 1 {
		if res.Tier1Stubbed > 0 {
			res.Applied = true
			res.AfterTokensEst = afterTier1
			res.CompactedMsgs = work
			res.CardTitle = fmt.Sprintf("Compacted %d older tool outputs (%d -> %d tok)", res.Tier1Stubbed, res.BeforeTokensEst, res.AfterTokensEst)
			res.CardBody = "Replaced older tool results with compact stubs while preserving recent turns."
		}
		return res
	}

	// Tier 2: summarize apiMsgs[:cutIdx] and keep work[cutIdx:] verbatim.
	older := work[:cutIdx]
	tail := work[cutIdx:]
	summary := summarizeOlderTurns(ctx, baseURL, older)
	summaryMsg := runner.APIMessage{
		Role:    "user",
		Content: "[Conversation summary — older turns compacted; prompt cache prefix restarted]\n" + summary,
	}
	compacted := make([]runner.APIMessage, 0, 1+len(tail))
	compacted = append(compacted, summaryMsg)
	compacted = append(compacted, tail...)

	res.Applied = true
	res.Tier2Summarized = true
	res.SummaryText = summary
	res.DroppedMsgCount = len(older)
	res.CompactedMsgs = compacted
	res.AfterTokensEst = countAPIMessagesTokensHeur(compacted)
	res.CardTitle = fmt.Sprintf("Compacted %d older messages (%d -> %d tok) · prompt-cache prefix restarted",
		res.DroppedMsgCount, res.BeforeTokensEst, res.AfterTokensEst)
	res.CardBody = summary
	return res
}

func findKeepTailIndex(msgs []runner.APIMessage, keepUsers int) int {
	seenUsers := 0
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" && !strings.HasPrefix(msgs[i].Content, "[Conversation summary") {
			seenUsers++
			if seenUsers >= keepUsers {
				return i
			}
		}
	}
	return 0
}

func summarizeOlderTurns(ctx context.Context, baseURL string, older []runner.APIMessage) string {
	var transcript strings.Builder
	for _, m := range older {
		content := strings.TrimSpace(m.Content)
		if len(content) > 600 {
			content = content[:600] + "…"
		}
		if content == "" && len(m.ToolCalls) > 0 {
			var names []string
			for _, tc := range m.ToolCalls {
				names = append(names, tc.Function.Name)
			}
			content = "[called tools: " + strings.Join(names, ", ") + "]"
		}
		transcript.WriteString(fmt.Sprintf("%s: %s\n", strings.ToUpper(m.Role), content))
		if transcript.Len() > 14000 {
			break
		}
	}

	if strings.TrimSpace(baseURL) != "" {
		sCtx, cancel := context.WithTimeout(ctx, 25*time.Second)
		defer cancel()
		var b strings.Builder
		err := runner.StreamChatURL(sCtx, baseURL, runner.ChatRequest{
			Messages: []runner.APIMessage{
				{
					Role:    "system",
					Content: "Summarize the conversation transcript concisely in bullet points preserving: (1) User goal and constraints, (2) Exact file paths/commands/facts established, (3) Decisions made and pending next steps. Keep under 250 words.",
				},
				{
					Role:    "user",
					Content: transcript.String(),
				},
			},
			MaxTokens:   420,
			Temperature: 0.1,
			Stream:      true,
		}, func(d runner.ChatDelta) {
			b.WriteString(d.Content)
		})
		if err == nil && strings.TrimSpace(b.String()) != "" {
			return strings.TrimSpace(b.String())
		}
	}

	// Deterministic fallback summary when offline or model request unavailable.
	var fallback strings.Builder
	fallback.WriteString("Compacted conversation summary:\n")
	count := 0
	for _, m := range older {
		if m.Role == "user" || m.Role == "assistant" {
			s := strings.Join(strings.Fields(strings.TrimSpace(m.Content)), " ")
			if s == "" {
				continue
			}
			if len(s) > 140 {
				s = s[:140] + "…"
			}
			fallback.WriteString(fmt.Sprintf("- %s: %s\n", m.Role, s))
			count++
			if count >= 10 {
				break
			}
		}
	}
	return strings.TrimSpace(fallback.String())
}
