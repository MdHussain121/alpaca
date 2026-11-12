package session

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestLoadAgentsMDGlobalWorkspaceDedupAndLivePickup(t *testing.T) {
	appRoot := t.TempDir()
	ws := filepath.Join(appRoot, "workspace")
	if err := os.MkdirAll(ws, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(appRoot, "AGENTS.md"), []byte("Global Rule: Never skip tests."), 0o644); err != nil {
		t.Fatal(err)
	}
	wsAgents := filepath.Join(ws, "AGENTS.md")
	if err := os.WriteFile(wsAgents, []byte("Workspace Rule: Use tabs in Go."), 0o644); err != nil {
		t.Fatal(err)
	}

	res1 := LoadAgentsMD(appRoot, ws)
	if !strings.Contains(res1.CombinedPrompt, "Global Rule") || !strings.Contains(res1.CombinedPrompt, "Workspace Rule") {
		t.Fatalf("expected both global and workspace AGENTS.md, got:\n%s", res1.CombinedPrompt)
	}
	if res1.TokensEst <= 0 {
		t.Fatalf("expected >0 TokensEst, got %d", res1.TokensEst)
	}

	// Same root must deduplicate rather than loading AGENTS.md twice.
	dedup := LoadAgentsMD(appRoot, appRoot)
	if strings.Count(dedup.CombinedPrompt, "Global Rule") != 1 {
		t.Fatalf("expected AGENTS.md deduplicated when appRoot == workDir:\n%s", dedup.CombinedPrompt)
	}

	// Editing workspace AGENTS.md is picked up immediately on next call.
	if err := os.WriteFile(wsAgents, []byte("Workspace Rule Updated: Also run go vet."), 0o644); err != nil {
		t.Fatal(err)
	}
	res2 := LoadAgentsMD(appRoot, ws)
	if !strings.Contains(res2.CombinedPrompt, "go vet") {
		t.Fatalf("expected updated AGENTS.md on next turn, got:\n%s", res2.CombinedPrompt)
	}
}

func TestSessionJSONLRoundTripBranchingSearchAndExport(t *testing.T) {
	sessDir := t.TempDir()
	doc := NewSessionDoc("sess-alpha", filepath.Join(sessDir, "ws"), "Qwen3.5-4B", "default", "coding")
	doc.ReplaceActiveTurns([]BranchTurn{
		{Role: "user", Content: "Implement parser in parser.go", APIMessages: []runner.APIMessage{{Role: "user", Content: "Implement parser in parser.go"}}},
		{Role: "assistant", Reasoning: "Think steps", Content: "Created parser.go cleanly.", Tokens: 42, ToolCalls: []ToolCallRecord{{Name: "write_file", Status: "ok", Summary: "parser.go"}}},
		{Role: "user", Content: "Old second prompt"},
		{Role: "assistant", Content: "Old second reply"},
	})
	if err := SaveSession(sessDir, doc); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	// Fork at turn index 2 (edit 2nd user prompt)
	brID, prefix := doc.ForkBeforeActiveIndex(2, "edit-second")
	if brID != "b2" || len(prefix) != 2 {
		t.Fatalf("expected branch b2 with 2 prefix turns, got %s len=%d", brID, len(prefix))
	}
	newActive := append(prefix,
		BranchTurn{Role: "user", Content: "Edited prompt about zeta keyword"},
		BranchTurn{Role: "assistant", Content: "New branched response!"},
	)
	doc.ReplaceActiveTurns(newActive)
	if err := SaveSession(sessDir, doc); err != nil {
		t.Fatalf("SaveSession after fork: %v", err)
	}

	// Load back and verify both branches exist + active branch is b2
	loaded, err := LoadSessionByID(sessDir, "sess-alpha")
	if err != nil {
		t.Fatalf("LoadSessionByID: %v", err)
	}
	if loaded.Meta.ActiveBranch != "b2" || len(loaded.Meta.Branches) != 2 {
		t.Fatalf("unexpected branch state after reload: active=%s branches=%d", loaded.Meta.ActiveBranch, len(loaded.Meta.Branches))
	}
	act := loaded.ActiveTurns()
	if len(act) != 4 || act[2].Content != "Edited prompt about zeta keyword" {
		t.Fatalf("unexpected active turns on b2: %+v", act)
	}

	// Switch back to b1 and confirm original reply preserved
	if err := loaded.SwitchBranch("b1"); err != nil {
		t.Fatal(err)
	}
	b1Turns := loaded.ActiveTurns()
	if len(b1Turns) != 4 || b1Turns[3].Content != "Old second reply" {
		t.Fatalf("expected original b1 branch intact, got %+v", b1Turns)
	}

	// Search by content + Rename + Export Markdown + Delete
	hits, err := SearchSessions(sessDir, "zeta keyword")
	if err != nil || len(hits) != 1 || hits[0].ID != "sess-alpha" {
		t.Fatalf("SearchSessions content match failed: err=%v hits=%+v", err, hits)
	}
	if err := RenameSession(sessDir, "sess-alpha", "Parser Feature"); err != nil {
		t.Fatal(err)
	}
	hitsTitle, _ := SearchSessions(sessDir, "Parser Feature")
	if len(hitsTitle) != 1 || hitsTitle[0].Title != "Parser Feature" {
		t.Fatalf("SearchSessions title match failed: %+v", hitsTitle)
	}
	md := ExportMarkdown(loaded)
	if !strings.Contains(md, "Created parser.go cleanly.") || !strings.Contains(md, "Tool `write_file`") {
		t.Fatalf("unexpected exported markdown:\n%s", md)
	}
	if err := DeleteSession(sessDir, "sess-alpha"); err != nil {
		t.Fatal(err)
	}
	afterDel, _ := ListSessions(sessDir)
	if len(afterDel) != 0 {
		t.Fatalf("expected 0 sessions after delete, got %d", len(afterDel))
	}
}

func TestMeasureBreakdownCachingAndTieredCompaction(t *testing.T) {
	var inputTokenProbes atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions/input_tokens":
			inputTokenProbes.Add(1)
			var req runner.ChatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			b, _ := json.Marshal(req)
			_ = json.NewEncoder(w).Encode(map[string]int{"input_tokens": len(b)/4 + 10})
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"- Goal: Keep CANARY-777 intact.\\n- Modified auth.go.\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer srv.Close()

	cache := NewBreakdownCache()
	tools := []runner.ToolDef{{Type: "function", Function: runner.ToolFunctionDef{Name: "read_file", Description: "Read file"}}}
	msgs := []runner.APIMessage{{Role: "user", Content: "hi"}}

	bd1 := MeasureBreakdown(context.Background(), srv.URL, cache, "Base system", "<project_instructions>AGENTS</project_instructions>", "<available_skills>- a: b</available_skills>", tools, msgs, 12, 8192, 50)
	if !bd1.LiveMeasured || bd1.TotalPromptTokens <= 0 || bd1.AgentsMDTokens <= 0 || bd1.SkillListTokens <= 0 {
		t.Fatalf("unexpected live MeasureBreakdown result: %+v", bd1)
	}
	probes1 := inputTokenProbes.Load()
	_ = MeasureBreakdown(context.Background(), srv.URL, cache, "Base system", "<project_instructions>AGENTS</project_instructions>", "<available_skills>- a: b</available_skills>", tools, msgs, 12, 8192, 50)
	if inputTokenProbes.Load() != probes1 {
		t.Fatalf("expected BreakdownCache to avoid repeat HTTP probes (was %d, now %d)", probes1, inputTokenProbes.Load())
	}

	// Long history for tiered compaction
	var longMsgs []runner.APIMessage
	for i := 0; i < 8; i++ {
		longMsgs = append(longMsgs,
			runner.APIMessage{Role: "user", Content: fmt.Sprintf("Step %d prompt with context CANARY-777.", i+1)},
			runner.APIMessage{Role: "assistant", Content: "Calling read_file.", ToolCalls: []runner.ToolCall{{ID: fmt.Sprintf("c%d", i)}}},
			runner.APIMessage{Role: "tool", ToolCallID: fmt.Sprintf("c%d", i), Name: "read_file", Content: strings.Repeat("verbose tool output line\n", 40)},
			runner.APIMessage{Role: "assistant", Content: fmt.Sprintf("Completed step %d.", i+1)},
		)
	}

	comp := CompactHistory(context.Background(), srv.URL, longMsgs, 2, false, 300)
	if !comp.Applied || comp.Tier1Stubbed == 0 || !comp.Tier2Summarized {
		t.Fatalf("expected both Tier1 stubbing and Tier2 summarization, got %+v", comp)
	}
	if comp.AfterTokensEst >= comp.BeforeTokensEst {
		t.Fatalf("expected compaction token reduction: before=%d after=%d", comp.BeforeTokensEst, comp.AfterTokensEst)
	}
	if !strings.Contains(comp.SummaryText, "CANARY-777") || !strings.Contains(comp.CardTitle, "prompt-cache prefix restarted") {
		t.Fatalf("unexpected compaction summary/card: title=%q summary=%q", comp.CardTitle, comp.SummaryText)
	}
	// Last 2 user turns (Steps 7 and 8) must remain verbatim at tail!
	joinedTail := fmt.Sprintf("%+v", comp.CompactedMsgs)
	if !strings.Contains(joinedTail, "Step 7 prompt") || !strings.Contains(joinedTail, "Step 8 prompt") {
		t.Fatalf("expected last 2 user turns preserved verbatim after compaction")
	}
}
