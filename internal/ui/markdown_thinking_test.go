package ui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/runner"
)

func renderChatAll(m *Model) string {
	innerW := m.chatView.Width
	if innerW < 20 {
		innerW = 60
	}
	var out []string
	for i := range m.chatMsgs {
		out = append(out, m.renderChatTurn(&m.chatMsgs[i], innerW, i == len(m.chatMsgs)-1)...)
	}
	return stripANSI(strings.Join(out, "\n"))
}

func TestThinkingStreamPreviewCollapseToggleAndNeverResent(t *testing.T) {
	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 32
	m.resizeRunViews()

	var captured []runner.ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/props":
			_, _ = w.Write([]byte(`{"model_alias":"qwen3.5-4b","chat_template":"{% if enable_thinking %}<think>{% endif %}","default_generation_settings":{"n_ctx":8192}}`))
		case "/v1/chat/completions":
			var req runner.ChatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			captured = append(captured, req)
			w.Header().Set("Content-Type", "text/event-stream")
			switch len(captured) {
			case 1:
				_, _ = w.Write([]byte(
					"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step 1 examine input\\nstep 2 check edge cases\\nstep 3 consider complexity\\nstep 4 draft answer\"}}]}\n\n" +
						"data: {\"choices\":[{\"delta\":{\"content\":\"Here is the solution.\"},\"finish_reason\":\"stop\"}],\"timings\":{\"predicted_per_second\":42.0,\"predicted_n\":5,\"prompt_n\":12}}\n\n" +
						"data: [DONE]\n\n",
				))
			default:
				_, _ = w.Write([]byte(
					"data: {\"choices\":[{\"delta\":{\"content\":\"Follow-up OK.\"},\"finish_reason\":\"stop\"}],\"timings\":{\"predicted_per_second\":45.0,\"predicted_n\":3,\"prompt_n\":20}}\n\n" +
						"data: [DONE]\n\n",
				))
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	m.chatBaseURL = srv.URL

	// Query /props like a live launch does.
	if pm, ok := fetchPropsCmd(m.activePort, srv.URL)().(propsMsg); ok && pm.err == nil {
		m.modelCaps = pm.props
	} else {
		t.Fatalf("fetchPropsCmd failed: %#v", pm)
	}

	// Simulate live streaming delta with reasoning BEFORE answer arrives:
	m.chatInput.Focus()
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "first question"},
		{Role: "assistant", ThinkStart: time.Now().Add(-1500 * time.Millisecond)},
	}
	m.chatStreaming = true
	m.handleChatDelta(runner.ChatDelta{
		Reasoning: "line one\nline two\nline three\nline four\nline five",
	})
	live := renderChatAll(m)
	if !strings.Contains(live, "Thinking ") {
		t.Fatalf("expected active Thinking header during reasoning stream, got:\n%s", live)
	}
	// Live preview must show trailing 3 lines (not line one / line two).
	if strings.Contains(live, "line one") || !strings.Contains(live, "line three") || !strings.Contains(live, "line five") {
		t.Fatalf("expected sliding 3-line reasoning preview during streaming, got:\n%s", live)
	}

	// Answer delta closes thinking and auto-collapses block.
	m.handleChatDelta(runner.ChatDelta{Content: "Final concise answer."})
	m.finishChatStream(nil)
	collapsed := renderChatAll(m)
	if !strings.Contains(collapsed, "Thought for ") || strings.Contains(collapsed, "line five") {
		t.Fatalf("expected auto-collapsed 'Thought for ...' hiding reasoning body, got:\n%s", collapsed)
	}

	// Ctrl+T expands the collapsed reasoning trace, second Ctrl+T collapses it again.
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	expanded := renderChatAll(m)
	if !strings.Contains(expanded, "line one") || !strings.Contains(expanded, "line five") {
		t.Fatalf("ctrl+t should expand reasoning body, got:\n%s", expanded)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	recollapsed := renderChatAll(m)
	if strings.Contains(recollapsed, "line five") {
		t.Fatalf("second ctrl+t should collapse reasoning body, got:\n%s", recollapsed)
	}

	// Turn 2 must NEVER re-send turn 1's Reasoning in APIMessages.
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "turn 1 prompt"},
		{Role: "assistant", Reasoning: "SECRET_REASONING_NEVER_RESEND", Content: "Turn 1 answer.", ThinkDone: true},
	}
	_, cmd := m.startChatCompletion("turn 2 follow-up")
	for cmd != nil {
		msg := cmd()
		_, cmd = m.Update(msg)
	}
	if len(captured) != 1 {
		t.Fatalf("expected 1 HTTP chat request, got %d", len(captured))
	}
	rawPayload, _ := json.Marshal(captured[0].Messages)
	if strings.Contains(string(rawPayload), "SECRET_REASONING_NEVER_RESEND") {
		t.Fatalf("reasoning leaked back into subsequent API request messages: %s", string(rawPayload))
	}
	if !strings.Contains(string(rawPayload), "Turn 1 answer.") {
		t.Fatalf("expected assistant content in history, got: %s", string(rawPayload))
	}
}

func TestThinkCommandCapabilitiesHiddenForLFMAndShownForQwen(t *testing.T) {
	// LFM2.5-8B props: no enable_thinking in chat_template.
	lfmProps, err := runner.ParsePropsJSON([]byte(`{
		"model_alias": "lfm2.5-8b",
		"chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{% endfor %}",
		"default_generation_settings": {"n_ctx": 8192}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 120, 32
	m.resizeRunViews()
	m.modelCaps = lfmProps

	if m.thinkToggleSupported() {
		t.Fatalf("expected thinkToggleSupported=false for LFM2.5 template")
	}
	m.chatInput.SetValue("/help")
	m.submitChatInput()
	helpOut := renderChatAll(m)
	if strings.Contains(helpOut, "/think [on|off]") {
		t.Fatalf("/help must hide /think [on|off] when model lacks enable_thinking:\n%s", helpOut)
	}

	m.chatInput.SetValue("/think off")
	m.submitChatInput()
	infoOut := renderChatAll(m)
	if !strings.Contains(infoOut, "not supported") {
		t.Fatalf("/think off on unsupported model must explain incompatibility:\n%s", infoOut)
	}

	// Qwen3.5-4B props: supports enable_thinking.
	qwenProps, err := runner.ParsePropsJSON([]byte(`{
		"model_alias": "qwen3.5-4b",
		"chat_template": "{%- if enable_thinking is defined %}<think>{% endif %}",
		"default_generation_settings": {"n_ctx": 8192}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	m.resetChat()
	m.modelCaps = qwenProps
	m.chatInput.SetValue("/help")
	m.submitChatInput()
	qwenHelp := renderChatAll(m)
	if !strings.Contains(qwenHelp, "/think [on|off]") {
		t.Fatalf("/help should advertise /think [on|off] when supported:\n%s", qwenHelp)
	}

	// Captured request with /think off includes enable_thinking=false only on capable model.
	var gotReq runner.ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotReq)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()
	m.chatBaseURL = srv.URL
	m.chatInput.SetValue("/think off")
	m.submitChatInput()
	_, cmd := m.startChatCompletion("hi")
	for cmd != nil {
		msg := cmd()
		_, cmd = m.Update(msg)
	}
	if gotReq.ChatTemplateKwargs["enable_thinking"] != false {
		t.Fatalf("expected enable_thinking=false when /think off on capable model, got %+v", gotReq.ChatTemplateKwargs)
	}
}

func TestMarkdownCodeBlockBordersTabsLineNumsCopyAndThemeHighlighting(t *testing.T) {
	origClip := clipboardWriter
	defer func() { clipboardWriter = origClip }()
	var copied string
	clipboardWriter = func(s string) error {
		copied = s
		return nil
	}

	m := testUI(t)
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.width, m.height = 80, 28
	m.resizeRunViews()

	// Ensure truecolor Chroma ANSI styling emits SGR color sequences when not ASCII.
	ansiSty := newStyles(Mocha, false)
	ansiSty.true = true
	goSample := "package main\n\nfunc main() {\n\treturn // check very_long_identifier_name_that_must_wrap_cleanly_inside_box_borders_without_breaking(12345)\n}"
	md := "Look at this:\n\n```go\n" + goSample + "\n```\n\n```python\nprint('block2')\n```"

	lines := renderMarkdownLines(ansiSty, 74, md, true, false, &markdownRenderCache{})
	joinedANSI := strings.Join(lines, "\n")
	if !strings.Contains(joinedANSI, "\x1b[") {
		t.Fatalf("expected Chroma ANSI escape sequences in non-ASCII mode")
	}
	joinedPlain := stripANSI(joinedANSI)
	if !strings.Contains(joinedPlain, "go") || !strings.Contains(joinedPlain, "[c] copy") {
		t.Fatalf("expected language + [c] copy header, got:\n%s", joinedPlain)
	}
	if strings.Contains(joinedPlain, "\t") {
		t.Fatalf("raw tab character leaked into rendered code block (tabs must be expanded):\n%s", joinedPlain)
	}
	if !strings.Contains(joinedPlain, "1 │") || !strings.Contains(joinedPlain, "4 │") {
		t.Fatalf("expected line numbers in rendered code block, got:\n%s", joinedPlain)
	}

	// Verify EVERY box line has identical visual display width (wrapping never breaks borders).
	var boxWidths []int
	for _, l := range strings.Split(joinedANSI, "\n") {
		plain := stripANSI(l)
		trim := strings.TrimSpace(plain)
		if strings.HasPrefix(trim, "╭") || strings.HasPrefix(trim, "│") || strings.HasPrefix(trim, "╰") {
			boxWidths = append(boxWidths, lipgloss.Width(l))
		}
	}
	if len(boxWidths) < 6 {
		t.Fatalf("expected at least 6 code box lines, got %d", len(boxWidths))
	}
	for i, w := range boxWidths {
		if w != boxWidths[0] {
			t.Fatalf("box line %d width=%d mismatches header width=%d (border broken!):\n%s", i, w, boxWidths[0], joinedPlain)
		}
	}

	// Test /lines toggle off & back on (line numbers default ON).
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "show code"},
		{Role: "assistant", Content: md},
	}
	if !strings.Contains(renderChatAll(m), "1 │") {
		t.Fatalf("expected line numbers ON by default")
	}
	m.chatInput.SetValue("/lines")
	m.submitChatInput()
	if strings.Contains(renderChatAll(m), "1 │") {
		t.Fatalf("expected /lines toggle to disable line numbers")
	}
	m.chatInput.SetValue("/lines")
	m.submitChatInput()
	if !strings.Contains(renderChatAll(m), "1 │") {
		t.Fatalf("expected second /lines toggle to re-enable line numbers")
	}

	// Test /copy 1 (raw unpadded code, no line numbers/borders).
	m.chatInput.SetValue("/copy 1")
	_, cmd := m.submitChatInput()
	if cmd == nil {
		t.Fatalf("expected /copy 1 to return clipboard cmd")
	}
	_ = cmd()
	if copied != goSample {
		t.Fatalf("/copy 1 copied %q, want exact raw block %q", copied, goSample)
	}

	// Test unfocused `c` key copies latest code block (block 2 = python).
	copied = ""
	m.chatInput.Blur()
	_, cCmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'c'}})
	if cCmd == nil {
		t.Fatalf("expected unfocused 'c' to trigger copyLatestCodeOrReply cmd")
	}
	_ = cCmd()
	if copied != "print('block2')" {
		t.Fatalf("unfocused 'c' copied %q, want latest code block %q", copied, "print('block2')")
	}
}

func TestIncrementalMarkdownHighlightNeverRehighlightsCompletedLines(t *testing.T) {
	sty := newStyles(TokyoNight, false)
	sty.true = true
	cache := &markdownRenderCache{}

	// Simulate streaming 3 lines, growing line-by-line cleanly.
	renderMarkdownLines(sty, 100, "```go\npackage main\nfunc a() {}\nvar x = 1", false, true, cache)
	opsAfter3 := cache.ops
	if opsAfter3 < 3 {
		t.Fatalf("expected >= 3 initial highlight ops, got %d", opsAfter3)
	}

	// Append a 4th line delta and partial 5th line: lines 0, 1, 2 were completed previously
	// and MUST NOT be re-highlighted (at most line 3 revised + line 4 = 2 ops).
	renderMarkdownLines(sty, 100, "```go\npackage main\nfunc a() {}\nvar x = 1\nvar y = 2", false, true, cache)
	deltaOps := cache.ops - opsAfter3
	if deltaOps > 2 {
		t.Fatalf("incremental cache re-highlighted completed lines: deltaOps=%d (want <=2; total %d -> %d)", deltaOps, opsAfter3, cache.ops)
	}
}

func TestChatGoldenSnapshots(t *testing.T) {
	update := os.Getenv("UPDATE_GOLDEN") == "1"

	for _, sz := range []struct {
		w, h int
		name string
	}{
		{80, 24, "chat-80x24.golden"},
		{120, 32, "chat-120x32.golden"},
		{160, 40, "chat-160x40.golden"},
	} {
		t.Run(sz.name, func(t *testing.T) {
			m := testUI(t)
			m.agentWorkDir = "demo-ws"
			m.screen = screenRun
			m.runTab = runTabChat
			m.status = "READY"
			m.width, m.height = sz.w, sz.h
			m.activePort = 8080
			m.codeLineNums = true
			m.telemetry = runner.Telemetry{GenTokPerSec: 42.5, KVCacheRatio: 0.25}
			m.chatCtxUsed = 1024
			m.modelCaps = runner.ModelProps{SupportsEnableThinking: true, Loaded: true}
			m.resizeRunViews()

			start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
			expanded := true
			m.chatMsgs = []chatTurn{
				{Role: "user", Content: "Write a fast Go Fibonacci helper and explain complexity."},
				{
					Role:          "assistant",
					Reasoning:     "Need iterative O(n) loop.\nAvoid exponential recursion.",
					ThinkStart:    start,
					ThinkEnd:      start.Add(1400 * time.Millisecond),
					ThinkTokens:   18,
					ThinkDone:     true,
					ThinkExpanded: &expanded,
					GenTokSec:     42.5,
					Tokens:        64,
					Content:       "### Iterative Go Implementation\n\nHere is a constant-space version:\n\n```go\npackage mathutil\n\n// Fib returns the n-th Fibonacci number in O(n) time and O(1) space.\nfunc Fib(n int) int {\n\tif n < 2 {\n\t\treturn n\n\t}\n\ta, b := 0, 1\n\tfor i := 2; i <= n; i++ {\n\t\ta, b = b, a+b\n\t}\n\treturn b\n}\n```\n\n- Time: **O(n)** using `a, b = b, a+b`.",
				},
			}
			m.refreshChatView(false)

			got := strings.TrimSpace(stripANSI(m.View()))
			path := filepath.Join("testdata", sz.name)
			if update {
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(got+"\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			wantBytes, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("missing golden file %s (run UPDATE_GOLDEN=1 go test): %v", path, err)
			}
			want := strings.TrimSpace(string(wantBytes))
			if got != want {
				t.Fatalf("snapshot mismatch for %s:\n--- GOT ---\n%s\n--- WANT ---\n%s", sz.name, got, want)
			}
		})
	}
}

func BenchmarkStreamCodeBlock200Lines(b *testing.B) {
	sty := newStyles(Mocha, false)
	sty.true = true
	lines := make([]string, 200)
	lines[0] = "```go"
	for i := 1; i < 199; i++ {
		lines[i] = fmt.Sprintf("\tif v%d := computeValue(%d); v%d > 0 { sum += v%d }", i, i, i, i)
	}
	lines[199] = "```"

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cache := &markdownRenderCache{}
		var sb strings.Builder
		for n, ln := range lines {
			if n > 0 {
				sb.WriteByte('\n')
			}
			sb.WriteString(ln)
			_ = renderMarkdownLines(sty, 110, sb.String(), true, n < len(lines)-1, cache)
		}
	}
}
