package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/MdHussain121/alpaca/internal/runner"
)

func replaySSEFixture(t *testing.T, fixturePath string) (content, reasoning string, calls []runner.ToolCall) {
	t.Helper()
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture %s: %v", fixturePath, err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(raw)
	}))
	defer srv.Close()

	var acc ToolCallAccumulator
	var cBuf, rBuf string
	err = runner.StreamChatURL(context.Background(), srv.URL, runner.ChatRequest{
		Messages: []runner.APIMessage{{Role: "user", Content: "test"}},
	}, func(d runner.ChatDelta) {
		cBuf += d.Content
		rBuf += d.Reasoning
		if len(d.ToolCalls) > 0 {
			acc.Feed(d.ToolCalls)
		}
	})
	if err != nil {
		t.Fatalf("stream fixture %s: %v", fixturePath, err)
	}
	calls = acc.Results()
	if len(calls) == 0 {
		calls, cBuf = ExtractFallbackTextToolCalls(cBuf)
	}
	return cBuf, rBuf, calls
}

func TestSSEToolFixturesAllModels(t *testing.T) {
	models := []string{"qwen3.5-4b", "spark-x2.6-4b", "lfm2.5-8b", "tiel-coder-35b"}
	for _, m := range models {
		t.Run(m+"/tool_call", func(t *testing.T) {
			p := filepath.Join("..", "..", "testdata", "streams", m, "tool_call.sse")
			_, _, calls := replaySSEFixture(t, p)
			if len(calls) < 1 {
				t.Fatalf("expected >=1 tool call in %s, got 0", p)
			}
			for _, tc := range calls {
				args, _, err := DecodeToolArgs(tc.Function.Arguments)
				if err != nil {
					t.Fatalf("%s tool %q decode args error: %v (raw=%q)", m, tc.Function.Name, err, tc.Function.Arguments)
				}
				if args == nil {
					t.Fatalf("%s tool %q nil args", m, tc.Function.Name)
				}
			}
		})

		t.Run(m+"/parallel_tool_calls", func(t *testing.T) {
			p := filepath.Join("..", "..", "testdata", "streams", m, "parallel_tool_calls.sse")
			_, _, calls := replaySSEFixture(t, p)
			if len(calls) < 2 {
				t.Fatalf("expected >=2 parallel tool calls in %s, got %d (%+v)", p, len(calls), calls)
			}
		})

		t.Run(m+"/tool_call_bad_args", func(t *testing.T) {
			p := filepath.Join("..", "..", "testdata", "streams", m, "tool_call_bad_args.sse")
			content, reasoning, calls := replaySSEFixture(t, p)
			// Models may emit malformed tool_calls OR hit max_tokens inside reasoning/content.
			for _, tc := range calls {
				_, _, err := DecodeToolArgs(tc.Function.Arguments)
				t.Logf("%s bad_args tool=%s raw=%q decodeErr=%v", m, tc.Function.Name, tc.Function.Arguments, err)
			}
			if len(calls) == 0 && content == "" && reasoning == "" {
				t.Fatalf("expected either tool_calls, content, or reasoning in %s", p)
			}
		})
	}
}

func TestDecodeToolArgsRepairAndFallbackFormats(t *testing.T) {
	cases := []struct {
		raw      string
		wantOK   bool
		wantPath string
	}{
		{raw: `{"path":"a.txt",}`, wantOK: true, wantPath: "a.txt"},
		{raw: `{"path":"a.txt","content":"hel`, wantOK: true, wantPath: "a.txt"},
		{raw: `{"path":`, wantOK: true},
		{raw: "```json\n{\"path\":\"b.txt\"}\n```", wantOK: true, wantPath: "b.txt"},
		{raw: `not json at all`, wantOK: false},
	}
	for _, tc := range cases {
		args, _, err := DecodeToolArgs(tc.raw)
		if tc.wantOK && err != nil {
			t.Fatalf("DecodeToolArgs(%q) unexpected err: %v", tc.raw, err)
		}
		if !tc.wantOK && err == nil {
			t.Fatalf("DecodeToolArgs(%q) expected error, got %+v", tc.raw, args)
		}
		if tc.wantPath != "" && args["path"] != tc.wantPath {
			t.Fatalf("DecodeToolArgs(%q) path=%v want=%q", tc.raw, args["path"], tc.wantPath)
		}
	}

	xmlCalls, stripped := ExtractFallbackTextToolCalls("Checking\n<tool_call>\n<function=read_file><parameter=path>go.mod</parameter></function>\n</tool_call>")
	if len(xmlCalls) != 1 || xmlCalls[0].Function.Name != "read_file" || stripped != "Checking" {
		t.Fatalf("XML fallback mismatch: calls=%+v stripped=%q", xmlCalls, stripped)
	}
	mistralCalls, stripped2 := ExtractFallbackTextToolCalls(`[TOOL_CALLS] [{"name":"list_dir","arguments":{"path":"."}}]`)
	if len(mistralCalls) != 1 || mistralCalls[0].Function.Name != "list_dir" || stripped2 != "" {
		t.Fatalf("Mistral fallback mismatch: calls=%+v stripped=%q", mistralCalls, stripped2)
	}
}
