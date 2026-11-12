package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStreamChatURLParsesSSEAndTimings(t *testing.T) {
	var gotReq ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning_content":"think "}}]}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}],"timings":{"cache_n":10,"prompt_n":20,"prompt_per_second":250.5,"predicted_n":1,"predicted_per_second":42.2}}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":2},"timings":{"cache_n":10,"prompt_n":20,"prompt_per_second":250.5,"predicted_n":2,"predicted_per_second":64.8}}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer srv.Close()

	var text, think strings.Builder
	var last ChatDelta
	var sawDone bool
	err := StreamChatURL(context.Background(), srv.URL, ChatRequest{
		Messages:        []APIMessage{{Role: "user", Content: "hi"}},
		ReasoningEffort: "high",
		Temperature:     0.6,
		TopP:            0.9,
	}, func(d ChatDelta) {
		if d.Done {
			sawDone = true
			return
		}
		think.WriteString(d.Reasoning)
		text.WriteString(d.Content)
		last = d
	})
	if err != nil {
		t.Fatalf("StreamChatURL error: %v", err)
	}
	if !gotReq.Stream || !gotReq.TimingsPerToken || gotReq.ReasoningEffort != "high" {
		t.Fatalf("unexpected request forwarded: %+v", gotReq)
	}
	if gotReq.CachePrompt == nil || !*gotReq.CachePrompt {
		t.Fatalf("expected cache_prompt=true by default, got %v", gotReq.CachePrompt)
	}
	if think.String() != "think " || text.String() != "hello world" {
		t.Fatalf("got think=%q text=%q", think.String(), text.String())
	}
	if !sawDone || last.PromptTokens != 30 || last.CompletionTokens != 2 || last.GenTokPerSec != 64.8 {
		t.Fatalf("final metrics mismatch: done=%v last=%+v", sawDone, last)
	}
}

func TestFetchTelemetryURLParsesMetricsAndSlots(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			fmt.Fprintln(w, "# TYPE llamacpp:predicted_tokens_seconds gauge")
			fmt.Fprintln(w, "llamacpp:prompt_tokens_seconds 412.5")
			fmt.Fprintln(w, "llamacpp:predicted_tokens_seconds 71.25")
			fmt.Fprintln(w, "llamacpp:kv_cache_usage_ratio 0.25")
			fmt.Fprintln(w, "llamacpp:requests_processing 1")
		case "/slots":
			fmt.Fprintln(w, `[{"id":0,"n_ctx":16384,"is_processing":true}]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	tel, err := FetchTelemetryURL(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("FetchTelemetryURL: %v", err)
	}
	if tel.GenTokPerSec != 71.25 || tel.PromptTokSec != 412.5 {
		t.Fatalf("tok/s = %+v", tel)
	}
	if tel.NCtx != 16384 || tel.SlotsProcessing != 1 || tel.KVCacheRatio != 0.25 {
		t.Fatalf("slots/kv = %+v", tel)
	}
}
