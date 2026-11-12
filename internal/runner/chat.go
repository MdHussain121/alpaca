package runner

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ToolFunctionDef describes a JSON-schema callable tool exposed to the model.
type ToolFunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

// ToolDef is an OpenAI-compatible function tool definition.
type ToolDef struct {
	Type     string          `json:"type"`
	Function ToolFunctionDef `json:"function"`
}

// ToolFunctionCall holds function name and raw JSON arguments emitted by a model.
type ToolFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ToolCall represents a single completed or accumulated tool call on an assistant message.
type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolFunctionCall `json:"function"`
}

// ToolCallDelta holds one streamed incremental delta piece of a tool call.
type ToolCallDelta struct {
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolFunctionCall `json:"function"`
}

// APIMessage is a single message sent to /v1/chat/completions.
type APIMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// StreamOptions asks OpenAI-compatible servers for final token usage counters.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ChatRequest configures a streaming completion request against llama-server.
type ChatRequest struct {
	Messages           []APIMessage   `json:"messages"`
	Tools              []ToolDef      `json:"tools,omitempty"`
	ToolChoice         any            `json:"tool_choice,omitempty"`
	Stream             bool           `json:"stream"`
	TimingsPerToken    bool           `json:"timings_per_token"`
	CachePrompt        *bool          `json:"cache_prompt,omitempty"`
	StreamOptions      *StreamOptions `json:"stream_options,omitempty"`
	ReasoningEffort    string         `json:"reasoning_effort,omitempty"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
	MaxTokens          int            `json:"max_tokens,omitempty"`
	Temperature        float64        `json:"temperature"`
	TopP               float64        `json:"top_p"`
	TopK               int            `json:"top_k,omitempty"`
	MinP               float64        `json:"min_p"`
}

// ChatDelta carries incremental text, reasoning, tool calls, and live telemetry from an SSE chunk.
type ChatDelta struct {
	Content            string
	Reasoning          string
	ToolCalls          []ToolCallDelta
	FinishReason       string
	PromptTokens       int
	CachedPromptTokens int
	CompletionTokens   int
	PromptTokSec       float64
	GenTokPerSec       float64
	Done               bool
}

// Telemetry holds live server metrics polled from /metrics and /slots.
type Telemetry struct {
	PromptTokSec       float64
	GenTokPerSec       float64
	KVCacheRatio       float64
	RequestsProcessing int
	NCtx               int
	SlotsProcessing    int
	TotalPromptTokens  int
	TotalGenTokens     int
}

type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content          string          `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			Reasoning        string          `json:"reasoning"`
			ToolCalls        []ToolCallDelta `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	Timings *struct {
		CacheN             int     `json:"cache_n"`
		PromptN            int     `json:"prompt_n"`
		PromptPerSecond    float64 `json:"prompt_per_second"`
		PredictedN         int     `json:"predicted_n"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
	} `json:"timings"`
}

// Shared keep-alive HTTP transport & clients so multi-turn chat reuses TCP
// connections instead of allocating a fresh Client/Transport per request.
var (
	sharedTransport = &http.Transport{
		MaxIdleConns:        8,
		MaxIdleConnsPerHost: 4,
		IdleConnTimeout:     90 * time.Second,
	}
	chatHTTPClient      = &http.Client{Transport: sharedTransport}
	telemetryHTTPClient = &http.Client{Transport: sharedTransport, Timeout: 900 * time.Millisecond}
)

// StreamChat posts req to http://127.0.0.1:<port>/v1/chat/completions (or baseURL if overridden
// via StreamChatURL) and streams incremental ChatDelta values until completion or cancellation.
func StreamChat(ctx context.Context, port int, req ChatRequest, onDelta func(ChatDelta)) error {
	return StreamChatURL(ctx, fmt.Sprintf("http://127.0.0.1:%d", port), req, onDelta)
}

// StreamChatURL posts req to baseURL+"/v1/chat/completions" and streams SSE deltas.
func StreamChatURL(ctx context.Context, baseURL string, req ChatRequest, onDelta func(ChatDelta)) error {
	req.Stream = true
	req.TimingsPerToken = true
	if req.CachePrompt == nil {
		on := true
		req.CachePrompt = &on
	}
	if req.StreamOptions == nil {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}
	if req.ReasoningEffort == "default" {
		req.ReasoningEffort = ""
	}

	body, err := json.Marshal(req)
	if err != nil {
		return err
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := chatHTTPClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		msg := strings.TrimSpace(string(raw))
		if msg == "" {
			msg = resp.Status
		}
		return fmt.Errorf("chat HTTP %d: %s", resp.StatusCode, msg)
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			if onDelta != nil {
				onDelta(ChatDelta{Done: true})
			}
			return nil
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		var d ChatDelta
		if len(chunk.Choices) > 0 {
			c := chunk.Choices[0]
			d.Content = c.Delta.Content
			d.Reasoning = c.Delta.ReasoningContent
			if d.Reasoning == "" {
				d.Reasoning = c.Delta.Reasoning
			}
			if len(c.Delta.ToolCalls) > 0 {
				d.ToolCalls = c.Delta.ToolCalls
			}
			d.FinishReason = c.FinishReason
		}
		if chunk.Timings != nil {
			d.PromptTokens = chunk.Timings.CacheN + chunk.Timings.PromptN
			d.CachedPromptTokens = chunk.Timings.CacheN
			d.CompletionTokens = chunk.Timings.PredictedN
			d.PromptTokSec = chunk.Timings.PromptPerSecond
			d.GenTokPerSec = chunk.Timings.PredictedPerSecond
		}
		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				d.PromptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				d.CompletionTokens = chunk.Usage.CompletionTokens
			}
			if chunk.Usage.PromptTokensDetails != nil && chunk.Usage.PromptTokensDetails.CachedTokens > 0 {
				d.CachedPromptTokens = chunk.Usage.PromptTokensDetails.CachedTokens
			}
		}
		if onDelta != nil {
			onDelta(d)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return nil
}

// InputTokens queries POST /v1/chat/completions/input_tokens to get exact templated
// prompt token counts (including tool schemas) from llama-server.
func InputTokens(ctx context.Context, port int, req ChatRequest) (int, error) {
	return InputTokensURL(ctx, fmt.Sprintf("http://127.0.0.1:%d", port), req)
}

// InputTokensURL posts req to baseURL+"/v1/chat/completions/input_tokens".
func InputTokensURL(ctx context.Context, baseURL string, req ChatRequest) (int, error) {
	payload := struct {
		Messages []APIMessage `json:"messages"`
		Tools    []ToolDef    `json:"tools,omitempty"`
	}{
		Messages: req.Messages,
		Tools:    req.Tools,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/chat/completions/input_tokens"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := telemetryHTTPClient.Do(httpReq)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("input_tokens HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var out struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, err
	}
	return out.InputTokens, nil
}

// FetchTelemetry queries /metrics and /slots from a running llama-server instance.
func FetchTelemetry(ctx context.Context, port int) (Telemetry, error) {
	return FetchTelemetryURL(ctx, fmt.Sprintf("http://127.0.0.1:%d", port))
}

// FetchTelemetryURL queries /metrics and /slots from baseURL.
func FetchTelemetryURL(ctx context.Context, baseURL string) (Telemetry, error) {
	base := strings.TrimRight(baseURL, "/")
	client := telemetryHTTPClient
	var out Telemetry

	// 1. Prometheus /metrics for prompt & generation throughput.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/metrics", nil)
	if err == nil {
		if resp, err := client.Do(req); err == nil {
			if resp.StatusCode == http.StatusOK {
				raw, _ := io.ReadAll(io.LimitReader(resp.Body, 256*1024))
				parsePromMetrics(string(raw), &out)
			} else {
				io.Copy(io.Discard, resp.Body)
			}
			resp.Body.Close()
		}
	}

	// 2. /slots for active n_ctx and busy state.
	sreq, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/slots", nil)
	if err == nil {
		if resp, err := client.Do(sreq); err == nil {
			if resp.StatusCode == http.StatusOK {
				var slots []struct {
					NCtx         int  `json:"n_ctx"`
					IsProcessing bool `json:"is_processing"`
				}
				if err := json.NewDecoder(io.LimitReader(resp.Body, 256*1024)).Decode(&slots); err == nil {
					for _, sl := range slots {
						if out.NCtx == 0 && sl.NCtx > 0 {
							out.NCtx = sl.NCtx
						}
						if sl.IsProcessing {
							out.SlotsProcessing++
						}
					}
				}
			} else {
				io.Copy(io.Discard, resp.Body)
			}
			resp.Body.Close()
		}
	}

	return out, nil
}

func parsePromMetrics(text string, t *Telemetry) {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := fields[0]
		if idx := strings.IndexByte(name, '{'); idx >= 0 {
			name = name[:idx]
		}
		val, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err != nil {
			continue
		}
		switch name {
		case "llamacpp:prompt_tokens_seconds":
			t.PromptTokSec = val
		case "llamacpp:predicted_tokens_seconds":
			t.GenTokPerSec = val
		case "llamacpp:kv_cache_usage_ratio":
			t.KVCacheRatio = val
		case "llamacpp:requests_processing":
			t.RequestsProcessing = int(val)
		case "llamacpp:prompt_tokens_total":
			t.TotalPromptTokens = int(val)
		case "llamacpp:tokens_predicted_total":
			t.TotalGenTokens = int(val)
		}
	}
}
