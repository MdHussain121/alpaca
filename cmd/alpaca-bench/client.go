package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type PromptItem struct {
	ID        string `json:"id"`
	Tier      string `json:"tier"`
	Category  string `json:"category"`
	MaxTokens int    `json:"max_tokens"`
	ApproxTok int    `json:"approx_tokens"`
	Text      string `json:"text"`
}

type Timings struct {
	CacheN             int     `json:"cache_n"`
	PromptN            int     `json:"prompt_n"`
	PromptMs           float64 `json:"prompt_ms"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	PredictedN         int     `json:"predicted_n"`
	PredictedMs        float64 `json:"predicted_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
	DraftN             int     `json:"draft_n"`
	DraftNAccepted     int     `json:"draft_n_accepted"`
}

type PromptResult struct {
	ID          string  `json:"id"`
	Tier        string  `json:"tier"`
	Category    string  `json:"category"`
	TTFTMs      float64 `json:"ttft_ms"`
	TotalWallMs float64 `json:"total_wall_ms"`
	Timings     Timings `json:"timings"`
	OutputChars int     `json:"output_chars"`
	OutputSHA   string  `json:"output_sha256"`
	OutputText  string  `json:"output_text,omitempty"`
	Error       string  `json:"error,omitempty"`
}

type benchClient struct {
	http *http.Client
	base string
}

func newBenchClient(port int) *benchClient {
	tr := &http.Transport{
		MaxIdleConns:        8,
		MaxIdleConnsPerHost: 8,
		IdleConnTimeout:     90 * time.Second,
	}
	return &benchClient{
		http: &http.Client{Transport: tr, Timeout: 10 * time.Minute},
		base: fmt.Sprintf("http://127.0.0.1:%d", port),
	}
}

func (c *benchClient) checkRoot(ctx context.Context) (status int, sizeBytes int, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/", nil)
	if err != nil {
		return 0, 0, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return resp.StatusCode, len(body), err
}

type chatReq struct {
	Messages    []chatMsg `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature float64   `json:"temperature"`
	TopP        float64   `json:"top_p,omitempty"`
	TopK        int       `json:"top_k,omitempty"`
	MinP        *float64  `json:"min_p,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Seed        int       `json:"seed"`
	CachePrompt *bool     `json:"cache_prompt,omitempty"`
	StreamOpts  *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Timings *Timings `json:"timings"`
}

func (c *benchClient) runPrompt(ctx context.Context, p PromptItem, temp float64, seed int, cachePrompt bool, saveText bool) PromptResult {
	res := PromptResult{
		ID:       p.ID,
		Tier:     p.Tier,
		Category: p.Category,
	}
	zero := 0.0
	body := chatReq{
		Messages:    []chatMsg{{Role: "user", Content: p.Text}},
		Stream:      true,
		Temperature: temp,
		MaxTokens:   p.MaxTokens,
		Seed:        seed,
		CachePrompt: &cachePrompt,
		StreamOpts: &struct {
			IncludeUsage bool `json:"include_usage"`
		}{IncludeUsage: true},
	}
	if temp == 0 {
		// Pin greedy sampling parameters explicitly so server defaults never jitter.
		body.TopK = 1
		body.TopP = 1.0
		body.MinP = &zero
	}
	raw, err := json.Marshal(body)
	if err != nil {
		res.Error = err.Error()
		return res
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		res.Error = err.Error()
		return res
	}
	req.Header.Set("Content-Type", "application/json")
	// cpp-httplib closes keep-alive sockets after a small max request/timeout threshold;
	// setting Close avoids Go http.Transport non-idempotent POST EOF races.
	req.Close = true

	start := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		slurp, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		res.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(slurp)))
		return res
	}

	br := bufio.NewReaderSize(resp.Body, 64*1024)
	var out strings.Builder
	gotFirst := false

	for {
		line, rerr := br.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "data: ") {
				payload := strings.TrimPrefix(trimmed, "data: ")
				if payload == "[DONE]" {
					continue
				}
				var ch sseChunk
				if uerr := json.Unmarshal([]byte(payload), &ch); uerr == nil {
					for _, choice := range ch.Choices {
						tok := choice.Delta.Content
						if tok == "" {
							tok = choice.Delta.ReasoningContent
						}
						if tok != "" {
							if !gotFirst {
								res.TTFTMs = float64(time.Since(start).Microseconds()) / 1000.0
								gotFirst = true
							}
							out.WriteString(tok)
						}
					}
					if ch.Timings != nil {
						res.Timings = *ch.Timings
					}
				}
			}
		}
		if rerr != nil {
			if !errors.Is(rerr, io.EOF) {
				res.Error = rerr.Error()
			}
			break
		}
	}

	res.TotalWallMs = float64(time.Since(start).Microseconds()) / 1000.0
	txt := out.String()
	res.OutputChars = len(txt)
	sum := sha256.Sum256([]byte(txt))
	res.OutputSHA = hex.EncodeToString(sum[:])
	if saveText {
		res.OutputText = txt
	}
	return res
}
