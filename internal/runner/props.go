package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ModelProps summarizes capabilities returned by llama-server's GET /props endpoint.
type ModelProps struct {
	ModelAlias              string
	NCtx                    int
	ChatTemplate            string
	SupportsEnableThinking  bool
	SupportsTools           bool
	SupportsToolCalls       bool
	SupportsParallelTools   bool
	SupportsReasoningEffort bool
	ReasoningEffortLevels   []string
	Loaded                  bool
}

type propsResponse struct {
	ModelAlias                string `json:"model_alias"`
	ChatTemplate              string `json:"chat_template"`
	DefaultGenerationSettings struct {
		NCtx int `json:"n_ctx"`
	} `json:"default_generation_settings"`
	ChatTemplateCaps struct {
		SupportsTools             bool     `json:"supports_tools"`
		SupportsToolCalls         bool     `json:"supports_tool_calls"`
		SupportsParallelToolCalls bool     `json:"supports_parallel_tool_calls"`
		SupportsReasoningEffort   bool     `json:"supports_reasoning_effort"`
		ReasoningEffortLevels     []string `json:"reasoning_effort_levels"`
	} `json:"chat_template_caps"`
}

// FetchProps queries GET /props on the local llama-server port.
func FetchProps(ctx context.Context, port int) (ModelProps, error) {
	return FetchPropsURL(ctx, fmt.Sprintf("http://127.0.0.1:%d", port))
}

// FetchPropsURL queries GET /props on baseURL and extracts template capabilities.
func FetchPropsURL(ctx context.Context, baseURL string) (ModelProps, error) {
	endpoint := strings.TrimRight(baseURL, "/") + "/props"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return ModelProps{}, err
	}
	resp, err := telemetryHTTPClient.Do(req)
	if err != nil {
		return ModelProps{}, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	if err != nil {
		return ModelProps{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return ModelProps{}, fmt.Errorf("props HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	return ParsePropsJSON(raw)
}

// ParsePropsJSON decodes llama-server /props JSON into ModelProps.
func ParsePropsJSON(raw []byte) (ModelProps, error) {
	var pr propsResponse
	if err := json.Unmarshal(raw, &pr); err != nil {
		return ModelProps{}, err
	}
	mp := ModelProps{
		ModelAlias:              pr.ModelAlias,
		NCtx:                    pr.DefaultGenerationSettings.NCtx,
		ChatTemplate:            pr.ChatTemplate,
		SupportsEnableThinking:  strings.Contains(pr.ChatTemplate, "enable_thinking"),
		SupportsTools:           pr.ChatTemplateCaps.SupportsTools,
		SupportsToolCalls:       pr.ChatTemplateCaps.SupportsToolCalls,
		SupportsParallelTools:   pr.ChatTemplateCaps.SupportsParallelToolCalls,
		SupportsReasoningEffort: pr.ChatTemplateCaps.SupportsReasoningEffort,
		ReasoningEffortLevels:   append([]string(nil), pr.ChatTemplateCaps.ReasoningEffortLevels...),
		Loaded:                  true,
	}
	return mp, nil
}
