package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/MdHussain121/alpaca/internal/runner"
)

// ToolCallAccumulator collects streamed OpenAI delta.tool_calls chunks by index.
type ToolCallAccumulator struct {
	calls []*runner.ToolCall
}

func (a *ToolCallAccumulator) Feed(deltas []runner.ToolCallDelta) {
	for _, d := range deltas {
		idx := d.Index
		if idx < 0 {
			idx = 0
		}
		for len(a.calls) <= idx {
			a.calls = append(a.calls, nil)
		}
		tc := a.calls[idx]
		if tc == nil {
			tc = &runner.ToolCall{Type: "function"}
			a.calls[idx] = tc
		}
		if d.ID != "" {
			tc.ID = d.ID
		}
		if d.Type != "" {
			tc.Type = d.Type
		}
		if d.Function.Name != "" {
			tc.Function.Name += d.Function.Name
		}
		if d.Function.Arguments != "" {
			tc.Function.Arguments += d.Function.Arguments
		}
	}
}

func (a *ToolCallAccumulator) Results() []runner.ToolCall {
	var out []runner.ToolCall
	for i, tc := range a.calls {
		if tc == nil || strings.TrimSpace(tc.Function.Name) == "" {
			continue
		}
		cp := *tc
		if cp.ID == "" {
			cp.ID = fmt.Sprintf("call_%d", i)
		}
		if cp.Type == "" {
			cp.Type = "function"
		}
		out = append(out, cp)
	}
	return out
}

var (
	qwenToolTagRe   = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)
	qwenXMLFnRe     = regexp.MustCompile(`(?s)<function=([A-Za-z0-9_.\-]+)>(.*?)</function>`)
	qwenXMLParamRe  = regexp.MustCompile(`(?s)<parameter=([A-Za-z0-9_.\-]+)>(.*?)</parameter>`)
	bracketCallRe   = regexp.MustCompile(`(?s)\[TOOL_CALLS?\]\s*(\[.*?\]|\{.*?\})`)
	trailingCommaRe = regexp.MustCompile(`,\s*([}\]])`)
)

// ExtractFallbackTextToolCalls parses models that emit tool calls as inline text
// (e.g. <tool_call>{...}</tool_call>, Qwen XML <function=...>, [TOOL_CALLS][...], or fenced JSON).
func ExtractFallbackTextToolCalls(content string) (calls []runner.ToolCall, strippedText string) {
	s := strings.TrimSpace(content)
	if s == "" {
		return nil, content
	}

	// 1. <tool_call>{"name":...,"arguments":{...}}</tool_call>
	if matches := qwenToolTagRe.FindAllStringSubmatch(s, -1); len(matches) > 0 {
		for i, m := range matches {
			if tc, ok := parseSingleCallJSON(m[1], i); ok {
				calls = append(calls, tc)
			}
		}
		if len(calls) > 0 {
			cleaned := strings.TrimSpace(qwenToolTagRe.ReplaceAllString(s, ""))
			return calls, cleaned
		}
	}

	// 2. Qwen3 XML fallback (<function=name><parameter=k>v</parameter></function>)
	if fms := qwenXMLFnRe.FindAllStringSubmatch(s, -1); len(fms) > 0 {
		for i, fm := range fms {
			args := map[string]any{}
			for _, pm := range qwenXMLParamRe.FindAllStringSubmatch(fm[2], -1) {
				args[pm[1]] = coerceScalarOrJSON(strings.TrimSpace(pm[2]))
			}
			b, _ := json.Marshal(args)
			calls = append(calls, runner.ToolCall{
				ID:   fmt.Sprintf("call_xml_%d", i),
				Type: "function",
				Function: runner.ToolFunctionCall{
					Name:      fm[1],
					Arguments: string(b),
				},
			})
		}
		cleaned := strings.TrimSpace(qwenXMLFnRe.ReplaceAllString(s, ""))
		cleaned = strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(cleaned, "<tool_call>", ""), "</tool_call>", ""))
		return calls, cleaned
	}

	// 3. Mistral / LFM style [TOOL_CALLS][{"name":...,"arguments":{...}}]
	if bm := bracketCallRe.FindStringSubmatch(s); len(bm) == 2 {
		parsed := parseArrayOrObjectCalls(bm[1])
		if len(parsed) > 0 {
			cleaned := strings.TrimSpace(bracketCallRe.ReplaceAllString(s, ""))
			return parsed, cleaned
		}
	}

	// 4. Whole content is pure {"name":...,"arguments":...}
	if strings.HasPrefix(s, "{") && strings.Contains(s, `"name"`) {
		if tc, ok := parseSingleCallJSON(s, 0); ok {
			return []runner.ToolCall{tc}, ""
		}
	}

	return nil, content
}

func parseArrayOrObjectCalls(raw string) []runner.ToolCall {
	var arr []json.RawMessage
	if err := json.Unmarshal([]byte(raw), &arr); err == nil {
		var out []runner.ToolCall
		for i, item := range arr {
			if tc, ok := parseSingleCallJSON(string(item), i); ok {
				out = append(out, tc)
			}
		}
		return out
	}
	if tc, ok := parseSingleCallJSON(raw, 0); ok {
		return []runner.ToolCall{tc}
	}
	return nil
}

func parseSingleCallJSON(raw string, idx int) (runner.ToolCall, bool) {
	var env struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(raw), &env); err != nil {
		repaired := repairToolJSON(raw)
		if err2 := json.Unmarshal([]byte(repaired), &env); err2 != nil {
			return runner.ToolCall{}, false
		}
	}
	if strings.TrimSpace(env.Name) == "" {
		return runner.ToolCall{}, false
	}
	argsStr := "{}"
	if len(env.Arguments) > 0 {
		// Accept either embedded JSON object or JSON-encoded string.
		var asStr string
		if err := json.Unmarshal(env.Arguments, &asStr); err == nil {
			argsStr = asStr
		} else {
			argsStr = string(env.Arguments)
		}
	}
	return runner.ToolCall{
		ID:   fmt.Sprintf("call_txt_%d", idx),
		Type: "function",
		Function: runner.ToolFunctionCall{
			Name:      strings.TrimSpace(env.Name),
			Arguments: argsStr,
		},
	}, true
}

func coerceScalarOrJSON(s string) any {
	if s == "true" {
		return true
	}
	if s == "false" {
		return false
	}
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	if (strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}")) || (strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]")) {
		var v any
		if json.Unmarshal([]byte(s), &v) == nil {
			return v
		}
	}
	return s
}

// DecodeToolArgs parses raw JSON arguments into map[string]any, performing a single
// safe repair pass (removing trailing commas, closing truncated quotes/brackets/braces)
// if initial decoding fails.
func DecodeToolArgs(raw string) (args map[string]any, repaired bool, err error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]any{}, false, nil
	}
	if e := json.Unmarshal([]byte(trimmed), &args); e == nil && args != nil {
		return args, false, nil
	}

	fixed := repairToolJSON(trimmed)
	if fixed != trimmed {
		if e := json.Unmarshal([]byte(fixed), &args); e == nil && args != nil {
			return args, true, nil
		}
	}
	return nil, false, fmt.Errorf("invalid tool arguments JSON: %s", truncateForErr(trimmed, 160))
}

func repairToolJSON(raw string) string {
	s := strings.TrimSpace(raw)
	// Strip markdown fence if wrapped.
	if strings.HasPrefix(s, "```") {
		if nl := strings.IndexByte(s, '\n'); nl >= 0 {
			s = s[nl+1:]
		}
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
		s = strings.TrimSpace(s)
	}
	s = trailingCommaRe.ReplaceAllString(s, "$1")

	// Track quote & brace/bracket balance and close incomplete structures cleanly once.
	inStr := false
	escaped := false
	var stack []byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			stack = append(stack, c)
		case '}':
			if len(stack) > 0 && stack[len(stack)-1] == '{' {
				stack = stack[:len(stack)-1]
			}
		case ']':
			if len(stack) > 0 && stack[len(stack)-1] == '[' {
				stack = stack[:len(stack)-1]
			}
		}
	}

	var b strings.Builder
	b.WriteString(s)
	if inStr {
		if escaped {
			b.WriteByte('\\')
		}
		b.WriteByte('"')
	}
	out := trailingCommaRe.ReplaceAllString(b.String(), "$1")
	out = strings.TrimRight(out, ", \t\r\n")
	if strings.HasSuffix(out, ":") {
		out += "null"
	}
	for i := len(stack) - 1; i >= 0; i-- {
		if stack[i] == '{' {
			out += "}"
		} else {
			out += "]"
		}
	}
	return trailingCommaRe.ReplaceAllString(out, "$1")
}

func truncateForErr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "…"
}
