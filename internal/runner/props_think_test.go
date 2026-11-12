package runner

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestThinkStreamParserSplitTags(t *testing.T) {
	chunks := []string{
		"Pre<thi",
		"nk>line1\n",
		"line2</th",
		"ink>\n\nFinal ",
		"<4> answer",
	}
	var p ThinkStreamParser
	var reason, content strings.Builder
	for _, ch := range chunks {
		r, c := p.Feed(ch)
		reason.WriteString(r)
		content.WriteString(c)
	}
	r, c := p.Flush()
	reason.WriteString(r)
	content.WriteString(c)

	if reason.String() != "line1\nline2" {
		t.Fatalf("reasoning = %q, want %q", reason.String(), "line1\nline2")
	}
	if content.String() != "PreFinal <4> answer" {
		t.Fatalf("content = %q, want %q", content.String(), "PreFinal <4> answer")
	}
	if !p.Closed() {
		t.Fatal("expected p.Closed() == true")
	}
}

func TestThinkStreamParserUnclosed(t *testing.T) {
	var p ThinkStreamParser
	r1, c1 := p.Feed("<think>partial thought")
	r2, c2 := p.Flush()
	if r1+r2 != "partial thought" || c1+c2 != "" {
		t.Fatalf("got reason=%q content=%q", r1+r2, c1+c2)
	}
	if p.Closed() {
		t.Fatal("expected p.Closed() == false for unclosed think block")
	}
}

func TestParsePropsRealModels(t *testing.T) {
	cases := []struct {
		file           string
		enableThinking bool
		effortLevels   []string
	}{
		{"lfm2.5-8b_endpoints.json", false, nil},
		{"qwen3.5-4b_endpoints.json", true, nil},
		{"spark-x2.6-4b_endpoints.json", true, []string{"low", "high"}},
		{"tiel-coder-35b_endpoints.json", true, []string{"low", "max"}},
	}
	for _, tc := range cases {
		raw, err := os.ReadFile(filepath.Join("testdata", "endpoints", tc.file))
		if err != nil {
			t.Fatalf("read %s: %v", tc.file, err)
		}
		var doc struct {
			Props json.RawMessage `json:"props"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatalf("unmarshal envelope %s: %v", tc.file, err)
		}
		mp, err := ParsePropsJSON(doc.Props)
		if err != nil {
			t.Fatalf("ParsePropsJSON %s: %v", tc.file, err)
		}
		if !mp.Loaded || !mp.SupportsTools || !mp.SupportsToolCalls {
			t.Fatalf("%s missing expected loaded/tool caps: %+v", tc.file, mp)
		}
		if mp.SupportsEnableThinking != tc.enableThinking {
			t.Fatalf("%s SupportsEnableThinking = %v, want %v", tc.file, mp.SupportsEnableThinking, tc.enableThinking)
		}
		if len(mp.ReasoningEffortLevels) != len(tc.effortLevels) {
			t.Fatalf("%s ReasoningEffortLevels = %v, want %v", tc.file, mp.ReasoningEffortLevels, tc.effortLevels)
		}
	}
}

func TestRealSSEThinkingFixtures(t *testing.T) {
	for _, model := range []string{"qwen3.5-4b", "spark-x2.6-4b", "lfm2.5-8b", "tiel-coder-35b"} {
		f, err := os.Open(filepath.Join("..", "..", "testdata", "streams", model, "thinking.sse"))
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		var reason, content strings.Builder
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			p := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if p == "" || p == "[DONE]" {
				continue
			}
			var chunk sseChunk
			if json.Unmarshal([]byte(p), &chunk) != nil || len(chunk.Choices) == 0 {
				continue
			}
			reason.WriteString(chunk.Choices[0].Delta.ReasoningContent)
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
		_ = f.Close()
		if reason.Len() == 0 || content.Len() == 0 {
			t.Fatalf("%s thinking.sse parsed reason=%d content=%d, want non-empty both", model, reason.Len(), content.Len())
		}
	}
}
