package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/prompts"
)

type question struct {
	ID         int    `json:"id"`
	Answerable bool   `json:"answerable"`
	Text       string `json:"text"`
}

type result struct {
	question
	Final             string   `json:"final"`
	ToolCalls         int      `json:"tool_calls"`
	Searches          int      `json:"searches"`
	EmptySearches     int      `json:"empty_searches"`
	Fetches           int      `json:"fetches"`
	FailedFetches     int      `json:"failed_fetches"`
	UnseenURLFetches  int      `json:"unseen_url_fetches"`
	ToolOutputTokens  int      `json:"tool_output_tokens"`
	FetchedURLs       []string `json:"fetched_urls,omitempty"`
	CitedURLs         []string `json:"cited_urls,omitempty"`
	CitedRetrievedURL []string `json:"cited_retrieved_urls,omitempty"`
	Errs              []string `json:"errors,omitempty"`
	DurationMS        int64    `json:"duration_ms"`
}

type summary struct {
	Model            string   `json:"model"`
	BaseURL          string   `json:"base_url"`
	Questions        int      `json:"questions"`
	Answerable       int      `json:"answerable"`
	Unanswerable     int      `json:"unanswerable"`
	ToolCalls        int      `json:"tool_calls"`
	Searches         int      `json:"searches"`
	EmptySearches    int      `json:"empty_searches"`
	Fetches          int      `json:"fetches"`
	FailedFetches    int      `json:"failed_fetches"`
	UnseenURLFetches int      `json:"unseen_url_fetches"`
	ToolOutputTokens int      `json:"tool_output_tokens"`
	AnsweredWithSrc  int      `json:"answered_with_real_source"`
	AnsweredNoSrc    int      `json:"answered_without_real_source"`
	RefusedOrEmpty   int      `json:"refused_or_empty"`
	ErrorList        []string `json:"errors,omitempty"`
}

var urlRe = regexp.MustCompile(`https?://[^\s<>"'` + "`" + `)\]]+`)

func main() {
	baseURL := flag.String("base-url", "http://127.0.0.1:8080", "OpenAI-compatible llama-server base URL")
	model := flag.String("model", "", "model label for the report")
	questionsPath := flag.String("questions", "cmd/webbench/questions.txt", "question file")
	outPath := flag.String("out", "cmd/webbench/results.json", "JSON output path")
	limit := flag.Int("limit", 0, "limit questions (0 = all)")
	maxSteps := flag.Int("max-steps", 8, "agent steps per question")
	timeout := flag.Duration("timeout", 120*time.Second, "per-question timeout")
	flag.Parse()

	questions, err := loadQuestions(*questionsPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *limit > 0 && *limit < len(questions) {
		questions = questions[:*limit]
	}
	_ = os.MkdirAll(filepath.Dir(*outPath), 0o755)

	sum := summary{Model: *model, BaseURL: *baseURL, Questions: len(questions)}
	var all []result
	for _, q := range questions {
		if q.Answerable {
			sum.Answerable++
		} else {
			sum.Unanswerable++
		}
		r := runOne(*baseURL, q, *maxSteps, *timeout)
		all = append(all, r)
		sum.ToolCalls += r.ToolCalls
		sum.Searches += r.Searches
		sum.EmptySearches += r.EmptySearches
		sum.Fetches += r.Fetches
		sum.FailedFetches += r.FailedFetches
		sum.UnseenURLFetches += r.UnseenURLFetches
		sum.ToolOutputTokens += r.ToolOutputTokens
		sum.ErrorList = append(sum.ErrorList, r.Errs...)
		if len(r.CitedRetrievedURL) > 0 {
			sum.AnsweredWithSrc++
		} else if strings.TrimSpace(r.Final) == "" || looksRefusal(r.Final) {
			sum.RefusedOrEmpty++
		} else {
			sum.AnsweredNoSrc++
		}
		fmt.Printf("[%02d] %-12s tools=%d searches=%d fetches=%d failed=%d unseen=%d tok=%d cited=%d\n",
			q.ID, q.Text, r.ToolCalls, r.Searches, r.Fetches, r.FailedFetches, r.UnseenURLFetches, r.ToolOutputTokens, len(r.CitedRetrievedURL))
	}

	blob, _ := json.MarshalIndent(struct {
		Summary   summary  `json:"summary"`
		Questions []result `json:"questions"`
	}{sum, all}, "", "  ")
	if err := os.WriteFile(*outPath, blob, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("\nSUMMARY model=%s questions=%d searches=%d empty=%d fetches=%d failed=%d unseen=%d tool_tokens=%d with_source=%d without_source=%d refused_or_empty=%d\n",
		sum.Model, sum.Questions, sum.Searches, sum.EmptySearches, sum.Fetches, sum.FailedFetches,
		sum.UnseenURLFetches, sum.ToolOutputTokens, sum.AnsweredWithSrc, sum.AnsweredNoSrc, sum.RefusedOrEmpty)
	fmt.Printf("wrote %s\n", *outPath)
}

func runOne(baseURL string, q question, maxSteps int, timeout time.Duration) result {
	r := result{question: q}
	reg := agent.NewDefaultRegistry()
	agent.RegisterWebTools(reg, agent.WebToolsConfig{Web: config.WebConfig{}})

	workspace, err := os.MkdirTemp("", "alpaca-webbench-")
	if err != nil {
		r.Errs = append(r.Errs, err.Error())
		return r
	}
	defer os.RemoveAll(workspace)

	seenURLs := map[string]bool{}
	sourceURLs := map[string]bool{}
	for _, u := range urlRe.FindAllString(q.Text, -1) {
		seenURLs[strings.TrimRight(u, ".,;:!?)]")] = true
	}
	retrieved := map[string]bool{}
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req := agent.LoopRequest{
		BaseURL:      baseURL,
		SystemPrompt: prompts.Render(prompts.RenderOptions{Workspace: workspace, PermissionMode: "read-auto", Date: time.Now().Format("2006-01-02"), Tools: reg.ToolDefs(), ContextSize: 8192}).Text,
		History:      []runner.APIMessage{{Role: "user", Content: q.Text}},
		MaxSteps:     maxSteps,
		MaxTokens:    1024,
		Temperature:  0.2,
		Registry:     reg,
		Permissions:  agent.NewPermissionEngine(agent.ModeReadAuto),
		ToolCtx: agent.ToolContext{
			WorkspaceRoot: workspace,
			SessionID:     fmt.Sprintf("webbench-%d", q.ID),
			TurnNumber:    1,
			Shell:         agent.ShellCmd,
		},
		AskPermission: func(ctx context.Context, p agent.PermissionPrompt) (agent.PermissionResponse, error) {
			return agent.PermissionResponse{Choice: agent.PermAllowOnce}, nil
		},
		OnEvent: func(ev agent.LoopEvent) {
			switch ev.Kind {
			case agent.EvtToolEnd:
				r.ToolCalls++
				r.ToolOutputTokens += estimateTokens(ev.Result.Output)
				switch ev.Tool {
				case "web_search", "web_research":
					r.Searches++
					if ev.Result.IsError && (strings.Contains(strings.ToLower(ev.Result.Output), "no results") || strings.Contains(strings.ToLower(ev.Result.Output), "no keyless")) {
						r.EmptySearches++
					}
					for _, u := range urlRe.FindAllString(ev.Result.Output, -1) {
						clean := strings.TrimRight(u, ".,;:!?)]")
						seenURLs[clean] = true
						sourceURLs[clean] = true
						if ev.Tool == "web_research" && !ev.Result.IsError {
							retrieved[clean] = true
						}
					}
				case "web_fetch":
					r.Fetches++
					fetchURL := extractURLArg(ev.ArgsPretty)
					if fetchURL != "" {
						r.FetchedURLs = append(r.FetchedURLs, fetchURL)
						if !seenURLs[strings.TrimRight(fetchURL, ".,;:!?)]")] {
							r.UnseenURLFetches++
						}
					}
					if ev.Result.IsError {
						r.FailedFetches++
					} else {
						clean := strings.TrimRight(fetchURL, ".,;:!?)]")
						retrieved[clean] = true
						sourceURLs[clean] = true
					}
				}
			}
		},
	}
	res := agent.RunTurn(ctx, req)
	r.DurationMS = time.Since(start).Milliseconds()
	if res.Err != nil {
		r.Errs = append(r.Errs, res.Err.Error())
	}
	r.Final = res.FinalContent
	for _, u := range urlRe.FindAllString(res.FinalContent, -1) {
		clean := strings.TrimRight(u, ".,;:!?)]")
		r.CitedURLs = append(r.CitedURLs, clean)
		if sourceURLs[clean] || retrieved[clean] {
			r.CitedRetrievedURL = append(r.CitedRetrievedURL, clean)
		}
	}
	return r
}

func loadQuestions(path string) ([]question, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []question
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		answerable := true
		if strings.HasPrefix(line, "U|") {
			answerable = false
			line = strings.TrimSpace(line[2:])
		} else if strings.HasPrefix(line, "A|") {
			line = strings.TrimSpace(line[2:])
		}
		if line == "" {
			continue
		}
		out = append(out, question{ID: len(out) + 1, Answerable: answerable, Text: line})
	}
	return out, sc.Err()
}

func extractURLArg(pretty string) string {
	var args map[string]any
	if json.Unmarshal([]byte(pretty), &args) == nil {
		if u, ok := args["url"].(string); ok {
			return strings.TrimSpace(u)
		}
	}
	if m := regexp.MustCompile(`"url"\s*:\s*"([^"]+)"`).FindStringSubmatch(pretty); len(m) == 2 {
		return m[1]
	}
	return ""
}

func estimateTokens(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	chars := len([]rune(s))
	est := (chars + 3) / 4
	if words := len(strings.FieldsFunc(s, unicode.IsSpace)); words > est {
		est = words
	}
	return est
}

func looksRefusal(s string) bool {
	low := strings.ToLower(s)
	return strings.Contains(low, "could not find") || strings.Contains(low, "cannot find") ||
		strings.Contains(low, "i don't know") || strings.Contains(low, "not able to") ||
		strings.Contains(low, "no reliable")
}
