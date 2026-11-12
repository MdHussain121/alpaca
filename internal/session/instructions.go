package session

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

const (
	MaxAgentsFileBytes  = 12 * 1024
	MaxAgentsTotalBytes = 16 * 1024
)

// AgentsMDResult contains loaded global and workspace AGENTS.md instructions.
type AgentsMDResult struct {
	GlobalPath       string
	WorkspacePath    string
	GlobalContent    string
	WorkspaceContent string
	CombinedPrompt   string
	TokensEst        int
	Truncated        bool
}

// LoadAgentsMD reads optional <appRoot>/AGENTS.md and <workDir>/AGENTS.md dynamically
// so edits take effect on the very next turn.
func LoadAgentsMD(appRoot, workDir string) AgentsMDResult {
	var res AgentsMDResult
	appAbs := cleanAbs(appRoot)
	wsAbs := cleanAbs(workDir)

	var parts []string
	totalLen := 0

	readOne := func(label, path string) (content string, used bool) {
		raw, err := os.ReadFile(path)
		if err != nil {
			return "", false
		}
		if !utf8.Valid(raw) {
			return "", false
		}
		text := strings.TrimSpace(strings.ReplaceAll(string(raw), "\r\n", "\n"))
		if text == "" {
			return "", false
		}
		if len(text) > MaxAgentsFileBytes {
			text = strings.TrimSpace(text[:MaxAgentsFileBytes]) + "\n… [truncated]"
			res.Truncated = true
		}
		rem := MaxAgentsTotalBytes - totalLen
		if rem <= 128 {
			res.Truncated = true
			return "", false
		}
		if len(text) > rem {
			text = strings.TrimSpace(text[:rem]) + "\n… [truncated]"
			res.Truncated = true
		}
		totalLen += len(text)
		parts = append(parts, fmt.Sprintf("### %s AGENTS.md (%s)\n%s", label, filepath.ToSlash(path), text))
		return text, true
	}

	var globalCandidate string
	if appAbs != "" {
		p := filepath.Join(appAbs, "AGENTS.md")
		if content, ok := readOne("Global", p); ok {
			res.GlobalPath = p
			res.GlobalContent = content
		}
		globalCandidate = p
	}

	if wsAbs != "" {
		wsCandidate := filepath.Join(wsAbs, "AGENTS.md")
		if !sameExistingPath(globalCandidate, wsCandidate) {
			if content, ok := readOne("Workspace", wsCandidate); ok {
				res.WorkspacePath = wsCandidate
				res.WorkspaceContent = content
			}
		}
	}

	if len(parts) > 0 {
		res.CombinedPrompt = "<project_instructions>\n" + strings.Join(parts, "\n\n") + "\n</project_instructions>"
		res.TokensEst = (len(res.CombinedPrompt) + 3) / 4
	}
	return res
}

func cleanAbs(p string) string {
	s := strings.TrimSpace(p)
	if s == "" {
		return ""
	}
	if abs, err := filepath.Abs(s); err == nil {
		return filepath.Clean(abs)
	}
	return filepath.Clean(s)
}

func sameExistingPath(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	if strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) {
		return true
	}
	sa, errA := os.Stat(a)
	sb, errB := os.Stat(b)
	return errA == nil && errB == nil && os.SameFile(sa, sb)
}
