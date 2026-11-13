package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// PermissionMode controls baseline approval behavior for agent tools.
type PermissionMode string

const (
	ModeAsk           PermissionMode = "ask"
	ModeReadAuto      PermissionMode = "read-auto"
	ModeWorkspaceAuto PermissionMode = "workspace-auto"
)

// ToolClass categorizes the privilege impact of a tool call.
type ToolClass string

const (
	ClassNone  ToolClass = "none"
	ClassRead  ToolClass = "read"
	ClassWrite ToolClass = "write"
	ClassNet   ToolClass = "net"
	ClassExec  ToolClass = "exec"
)

// Decision is the permission engine verdict for a proposed tool invocation.
type Decision string

const (
	DecisionAllow Decision = "allow"
	DecisionAsk   Decision = "ask"
	DecisionDeny  Decision = "deny"
)

// AllowlistEntry represents one approved tool for the session. Entries are either
// scoped to one normalized argument signature or to every call of a tool.
type AllowlistEntry struct {
	Key     string
	Tool    string
	Summary string
	AllArgs bool
}

// PermissionInput describes everything needed to evaluate a single tool invocation.
type PermissionInput struct {
	Tool          string
	CanonicalArgs map[string]any
	Class         ToolClass
	InWorkspace   bool
	ProtectedPath bool
	Tainted       bool
	HighRisk      bool
	Summary       string
}

// PermissionEngine evaluates tool permissions deterministically from mode, class,
// path containment, session taint, high-risk flags, and the session allowlist.
type PermissionEngine struct {
	mu        sync.RWMutex
	mode      PermissionMode
	allowlist map[string]AllowlistEntry
	order     []string
}

// NewPermissionEngine constructs an engine defaulting to ModeReadAuto.
func NewPermissionEngine(mode PermissionMode) *PermissionEngine {
	if mode == "" {
		mode = ModeReadAuto
	}
	return &PermissionEngine{
		mode:      mode,
		allowlist: make(map[string]AllowlistEntry),
	}
}

func (p *PermissionEngine) Mode() PermissionMode {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.mode
}

func (p *PermissionEngine) SetMode(mode PermissionMode) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if mode != "" {
		p.mode = mode
	}
}

func (p *PermissionEngine) Allowlist() []AllowlistEntry {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]AllowlistEntry, 0, len(p.order))
	for _, k := range p.order {
		if e, ok := p.allowlist[k]; ok {
			out = append(out, e)
		}
	}
	return out
}

func (p *PermissionEngine) ClearAllowlist() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.allowlist = make(map[string]AllowlistEntry)
	p.order = nil
}

func (p *PermissionEngine) AllowSession(tool string, args map[string]any, summary string) {
	key := SignatureKey(tool, args)
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, exists := p.allowlist[key]; !exists {
		p.order = append(p.order, key)
	}
	p.allowlist[key] = AllowlistEntry{
		Key:     key,
		Tool:    tool,
		Summary: summary,
	}
}

// AllowToolSession approves every call to a tool for the rest of the session,
// regardless of its arguments. This backs the "always allow" prompt choice.
func (p *PermissionEngine) AllowToolSession(tool string) {
	tool = strings.ToLower(strings.TrimSpace(tool))
	if tool == "" {
		return
	}
	key := "tool:" + tool
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, exists := p.allowlist[key]; !exists {
		p.order = append(p.order, key)
	}
	p.allowlist[key] = AllowlistEntry{
		Key:     key,
		Tool:    tool,
		Summary: "always allow (all calls this session)",
		AllArgs: true,
	}
}

func (p *PermissionEngine) toolAllowedLocked(tool string) bool {
	_, ok := p.allowlist["tool:"+strings.ToLower(strings.TrimSpace(tool))]
	return ok
}

// Evaluate decides allow / ask / deny from (mode, class, workspace, protected, tainted, highRisk, allowlist).
func (p *PermissionEngine) Evaluate(in PermissionInput) (Decision, string) {
	p.mu.RLock()
	mode := p.mode
	sig := SignatureKey(in.Tool, in.CanonicalArgs)
	_, allowlisted := p.allowlist[sig]
	toolAllowlisted := p.toolAllowedLocked(in.Tool)
	p.mu.RUnlock()

	if in.ProtectedPath && (in.Class == ClassWrite || in.Class == ClassExec) {
		return DecisionDeny, "protected internal path (.llm/ or skills/) is read-only"
	}

	if in.Class == ClassWrite && !in.InWorkspace {
		return DecisionDeny, "write target is outside the workspace sandbox"
	}

	if in.HighRisk {
		return DecisionAsk, "high-risk command pattern always requires confirmation"
	}

	if in.Class == ClassNone {
		return DecisionAllow, ""
	}

	if (allowlisted || toolAllowlisted) && !in.Tainted {
		return DecisionAllow, "matched session allowlist"
	}

	// Once a session is TAINTED (untrusted web content read),
	// writes/exec/network always require explicit confirmation even in workspace-auto.
	if in.Tainted && (in.Class == ClassWrite || in.Class == ClassExec || in.Class == ClassNet) {
		return DecisionAsk, "session TAINTED by untrusted content — explicit approval required"
	}

	switch mode {
	case ModeAsk:
		return DecisionAsk, "mode ask prompts for tool execution"
	case ModeReadAuto:
		if in.Class == ClassRead {
			return DecisionAllow, ""
		}
		return DecisionAsk, "mode read-auto asks before mutating/network/exec tools"
	case ModeWorkspaceAuto:
		if in.Class == ClassRead || (in.Class == ClassWrite && in.InWorkspace) {
			return DecisionAllow, ""
		}
		return DecisionAsk, "mode workspace-auto asks before exec/network tools"
	default:
		return DecisionAsk, "unknown permission mode"
	}
}

// SignatureKey builds a stable normalized key for (tool, canonicalArgs).
func SignatureKey(tool string, args map[string]any) string {
	b, _ := json.Marshal(canonicalizeMap(args))
	sum := sha256.Sum256(append([]byte(strings.ToLower(strings.TrimSpace(tool))+":"), b...))
	return hex.EncodeToString(sum[:12])
}

func canonicalizeMap(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make(map[string]any, len(m))
	for _, k := range keys {
		out[k] = m[k]
	}
	return out
}

var highRiskCommandPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bformat\s+[a-z]:`),
	regexp.MustCompile(`(?i)\bdiskpart\b`),
	regexp.MustCompile(`(?i)\brmdir\s+/s\b|\brd\s+/s\b|\bdel\s+/[fsq]*\s*\*`),
	regexp.MustCompile(`(?i)\bremove-item\b[^\n]*-recurse[^\n]*-force|\brm\s+-[a-z]*rf\b`),
	regexp.MustCompile(`(?i)\breg\s+(add|delete|import)\b`),
	regexp.MustCompile(`(?i)\bnet\s+(user|localgroup)\b|\bnetsh\b`),
	regexp.MustCompile(`(?i)\bshutdown\b|\brestart-computer\b|\bstop-computer\b`),
	regexp.MustCompile(`(?i)\bpowershell(?:\.exe)?\b[^\n]*-(?:e|ec|enc|encodedcommand)\b`),
	regexp.MustCompile(`(?i)\b(?:curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod)\b[^\n]*\|\s*(?:cmd|sh|bash|powershell|pwsh|iex|invoke-expression)\b`),
	regexp.MustCompile(`(?i)\b(?:iex|invoke-expression)\s*\([^\n]*(?:downloadstring|iwr|irm|curl)`),
	regexp.MustCompile(`(?i)\bbcdedit\b|\bvssadmin\s+delete\b|\bcipher\s+/w\b`),
}

// IsHighRiskCommand reports whether command matches destructive or interpreter-pipe patterns
// that must require user confirmation even in workspace-auto mode.
func IsHighRiskCommand(cmd string) (bool, string) {
	for _, re := range highRiskCommandPatterns {
		if m := re.FindString(cmd); m != "" {
			return true, strings.TrimSpace(m)
		}
	}
	return false, ""
}
