package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/web"
)

const (
	defaultReadMaxLines = 200
	defaultReadMaxBytes = 16 * 1024
	maxWriteFileBytes   = 256 * 1024
	maxGlobResults      = 200
	maxGrepMatches      = 100
	defaultCmdTimeout   = 60 * time.Second
	maxCmdTimeout       = 600 * time.Second
	maxCmdOutputBytes   = 24 * 1024
	maxToolResultBytes  = 32 * 1024
)

// ShellKind selects the shell used by run_command.
type ShellKind string

const (
	ShellCmd        ShellKind = "cmd"
	ShellPowerShell ShellKind = "powershell"
)

// AskUserFn pauses the agent loop cleanly to gather a user answer or option choice.
type AskUserFn func(ctx context.Context, question string, options []string) (string, error)

// SkillStagePrompt presents a staged model skill in .alpaca/skills-staging/ for user approval.
type SkillStagePrompt struct {
	Name          string
	Description   string
	Body          string
	Files         map[string]string
	StagingDir    string
	RenderedMD    string
	Diff          string
	DiffAdded     int
	DiffRemoved   int
	Overwrites    bool
	ModelAuthored bool
	TotalBytes    int
	Tainted       bool
}

// SkillStageResponse carries the user's decision (and optional edits/rejection reason) on a staged skill.
type SkillStageResponse struct {
	Approved bool
	Name     string
	Desc     string
	Body     string
	Reason   string
}

// StageSkillFn pauses the agent loop so the user can preview, edit, approve, or reject a staged skill.
type StageSkillFn func(ctx context.Context, pr SkillStagePrompt) (SkillStageResponse, error)

// ToolContext provides shared execution environment for local tools.
type ToolContext struct {
	WorkspaceRoot string
	SkillsDir     string
	SkillsDataDir string
	SessionID     string
	TurnNumber    int
	Checkpoints   *CheckpointStore
	Shell         ShellKind
	Tainted       bool
	ContextTokens int
	AskUser       AskUserFn
	StageSkill    StageSkillFn
}

// ToolExecResult is the structured outcome of running a tool.
type ToolExecResult struct {
	Output      string
	Summary     string
	Diff        string
	DiffAdded   int
	DiffRemoved int
	Mutated     bool
	Tainted     bool
	IsError     bool
	Interrupted bool
}

// ToolSpec defines a single tool available to the LLM agent loop.
type ToolSpec struct {
	Name        string
	Group       string // "files", "exec", "web", "skills", "mcp", "core"
	Class       ToolClass
	Description string
	Parameters  map[string]any
	InspectPerm func(ctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error)
	Execute     func(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult
}

// Registry stores tool definitions and active tool-group toggles.
type Registry struct {
	mu              sync.RWMutex
	tools           map[string]ToolSpec
	order           []string
	groupEnabled    map[string]bool
	webSearchReason string
	webSession      *web.Session
	webChain        *web.Chain
}

// NewDefaultRegistry creates a Registry preloaded with all local tools.
func NewDefaultRegistry() *Registry {
	r := &Registry{
		tools: make(map[string]ToolSpec),
		groupEnabled: map[string]bool{
			"core":   true,
			"files":  true,
			"exec":   true,
			"web":    true,
			"skills": true,
			"mcp":    true,
		},
	}
	for _, spec := range buildLocalTools() {
		r.Register(spec)
	}
	RegisterSkillTools(r)
	return r
}

func (r *Registry) Register(spec ToolSpec) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[spec.Name]; !exists {
		r.order = append(r.order, spec.Name)
		sort.Strings(r.order)
	}
	r.tools[spec.Name] = spec
}

func (r *Registry) Unregister(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[name]; !exists {
		return
	}
	delete(r.tools, name)
	filtered := r.order[:0]
	for _, n := range r.order {
		if n != name {
			filtered = append(filtered, n)
		}
	}
	r.order = filtered
}

func (r *Registry) SetWebSearchStatus(reason string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.webSearchReason = strings.TrimSpace(reason)
}

func (r *Registry) WebSearchStatus() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.webSearchReason
}

// SeedWebSession allowlists URLs from a user message before the model can
// call a web tool.
func (r *Registry) SeedWebSession(text string) {
	r.mu.RLock()
	s := r.webSession
	r.mu.RUnlock()
	s.SeedText(text)
}

// WebSession exposes session state for diagnostics and evaluation.
func (r *Registry) WebSession() *web.Session {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.webSession
}

func (r *Registry) SetGroupEnabled(group string, on bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.groupEnabled[strings.ToLower(strings.TrimSpace(group))] = on
}

func (r *Registry) GroupEnabled(group string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	v, ok := r.groupEnabled[strings.ToLower(strings.TrimSpace(group))]
	if !ok {
		return true
	}
	return v
}

func (r *Registry) GroupsSnapshot() map[string]bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]bool, len(r.groupEnabled))
	for k, v := range r.groupEnabled {
		out[k] = v
	}
	return out
}

// GroupSummary lists active toggleable tool groups (sorted) with tool counts.
type GroupInfo struct {
	Name    string
	Count   int
	Enabled bool
}

func (r *Registry) GroupsOrdered() []GroupInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	counts := make(map[string]int)
	for _, sp := range r.tools {
		if sp.Group != "" && sp.Group != "core" {
			counts[sp.Group]++
		}
	}
	var names []string
	for g, cnt := range counts {
		if cnt > 0 {
			names = append(names, g)
		}
	}
	sort.Strings(names)
	out := make([]GroupInfo, 0, len(names))
	for _, g := range names {
		en, has := r.groupEnabled[g]
		if !has {
			en = true
		}
		out = append(out, GroupInfo{Name: g, Count: counts[g], Enabled: en})
	}
	return out
}

func (r *Registry) Lookup(name string) (ToolSpec, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	sp, ok := r.tools[name]
	if !ok {
		return ToolSpec{}, false
	}
	if en, has := r.groupEnabled[sp.Group]; has && !en {
		return ToolSpec{}, false
	}
	return sp, true
}

// ToolDefs returns enabled tool schemas in deterministic sorted order for KV cache stability.
func (r *Registry) ToolDefs() []runner.ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var defs []runner.ToolDef
	for _, name := range r.order {
		sp := r.tools[name]
		if en, has := r.groupEnabled[sp.Group]; has && !en {
			continue
		}
		defs = append(defs, runner.ToolDef{
			Type: "function",
			Function: runner.ToolFunctionDef{
				Name:        sp.Name,
				Description: sp.Description,
				Parameters:  sp.Parameters,
			},
		})
	}
	return defs
}

func buildLocalTools() []ToolSpec {
	return []ToolSpec{
		{
			Name:        "read_file",
			Group:       "files",
			Class:       ClassRead,
			Description: "Read UTF-8 file contents with 1-based line numbers (capped at 200 lines / 16 KB per call; use start_line/end_line to page).",
			Parameters: schemaObj(map[string]any{
				"path":       schemaStr("Path relative to the workspace root."),
				"start_line": schemaInt("Optional 1-based starting line number."),
				"end_line":   schemaInt("Optional 1-based ending line number."),
			}, "path"),
			InspectPerm: inspectReadPathPerm("read_file", "path"),
			Execute:     execReadFile,
		},
		{
			Name:        "list_dir",
			Group:       "files",
			Class:       ClassRead,
			Description: "List directory entries inside the workspace with types and sizes (depth max 3; hides .alpaca/).",
			Parameters: schemaObj(map[string]any{
				"path":  schemaStr("Directory path relative to workspace root (default '.')."),
				"depth": schemaInt("Optional traversal depth from 1 to 3 (default 1)."),
			}, "path"),
			InspectPerm: inspectReadPathPerm("list_dir", "path"),
			Execute:     execListDir,
		},
		{
			Name:        "glob",
			Group:       "files",
			Class:       ClassRead,
			Description: "Find workspace files matching a glob pattern such as '**/*.go' or 'src/*.ts' (max 200 results; hides .alpaca/).",
			Parameters: schemaObj(map[string]any{
				"pattern": schemaStr("Glob pattern relative to workspace root."),
			}, "pattern"),
			InspectPerm: inspectSimplePerm("glob", ClassRead, "pattern"),
			Execute:     execGlob,
		},
		{
			Name:        "grep",
			Group:       "files",
			Class:       ClassRead,
			Description: "Search UTF-8 workspace files with a regular expression and return file:line matches (max 100).",
			Parameters: schemaObj(map[string]any{
				"pattern":     schemaStr("Regular expression to search for."),
				"path":        schemaStr("Optional file or subfolder path relative to workspace root."),
				"glob":        schemaStr("Optional filename glob filter like '*.go'."),
				"ignore_case": schemaBool("Case-insensitive search if true."),
			}, "pattern"),
			InspectPerm: inspectGrepPerm,
			Execute:     execGrep,
		},
		{
			Name:        "write_file",
			Group:       "files",
			Class:       ClassWrite,
			Description: "Create or overwrite a UTF-8 file inside the workspace (max 256 KB; creates parent dirs).",
			Parameters: schemaObj(map[string]any{
				"path":    schemaStr("Destination path relative to workspace root."),
				"content": schemaStr("Full UTF-8 file content to write."),
			}, "path", "content"),
			InspectPerm: inspectWriteFilePerm,
			Execute:     execWriteFile,
		},
		{
			Name:        "edit_file",
			Group:       "files",
			Class:       ClassWrite,
			Description: "Replace exact text inside an existing workspace file. Errors cleanly if old_text has 0 matches (or >1 without replace_all=true).",
			Parameters: schemaObj(map[string]any{
				"path":        schemaStr("Target path relative to workspace root."),
				"old_text":    schemaStr("Exact substring to find and replace."),
				"new_text":    schemaStr("Replacement substring."),
				"replace_all": schemaBool("Replace all occurrences when true; otherwise requires exactly 1 match."),
			}, "path", "old_text", "new_text"),
			InspectPerm: inspectEditFilePerm,
			Execute:     execEditFile,
		},
		{
			Name:        "make_dir",
			Group:       "files",
			Class:       ClassWrite,
			Description: "Create a directory (and any missing parents) inside the workspace.",
			Parameters: schemaObj(map[string]any{
				"path": schemaStr("Directory path relative to workspace root."),
			}, "path"),
			InspectPerm: inspectWritePathOnlyPerm("make_dir", "path"),
			Execute:     execMakeDir,
		},
		{
			Name:        "move_path",
			Group:       "files",
			Class:       ClassWrite,
			Description: "Move or rename a file or folder within the workspace (refuses to overwrite existing destination unless overwrite=true).",
			Parameters: schemaObj(map[string]any{
				"from":      schemaStr("Source path inside workspace."),
				"to":        schemaStr("Destination path inside workspace."),
				"overwrite": schemaBool("Overwrite existing destination if true (default false)."),
			}, "from", "to"),
			InspectPerm: inspectMovePerm,
			Execute:     execMovePath,
		},
		{
			Name:        "delete_path",
			Group:       "files",
			Class:       ClassWrite,
			Description: "Safely delete a workspace file or directory by moving it to .alpaca/trash/<timestamp>/.",
			Parameters: schemaObj(map[string]any{
				"path": schemaStr("Target path inside workspace to move to trash."),
			}, "path"),
			InspectPerm: inspectWritePathOnlyPerm("delete_path", "path"),
			Execute:     execDeletePath,
		},
		{
			Name:        "run_command",
			Group:       "exec",
			Class:       ClassExec,
			Description: "Run a shell command in the workspace (default timeout 60s, max 600s). Runs with the user's own permissions; side effects are NOT undone by /undo.",
			Parameters: schemaObj(map[string]any{
				"command":   schemaStr("Shell command string to execute."),
				"cwd":       schemaStr("Optional working directory inside workspace."),
				"timeout_s": schemaInt("Optional timeout in seconds (default 60, max 600)."),
			}, "command"),
			InspectPerm: inspectRunCommandPerm,
			Execute:     execRunCommand,
		},
		{
			Name:        "ask_user",
			Group:       "core",
			Class:       ClassNone,
			Description: "Pause agent execution to ask the user a clarifying question (optionally with suggested choices).",
			Parameters: schemaObj(map[string]any{
				"question": schemaStr("Question to present to the user."),
				"options": map[string]any{
					"type":        "array",
					"description": "Optional short answer choices.",
					"items":       map[string]any{"type": "string"},
				},
			}, "question"),
			InspectPerm: inspectSimplePerm("ask_user", ClassNone, "question"),
			Execute:     execAskUser,
		},
	}
}

func schemaObj(props map[string]any, req ...string) map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   req,
	}
}

func schemaStr(desc string) map[string]any {
	return map[string]any{"type": "string", "description": desc}
}
func schemaInt(desc string) map[string]any {
	return map[string]any{"type": "integer", "description": desc}
}
func schemaBool(desc string) map[string]any {
	return map[string]any{"type": "boolean", "description": desc}
}

func strArg(args map[string]any, key string) string {
	v, ok := args[key]
	if !ok || v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func intArg(args map[string]any, key string, def int) int {
	v, ok := args[key]
	if !ok || v == nil {
		return def
	}
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(x)); err == nil {
			return n
		}
	}
	return def
}

func boolArg(args map[string]any, key string) bool {
	v, ok := args[key]
	if !ok || v == nil {
		return false
	}
	if b, ok := v.(bool); ok {
		return b
	}
	if s, ok := v.(string); ok {
		return strings.EqualFold(strings.TrimSpace(s), "true")
	}
	return false
}

// --- Permission Inspection Helpers ---

func resolveReadPathWithSkills(tctx ToolContext, raw string) (string, string, error) {
	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, raw, AccessRead)
	if err == nil {
		if _, statErr := os.Stat(abs); statErr == nil || tctx.SkillsDir == "" {
			return abs, rel, nil
		}
	}
	norm := filepath.ToSlash(strings.TrimSpace(raw))
	low := strings.ToLower(norm)
	if tctx.SkillsDir != "" && (low == "skills" || strings.HasPrefix(low, "skills/")) {
		sub := ""
		if len(norm) > len("skills/") {
			sub = norm[len("skills/"):]
		} else if len(norm) == len("skills") {
			sub = "."
		}
		sAbs, sRel, sErr := ResolveWorkspacePath(tctx.SkillsDir, sub, AccessRead)
		if sErr == nil {
			outRel := "skills"
			if sRel != "." && sRel != "" {
				outRel = "skills/" + sRel
			}
			return sAbs, outRel, nil
		}
		return "", "", sErr
	}
	return abs, rel, err
}

func inspectReadPathPerm(toolName, argKey string) func(ToolContext, map[string]any) (PermissionInput, string, string, int, int, error) {
	return func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
		raw := strArg(args, argKey)
		if raw == "" && toolName == "list_dir" {
			raw = "."
		}
		_, rel, err := resolveReadPathWithSkills(tctx, raw)
		if err != nil {
			return PermissionInput{}, "", "", 0, 0, err
		}
		norm := map[string]any{argKey: rel}
		return PermissionInput{
			Tool:          toolName,
			CanonicalArgs: norm,
			Class:         ClassRead,
			InWorkspace:   true,
			Summary:       rel,
		}, "", "", 0, 0, nil
	}
}

func inspectSimplePerm(toolName string, class ToolClass, summaryKey string) func(ToolContext, map[string]any) (PermissionInput, string, string, int, int, error) {
	return func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
		val := strArg(args, summaryKey)
		return PermissionInput{
			Tool:          toolName,
			CanonicalArgs: canonicalizeMap(args),
			Class:         class,
			InWorkspace:   true,
			Summary:       val,
		}, "", "", 0, 0, nil
	}
}

func inspectGrepPerm(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
	rel := "."
	if p := strArg(args, "path"); p != "" {
		_, r, err := ResolveWorkspacePath(tctx.WorkspaceRoot, p, AccessRead)
		if err != nil {
			return PermissionInput{}, "", "", 0, 0, err
		}
		rel = r
	}
	norm := canonicalizeMap(args)
	norm["path"] = rel
	return PermissionInput{
		Tool:          "grep",
		CanonicalArgs: norm,
		Class:         ClassRead,
		InWorkspace:   true,
		Summary:       fmt.Sprintf("%q in %s", strArg(args, "pattern"), rel),
	}, "", "", 0, 0, nil
}

func inspectWriteFilePerm(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
	rawPath := strArg(args, "path")
	content := strArg(args, "content")
	if len(content) > maxWriteFileBytes {
		return PermissionInput{}, "", "", 0, 0, fmt.Errorf("write_file content (%d bytes) exceeds 256 KB limit", len(content))
	}
	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, rawPath, AccessCreate)
	inWS := !errors.Is(err, ErrOutsideWorkspace) && !errors.Is(err, ErrHostilePath)
	prot := errors.Is(err, ErrProtectedPath)
	if err != nil && !prot {
		return PermissionInput{}, "", "", 0, 0, err
	}
	before := ""
	if b, rErr := os.ReadFile(abs); rErr == nil {
		before = string(b)
	}
	diffText, added, removed := UnifiedDiff(rel, before, content)
	return PermissionInput{
		Tool:          "write_file",
		CanonicalArgs: map[string]any{"path": rel, "content": content},
		Class:         ClassWrite,
		InWorkspace:   inWS,
		ProtectedPath: prot,
		Summary:       fmt.Sprintf("%s (+%d/-%d)", rel, added, removed),
	}, diffText, rel, added, removed, nil
}

func inspectEditFilePerm(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
	rawPath := strArg(args, "path")
	oldText := strArg(args, "old_text")
	newText := strArg(args, "new_text")
	replaceAll := boolArg(args, "replace_all")

	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, rawPath, AccessWrite)
	inWS := !errors.Is(err, ErrOutsideWorkspace) && !errors.Is(err, ErrHostilePath)
	prot := errors.Is(err, ErrProtectedPath)
	if err != nil && !prot {
		return PermissionInput{}, "", "", 0, 0, err
	}
	rawBytes, rErr := os.ReadFile(abs)
	if rErr != nil {
		return PermissionInput{}, "", "", 0, 0, rErr
	}
	if !isTextUTF8(rawBytes) {
		return PermissionInput{}, "", "", 0, 0, fmt.Errorf("edit_file refuses binary or non-UTF-8 file %s", rel)
	}
	before := string(rawBytes)
	after, editErr := applyExactEdit(before, oldText, newText, replaceAll)
	if editErr != nil {
		return PermissionInput{}, "", "", 0, 0, editErr
	}
	diffText, added, removed := UnifiedDiff(rel, before, after)
	return PermissionInput{
		Tool:          "edit_file",
		CanonicalArgs: map[string]any{"path": rel, "old_text": oldText, "new_text": newText, "replace_all": replaceAll},
		Class:         ClassWrite,
		InWorkspace:   inWS,
		ProtectedPath: prot,
		Summary:       fmt.Sprintf("%s (+%d/-%d)", rel, added, removed),
	}, diffText, rel, added, removed, nil
}

func inspectWritePathOnlyPerm(toolName, key string) func(ToolContext, map[string]any) (PermissionInput, string, string, int, int, error) {
	return func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
		_, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, key), AccessCreate)
		inWS := !errors.Is(err, ErrOutsideWorkspace) && !errors.Is(err, ErrHostilePath)
		prot := errors.Is(err, ErrProtectedPath)
		if err != nil && !prot {
			return PermissionInput{}, "", "", 0, 0, err
		}
		return PermissionInput{
			Tool:          toolName,
			CanonicalArgs: map[string]any{key: rel},
			Class:         ClassWrite,
			InWorkspace:   inWS,
			ProtectedPath: prot,
			Summary:       rel,
		}, "", "", 0, 0, nil
	}
}

func inspectMovePerm(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
	_, fromRel, err1 := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "from"), AccessWrite)
	if err1 != nil && !errors.Is(err1, ErrProtectedPath) {
		return PermissionInput{}, "", "", 0, 0, err1
	}
	_, toRel, err2 := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "to"), AccessCreate)
	if err2 != nil && !errors.Is(err2, ErrProtectedPath) {
		return PermissionInput{}, "", "", 0, 0, err2
	}
	prot := errors.Is(err1, ErrProtectedPath) || errors.Is(err2, ErrProtectedPath)
	return PermissionInput{
		Tool:          "move_path",
		CanonicalArgs: map[string]any{"from": fromRel, "to": toRel, "overwrite": boolArg(args, "overwrite")},
		Class:         ClassWrite,
		InWorkspace:   true,
		ProtectedPath: prot,
		Summary:       fromRel + " -> " + toRel,
	}, "", "", 0, 0, nil
}

func inspectRunCommandPerm(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
	command := strings.TrimSpace(strArg(args, "command"))
	if command == "" {
		return PermissionInput{}, "", "", 0, 0, errors.New("command cannot be empty")
	}
	cwdRel := "."
	if rawCwd := strArg(args, "cwd"); rawCwd != "" {
		_, r, err := ResolveWorkspacePath(tctx.WorkspaceRoot, rawCwd, AccessRead)
		if err != nil {
			return PermissionInput{}, "", "", 0, 0, err
		}
		cwdRel = r
	}
	highRisk, riskMatch := IsHighRiskCommand(command)
	sum := command
	if highRisk && riskMatch != "" {
		sum = "[HIGH RISK: " + riskMatch + "] " + command
	}
	return PermissionInput{
		Tool:          "run_command",
		CanonicalArgs: map[string]any{"command": command, "cwd": cwdRel},
		Class:         ClassExec,
		InWorkspace:   true,
		HighRisk:      highRisk,
		Summary:       sum,
	}, "", "", 0, 0, nil
}

// --- Local Tool Implementations ---

func execReadFile(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	abs, rel, err := resolveReadPathWithSkills(tctx, strArg(args, "path"))
	if err != nil {
		return errRes(err)
	}
	st, err := os.Stat(abs)
	if err != nil {
		return errRes(err)
	}
	if st.IsDir() {
		return errRes(fmt.Errorf("%s is a directory; use list_dir instead", rel))
	}

	raw, err := os.ReadFile(abs)
	if err != nil {
		return errRes(err)
	}
	head := raw
	if len(head) > 8192 {
		head = head[:8192]
	}
	if bytes.IndexByte(head, 0) >= 0 || !utf8.Valid(raw) {
		return errRes(fmt.Errorf("refusing binary or non-UTF-8 file %s (%d bytes)", rel, len(raw)))
	}

	norm := strings.ReplaceAll(string(raw), "\r\n", "\n")
	lines := strings.Split(norm, "\n")
	totalLines := len(lines)

	startLine := intArg(args, "start_line", 1)
	if startLine < 1 {
		startLine = 1
	}
	endLine := intArg(args, "end_line", startLine+defaultReadMaxLines-1)
	if endLine > startLine+defaultReadMaxLines-1 {
		endLine = startLine + defaultReadMaxLines - 1
	}
	if endLine > totalLines {
		endLine = totalLines
	}
	if startLine > totalLines {
		return errRes(fmt.Errorf("start_line %d exceeds total lines (%d) in %s", startLine, totalLines, rel))
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("File: %s (lines %d-%d of %d)\n", rel, startLine, endLine, totalLines))
	writtenBytes := 0
	lastShown := startLine - 1
	for ln := startLine; ln <= endLine; ln++ {
		entry := fmt.Sprintf("%4d | %s\n", ln, lines[ln-1])
		if writtenBytes+len(entry) > defaultReadMaxBytes && ln > startLine {
			break
		}
		b.WriteString(entry)
		writtenBytes += len(entry)
		lastShown = ln
	}
	if lastShown < totalLines {
		b.WriteString(fmt.Sprintf("... truncated at line %d of %d. Call read_file with start_line=%d to continue.", lastShown, totalLines, lastShown+1))
	}
	return ToolExecResult{Output: b.String(), Summary: rel}
}

func execListDir(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	rawPath := strArg(args, "path")
	if rawPath == "" {
		rawPath = "."
	}
	abs, rel, err := resolveReadPathWithSkills(tctx, rawPath)
	if err != nil {
		return errRes(err)
	}
	depth := intArg(args, "depth", 1)
	if depth < 1 {
		depth = 1
	}
	if depth > 3 {
		depth = 3
	}

	var lines []string
	var walk func(dir, relBase string, curDepth int)
	walk = func(dir, relBase string, curDepth int) {
		if ctx.Err() != nil {
			return
		}
		des, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		sort.Slice(des, func(i, j int) bool {
			if des[i].IsDir() != des[j].IsDir() {
				return des[i].IsDir()
			}
			return strings.ToLower(des[i].Name()) < strings.ToLower(des[j].Name())
		})
		for _, de := range des {
			name := de.Name()
			childRel := name
			if relBase != "." && relBase != "" {
				childRel = relBase + "/" + name
			}
			if IsHiddenInternal(childRel) || name == ".alpaca" || name == ".llm" {
				continue
			}
			indent := strings.Repeat("  ", curDepth-1)
			if de.IsDir() {
				lines = append(lines, fmt.Sprintf("%s[dir]  %s/", indent, name))
				if curDepth < depth {
					walk(filepath.Join(dir, name), childRel, curDepth+1)
				}
			} else {
				sz := int64(0)
				if info, iErr := de.Info(); iErr == nil {
					sz = info.Size()
				}
				lines = append(lines, fmt.Sprintf("%s[file] %s (%d B)", indent, name, sz))
			}
		}
	}
	walk(abs, rel, 1)
	if len(lines) == 0 {
		return ToolExecResult{Output: fmt.Sprintf("Directory %s is empty.", rel), Summary: rel}
	}
	return ToolExecResult{Output: strings.Join(lines, "\n"), Summary: fmt.Sprintf("%s (%d entries)", rel, len(lines))}
}

func execGlob(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	pattern := filepath.ToSlash(strings.TrimSpace(strArg(args, "pattern")))
	if pattern == "" {
		return errRes(errors.New("pattern cannot be empty"))
	}
	canonRoot, err := canonicalRoot(tctx.WorkspaceRoot)
	if err != nil {
		return errRes(err)
	}

	var matches []string
	_ = filepath.WalkDir(canonRoot, func(p string, d os.DirEntry, wErr error) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if wErr != nil {
			return nil
		}
		rel, rErr := filepath.Rel(canonRoot, p)
		if rErr != nil {
			return nil
		}
		relSlash := filepath.ToSlash(rel)
		if relSlash == "." {
			return nil
		}
		if IsHiddenInternal(relSlash) || d.Name() == ".git" {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if matchGlobPattern(pattern, relSlash) {
			matches = append(matches, relSlash)
			if len(matches) >= maxGlobResults {
				return io.EOF
			}
		}
		return nil
	})

	sort.Strings(matches)
	out := fmt.Sprintf("Found %d path(s) matching %q", len(matches), pattern)
	if len(matches) >= maxGlobResults {
		out += fmt.Sprintf(" (capped at %d)", maxGlobResults)
	}
	if len(matches) > 0 {
		out += ":\n" + strings.Join(matches, "\n")
	}
	return ToolExecResult{Output: out, Summary: fmt.Sprintf("%s (%d)", pattern, len(matches))}
}

func matchGlobPattern(pattern, relSlash string) bool {
	// Convert ** glob cleanly to regex so recursive patterns work identically everywhere.
	var re strings.Builder
	re.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		if i+1 < len(pattern) && pattern[i] == '*' && pattern[i+1] == '*' {
			if i+2 < len(pattern) && pattern[i+2] == '/' {
				re.WriteString("(?:.*/)?")
				i += 2
			} else {
				re.WriteString(".*")
				i++
			}
			continue
		}
		switch pattern[i] {
		case '*':
			re.WriteString(`[^/]*`)
		case '?':
			re.WriteString(`[^/]`)
		default:
			re.WriteString(regexp.QuoteMeta(pattern[i : i+1]))
		}
	}
	re.WriteByte('$')
	compiled, err := regexp.Compile("(?i)" + re.String())
	if err != nil {
		ok, _ := filepath.Match(pattern, filepath.Base(relSlash))
		return ok
	}
	return compiled.MatchString(relSlash)
}

func execGrep(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	pat := strArg(args, "pattern")
	if pat == "" {
		return errRes(errors.New("pattern cannot be empty"))
	}
	if boolArg(args, "ignore_case") {
		pat = "(?i)" + pat
	}
	re, err := regexp.Compile(pat)
	if err != nil {
		return errRes(fmt.Errorf("invalid regex: %w", err))
	}

	rawStart := strArg(args, "path")
	if rawStart == "" {
		rawStart = "."
	}
	startAbs, startRel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, rawStart, AccessRead)
	if err != nil {
		return errRes(err)
	}
	canonRoot, _ := canonicalRoot(tctx.WorkspaceRoot)
	globFilter := strArg(args, "glob")

	var hits []string
	_ = filepath.WalkDir(startAbs, func(p string, d os.DirEntry, wErr error) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if wErr != nil {
			return nil
		}
		rel, rErr := filepath.Rel(canonRoot, p)
		if rErr != nil {
			return nil
		}
		relSlash := filepath.ToSlash(rel)
		if IsHiddenInternal(relSlash) || d.Name() == ".git" {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if globFilter != "" && !matchGlobPattern(globFilter, filepath.Base(relSlash)) && !matchGlobPattern(globFilter, relSlash) {
			return nil
		}
		raw, fErr := os.ReadFile(p)
		if fErr != nil || !isTextUTF8(raw) {
			return nil
		}
		lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
		for idx, ln := range lines {
			if re.MatchString(ln) {
				hits = append(hits, fmt.Sprintf("%s:%d: %s", relSlash, idx+1, truncateErrLine(ln, 240)))
				if len(hits) >= maxGrepMatches {
					return io.EOF
				}
			}
		}
		return nil
	})

	if len(hits) == 0 {
		return ToolExecResult{Output: fmt.Sprintf("No matches for %q in %s.", pat, startRel), Summary: "0 matches"}
	}
	hdr := fmt.Sprintf("Found %d match(es)", len(hits))
	if len(hits) >= maxGrepMatches {
		hdr += fmt.Sprintf(" (capped at %d)", maxGrepMatches)
	}
	return ToolExecResult{Output: hdr + ":\n" + strings.Join(hits, "\n"), Summary: fmt.Sprintf("%d matches", len(hits))}
}

func execWriteFile(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	content := strArg(args, "content")
	if len(content) > maxWriteFileBytes {
		return errRes(fmt.Errorf("write_file content (%d B) exceeds 256 KB limit", len(content)))
	}
	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "path"), AccessCreate)
	if err != nil {
		return errRes(err)
	}
	if err := tctx.Checkpoints.CaptureBefore(tctx.TurnNumber, abs, rel); err != nil {
		return errRes(fmt.Errorf("checkpoint failed: %w", err))
	}
	before := ""
	if b, rErr := os.ReadFile(abs); rErr == nil {
		before = string(b)
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return errRes(err)
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		return errRes(err)
	}
	diffText, added, removed := UnifiedDiff(rel, before, content)
	return ToolExecResult{
		Output:      fmt.Sprintf("Wrote %s (%d bytes, +%d/-%d lines)\n%s", rel, len(content), added, removed, diffText),
		Summary:     fmt.Sprintf("%s (+%d/-%d)", rel, added, removed),
		Diff:        diffText,
		DiffAdded:   added,
		DiffRemoved: removed,
		Mutated:     true,
	}
}

func execEditFile(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "path"), AccessWrite)
	if err != nil {
		return errRes(err)
	}
	raw, err := os.ReadFile(abs)
	if err != nil {
		return errRes(err)
	}
	if !isTextUTF8(raw) {
		return errRes(fmt.Errorf("cannot edit binary file %s", rel))
	}
	before := string(raw)
	oldText := strArg(args, "old_text")
	newText := strArg(args, "new_text")
	replaceAll := boolArg(args, "replace_all")

	after, err := applyExactEdit(before, oldText, newText, replaceAll)
	if err != nil {
		return errRes(err)
	}
	if err := tctx.Checkpoints.CaptureBefore(tctx.TurnNumber, abs, rel); err != nil {
		return errRes(fmt.Errorf("checkpoint failed: %w", err))
	}
	if err := os.WriteFile(abs, []byte(after), 0o644); err != nil {
		return errRes(err)
	}
	diffText, added, removed := UnifiedDiff(rel, before, after)
	return ToolExecResult{
		Output:      fmt.Sprintf("Edited %s (+%d/-%d lines)\n%s", rel, added, removed, diffText),
		Summary:     fmt.Sprintf("%s (+%d/-%d)", rel, added, removed),
		Diff:        diffText,
		DiffAdded:   added,
		DiffRemoved: removed,
		Mutated:     true,
	}
}

func applyExactEdit(before, oldText, newText string, replaceAll bool) (string, error) {
	if oldText == "" {
		return "", errors.New("old_text cannot be empty")
	}
	count := strings.Count(before, oldText)
	if count == 0 {
		return "", errors.New("old_text matched 0 times (exact match required)")
	}
	if count > 1 && !replaceAll {
		return "", fmt.Errorf("old_text matched %d times; provide a more unique snippet or set replace_all=true", count)
	}
	if replaceAll {
		return strings.ReplaceAll(before, oldText, newText), nil
	}
	return strings.Replace(before, oldText, newText, 1), nil
}

func execMakeDir(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "path"), AccessCreate)
	if err != nil {
		return errRes(err)
	}
	if err := tctx.Checkpoints.CaptureBefore(tctx.TurnNumber, abs, rel); err != nil {
		return errRes(fmt.Errorf("checkpoint failed: %w", err))
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return errRes(err)
	}
	return ToolExecResult{Output: "Created directory " + rel, Summary: rel, Mutated: true}
}

func execMovePath(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	fromAbs, fromRel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "from"), AccessWrite)
	if err != nil {
		return errRes(err)
	}
	toAbs, toRel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "to"), AccessCreate)
	if err != nil {
		return errRes(err)
	}
	if _, err := os.Stat(fromAbs); err != nil {
		return errRes(err)
	}
	overwrite := boolArg(args, "overwrite")
	if _, err := os.Stat(toAbs); err == nil && !overwrite {
		return errRes(fmt.Errorf("destination %s already exists (pass overwrite=true to replace)", toRel))
	}
	if err := tctx.Checkpoints.CaptureBefore(tctx.TurnNumber, fromAbs, fromRel); err != nil {
		return errRes(err)
	}
	if err := tctx.Checkpoints.CaptureBefore(tctx.TurnNumber, toAbs, toRel); err != nil {
		return errRes(err)
	}
	if err := os.MkdirAll(filepath.Dir(toAbs), 0o755); err != nil {
		return errRes(err)
	}
	if overwrite {
		_ = os.RemoveAll(toAbs)
	}
	if err := os.Rename(fromAbs, toAbs); err != nil {
		return errRes(err)
	}
	return ToolExecResult{Output: fmt.Sprintf("Moved %s -> %s", fromRel, toRel), Summary: fromRel + " -> " + toRel, Mutated: true}
}

func execDeletePath(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	abs, rel, err := ResolveWorkspacePath(tctx.WorkspaceRoot, strArg(args, "path"), AccessWrite)
	if err != nil {
		return errRes(err)
	}
	if _, err := os.Stat(abs); err != nil {
		return errRes(err)
	}
	if err := tctx.Checkpoints.CaptureBefore(tctx.TurnNumber, abs, rel); err != nil {
		return errRes(err)
	}
	stamp := time.Now().Format("20060102-150405.000")
	trashDest := filepath.Join(tctx.WorkspaceRoot, ".alpaca", "trash", stamp, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(trashDest), 0o755); err != nil {
		return errRes(err)
	}
	if err := os.Rename(abs, trashDest); err != nil {
		return errRes(err)
	}
	trashRel, _ := filepath.Rel(tctx.WorkspaceRoot, trashDest)
	return ToolExecResult{
		Output:  fmt.Sprintf("Moved %s to trash (%s); use /undo to restore.", rel, filepath.ToSlash(trashRel)),
		Summary: rel + " -> trash",
		Mutated: true,
	}
}

func execRunCommand(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	command := strings.TrimSpace(strArg(args, "command"))
	if command == "" {
		return errRes(errors.New("command cannot be empty"))
	}
	workDir := tctx.WorkspaceRoot
	if rawCwd := strArg(args, "cwd"); rawCwd != "" {
		abs, _, err := ResolveWorkspacePath(tctx.WorkspaceRoot, rawCwd, AccessRead)
		if err != nil {
			return errRes(err)
		}
		workDir = abs
	}

	timeout := time.Duration(intArg(args, "timeout_s", int(defaultCmdTimeout.Seconds()))) * time.Second
	if timeout <= 0 {
		timeout = defaultCmdTimeout
	}
	if timeout > maxCmdTimeout {
		timeout = maxCmdTimeout
	}

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var cmd *exec.Cmd
	if tctx.Shell == ShellPowerShell {
		cmd = exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command)
	} else {
		cmd = exec.Command("cmd.exe", "/d", "/s", "/c", command)
	}
	cmd.Dir = workDir
	cmd.Stdin = nil

	var outBuf cappedHeadTailBuffer
	outBuf.maxBytes = maxCmdOutputBytes
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	if err := cmd.Start(); err != nil {
		return errRes(err)
	}

	var job *runner.JobHandle
	if cmd.Process != nil {
		if j, jErr := runner.AttachJob(uint32(cmd.Process.Pid)); jErr == nil {
			job = j
			defer job.Close()
		}
	}

	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()

	var waitErr error
	var canceledByParent, timedOut bool
	select {
	case waitErr = <-waitCh:
	case <-runCtx.Done():
		canceledByParent = ctx.Err() != nil
		timedOut = runCtx.Err() == context.DeadlineExceeded && !canceledByParent
		if job != nil {
			_ = job.Terminate()
		} else if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		waitErr = <-waitCh
	}

	outStr := outBuf.String()
	if canceledByParent {
		return ToolExecResult{
			Output:      strings.TrimSpace(outStr + "\n[command interrupted by user]"),
			Summary:     "interrupted",
			IsError:     true,
			Interrupted: true,
		}
	}
	if timedOut {
		return ToolExecResult{
			Output:  strings.TrimSpace(outStr + fmt.Sprintf("\n[command timed out after %s and process tree was terminated]", timeout)),
			Summary: "timeout",
			IsError: true,
		}
	}
	if waitErr != nil {
		return ToolExecResult{
			Output:  strings.TrimSpace(outStr + fmt.Sprintf("\n[exit error: %v]", waitErr)),
			Summary: waitErr.Error(),
			IsError: true,
		}
	}
	if outStr == "" {
		outStr = "(no output; exit code 0)"
	}
	return ToolExecResult{Output: outStr, Summary: "exit 0"}
}

func execAskUser(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	q := strings.TrimSpace(strArg(args, "question"))
	if q == "" {
		return errRes(errors.New("question cannot be empty"))
	}
	var opts []string
	if rawOpts, ok := args["options"].([]any); ok {
		for _, o := range rawOpts {
			if s := strings.TrimSpace(fmt.Sprintf("%v", o)); s != "" {
				opts = append(opts, s)
			}
		}
	}
	if tctx.AskUser == nil {
		return errRes(errors.New("ask_user callback unavailable in non-interactive mode"))
	}
	ans, err := tctx.AskUser(ctx, q, opts)
	if err != nil {
		return ToolExecResult{
			Output:      "User did not answer (" + err.Error() + ").",
			Summary:     err.Error(),
			IsError:     true,
			Interrupted: ctx.Err() != nil,
		}
	}
	return ToolExecResult{Output: ans, Summary: truncateErrLine(ans, 48)}
}

// cappedHeadTailBuffer keeps head + tail bytes up to maxBytes and notes how much was omitted.
type cappedHeadTailBuffer struct {
	mu       sync.Mutex
	head     []byte
	tail     []byte
	total    int64
	maxBytes int
}

func (c *cappedHeadTailBuffer) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	half := c.maxBytes / 2
	if half <= 0 {
		half = 12 * 1024
	}
	c.total += int64(len(p))

	if len(c.head) < half {
		need := half - len(c.head)
		if len(p) <= need {
			c.head = append(c.head, p...)
			return len(p), nil
		}
		c.head = append(c.head, p[:need]...)
		p = p[need:]
	}
	c.tail = append(c.tail, p...)
	if len(c.tail) > half {
		c.tail = append([]byte(nil), c.tail[len(c.tail)-half:]...)
	}
	return len(p), nil
}

func (c *cappedHeadTailBuffer) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	kept := int64(len(c.head) + len(c.tail))
	if c.total <= kept {
		return string(c.head) + string(c.tail)
	}
	omitted := c.total - kept
	return fmt.Sprintf("%s\n... [%d bytes omitted (%d total bytes)] ...\n%s", string(c.head), omitted, c.total, string(c.tail))
}

func errRes(err error) ToolExecResult {
	return ToolExecResult{Output: "Error: " + err.Error(), Summary: err.Error(), IsError: true}
}

func isTextUTF8(b []byte) bool {
	head := b
	if len(head) > 8192 {
		head = head[:8192]
	}
	if bytes.IndexByte(head, 0) >= 0 {
		return false
	}
	return utf8.Valid(b)
}

func truncateErrLine(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "…"
}

func clampToolOutput(s string) string {
	if len(s) <= maxToolResultBytes {
		return s
	}
	head := maxToolResultBytes * 2 / 3
	tail := maxToolResultBytes - head - 128
	return fmt.Sprintf("%s\n... [%d chars omitted] ...\n%s", s[:head], len(s)-head-tail, s[len(s)-tail:])
}
