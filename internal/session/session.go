package session

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/MdHussain121/alpaca/internal/runner"
)

var validIDRegexp = regexp.MustCompile(`^[A-Za-z0-9._-]{1,80}$`)

// ToolCallRecord stores a rendered tool call card inside a saved session turn.
type ToolCallRecord struct {
	CallID      string `json:"call_id,omitempty"`
	Name        string `json:"name"`
	Summary     string `json:"summary,omitempty"`
	ArgsPretty  string `json:"args_pretty,omitempty"`
	Output      string `json:"output,omitempty"`
	Diff        string `json:"diff,omitempty"`
	DiffAdded   int    `json:"diff_added,omitempty"`
	DiffRemoved int    `json:"diff_removed,omitempty"`
	Status      string `json:"status"`
	DurationMs  int64  `json:"duration_ms,omitempty"`
}

// AttachmentRecord preserves @file attachments attached to a user turn.
type AttachmentRecord struct {
	Path      string `json:"path"`
	LineRange string `json:"line_range,omitempty"`
	Bytes     int    `json:"bytes"`
	Tokens    int    `json:"tokens"`
	Content   string `json:"content"`
}

// BranchTurn represents one persisted turn on a specific branch in JSONL.
type BranchTurn struct {
	TurnID              string              `json:"turn_id"`
	BranchID            string              `json:"branch_id"`
	Role                string              `json:"role"` // "user", "assistant", "info"
	Reasoning           string              `json:"reasoning,omitempty"`
	Content             string              `json:"content"`
	Attachments         []AttachmentRecord  `json:"attachments,omitempty"`
	ToolCalls           []ToolCallRecord    `json:"tool_calls,omitempty"`
	APIMessages         []runner.APIMessage `json:"api_messages,omitempty"`
	Tokens              int                 `json:"tokens,omitempty"`
	ThinkTokens         int                 `json:"think_tokens,omitempty"`
	GenTokSec           float64             `json:"gen_tok_sec,omitempty"`
	Aborted             bool                `json:"aborted,omitempty"`
	Err                 string              `json:"err,omitempty"`
	IsCompactionSummary bool                `json:"is_compaction_summary,omitempty"`
	CompactionDetail    string              `json:"compaction_detail,omitempty"`
	Timestamp           time.Time           `json:"ts"`
}

// BranchMeta describes one conversation branch created via edit/regenerate.
type BranchMeta struct {
	ID             string    `json:"id"`
	Name           string    `json:"name"`
	ParentBranchID string    `json:"parent_branch_id,omitempty"`
	ForkTurnIndex  int       `json:"fork_turn_index,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
}

// SessionMeta stores header metadata for a session JSONL file.
type SessionMeta struct {
	ID           string       `json:"id"`
	Title        string       `json:"title"`
	CreatedAt    time.Time    `json:"created_at"`
	UpdatedAt    time.Time    `json:"updated_at"`
	Workspace    string       `json:"workspace,omitempty"`
	Model        string       `json:"model,omitempty"`
	Preset       string       `json:"preset,omitempty"`
	Profile      string       `json:"profile,omitempty"`
	Tainted      bool         `json:"tainted,omitempty"`
	ActiveBranch string       `json:"active_branch"`
	Branches     []BranchMeta `json:"branches"`
}

// SessionDoc holds the full branched conversation persisted in sessions/<id>.jsonl.
type SessionDoc struct {
	Meta  SessionMeta
	Turns []BranchTurn
}

// SessionSummary is a lightweight index entry returned by ListSessions and SearchSessions.
type SessionSummary struct {
	ID          string
	Path        string
	Title       string
	UpdatedAt   time.Time
	Workspace   string
	Model       string
	Profile     string
	TurnCount   int
	BranchCount int
	Snippet     string
}

type jsonlEnvelope struct {
	Type string       `json:"type"`
	Meta *SessionMeta `json:"meta,omitempty"`
	Turn *BranchTurn  `json:"turn,omitempty"`
}

// SessionsRoot returns <appRoot>/sessions, falling back to <workDir>/.alpaca/sessions.
func SessionsRoot(appRoot, workDir string) string {
	base := strings.TrimSpace(appRoot)
	if base == "" {
		w := strings.TrimSpace(workDir)
		if w == "" {
			w = "."
		}
		return filepath.Join(w, ".alpaca", "sessions")
	}
	if abs, err := filepath.Abs(base); err == nil {
		base = abs
	}
	return filepath.Join(base, "sessions")
}

// NewSessionDoc initializes a fresh single-branch session document.
func NewSessionDoc(id, workspace, model, preset, profile string) *SessionDoc {
	now := time.Now().UTC()
	if strings.TrimSpace(id) == "" {
		id = fmt.Sprintf("sess-%d", now.UnixNano())
	}
	return &SessionDoc{
		Meta: SessionMeta{
			ID:           id,
			CreatedAt:    now,
			UpdatedAt:    now,
			Workspace:    workspace,
			Model:        model,
			Preset:       preset,
			Profile:      profile,
			ActiveBranch: "b1",
			Branches: []BranchMeta{
				{ID: "b1", Name: "main", CreatedAt: now},
			},
		},
	}
}

func sanitizeSessionID(id string) (string, error) {
	s := strings.TrimSpace(id)
	if !validIDRegexp.MatchString(s) || strings.Contains(s, "..") {
		return "", fmt.Errorf("invalid session id %q", id)
	}
	return s, nil
}

// ActiveTurns returns the chronological slice of turns belonging to ActiveBranch.
func (d *SessionDoc) ActiveTurns() []BranchTurn {
	if d == nil {
		return nil
	}
	br := d.Meta.ActiveBranch
	if br == "" {
		br = "b1"
	}
	var out []BranchTurn
	for _, t := range d.Turns {
		if t.BranchID == "" || t.BranchID == br {
			out = append(out, t)
		}
	}
	return out
}

// ReplaceActiveTurns updates turns on the currently active branch cleanly.
func (d *SessionDoc) ReplaceActiveTurns(activeTurns []BranchTurn) {
	if d == nil {
		return
	}
	br := d.Meta.ActiveBranch
	if br == "" {
		br = "b1"
		d.Meta.ActiveBranch = br
	}
	if len(d.Meta.Branches) == 0 {
		d.Meta.Branches = []BranchMeta{{ID: br, Name: "main", CreatedAt: time.Now().UTC()}}
	}
	var kept []BranchTurn
	for _, t := range d.Turns {
		if t.BranchID != "" && t.BranchID != br {
			kept = append(kept, t)
		}
	}
	for i := range activeTurns {
		activeTurns[i].BranchID = br
		if activeTurns[i].TurnID == "" {
			activeTurns[i].TurnID = fmt.Sprintf("%s-t%d", br, i+1)
		}
		if activeTurns[i].Timestamp.IsZero() {
			activeTurns[i].Timestamp = time.Now().UTC()
		}
	}
	d.Turns = append(kept, activeTurns...)
	if strings.TrimSpace(d.Meta.Title) == "" {
		for _, t := range activeTurns {
			if t.Role == "user" && strings.TrimSpace(t.Content) != "" {
				d.Meta.Title = autoTitle(t.Content)
				break
			}
		}
	}
}

// ForkBeforeActiveIndex clones active turns [0:cutIdx) into a new branch, switches ActiveBranch
// to the new branch, and leaves the old branch intact for future inspection or switching.
func (d *SessionDoc) ForkBeforeActiveIndex(cutIdx int, label string) (newBranchID string, prefix []BranchTurn) {
	if d == nil {
		return "", nil
	}
	oldBr := d.Meta.ActiveBranch
	if oldBr == "" {
		oldBr = "b1"
	}
	active := d.ActiveTurns()
	if cutIdx < 0 {
		cutIdx = 0
	}
	if cutIdx > len(active) {
		cutIdx = len(active)
	}

	nextNum := len(d.Meta.Branches) + 1
	newBranchID = fmt.Sprintf("b%d", nextNum)
	for branchExists(d.Meta.Branches, newBranchID) {
		nextNum++
		newBranchID = fmt.Sprintf("b%d", nextNum)
	}
	if strings.TrimSpace(label) == "" {
		label = fmt.Sprintf("branch-%d", nextNum)
	}

	d.Meta.Branches = append(d.Meta.Branches, BranchMeta{
		ID:             newBranchID,
		Name:           label,
		ParentBranchID: oldBr,
		ForkTurnIndex:  cutIdx,
		CreatedAt:      time.Now().UTC(),
	})
	d.Meta.ActiveBranch = newBranchID

	prefix = make([]BranchTurn, cutIdx)
	for i := 0; i < cutIdx; i++ {
		cp := active[i]
		cp.BranchID = newBranchID
		cp.TurnID = fmt.Sprintf("%s-t%d", newBranchID, i+1)
		prefix[i] = cp
		d.Turns = append(d.Turns, cp)
	}
	return newBranchID, prefix
}

// SwitchBranch switches ActiveBranch to targetID if it exists.
func (d *SessionDoc) SwitchBranch(branchID string) error {
	if d == nil {
		return errors.New("nil session")
	}
	for _, b := range d.Meta.Branches {
		if b.ID == branchID {
			d.Meta.ActiveBranch = b.ID
			return nil
		}
	}
	return fmt.Errorf("branch %q not found", branchID)
}

func branchExists(bs []BranchMeta, id string) bool {
	for _, b := range bs {
		if b.ID == id {
			return true
		}
	}
	return false
}

func autoTitle(prompt string) string {
	fields := strings.Fields(strings.TrimSpace(prompt))
	s := strings.Join(fields, " ")
	if len(s) > 56 {
		s = strings.TrimSpace(s[:56]) + "…"
	}
	return s
}

// SaveSession writes doc atomically as <sessionsDir>/<id>.jsonl.
func SaveSession(sessionsDir string, doc *SessionDoc) error {
	if doc == nil {
		return errors.New("nil session")
	}
	id, err := sanitizeSessionID(doc.Meta.ID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		return err
	}
	doc.Meta.UpdatedAt = time.Now().UTC()
	if doc.Meta.CreatedAt.IsZero() {
		doc.Meta.CreatedAt = doc.Meta.UpdatedAt
	}

	dst := filepath.Join(sessionsDir, id+".jsonl")
	tmp := dst + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 32*1024)
	writeLine := func(env jsonlEnvelope) error {
		b, mErr := json.Marshal(env)
		if mErr != nil {
			return mErr
		}
		if _, wErr := w.Write(b); wErr != nil {
			return wErr
		}
		return w.WriteByte('\n')
	}

	if err := writeLine(jsonlEnvelope{Type: "meta", Meta: &doc.Meta}); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	for i := range doc.Turns {
		cp := doc.Turns[i]
		if err := writeLine(jsonlEnvelope{Type: "turn", Turn: &cp}); err != nil {
			_ = f.Close()
			_ = os.Remove(tmp)
			return err
		}
	}
	if err := w.Flush(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

// LoadSession reads a JSONL session file cleanly without executing any side effects or tools.
func LoadSession(path string) (*SessionDoc, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	doc := &SessionDoc{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var env jsonlEnvelope
		if err := json.Unmarshal([]byte(line), &env); err != nil {
			// Skip corrupt/partial trailing lines rather than failing recovery.
			continue
		}
		switch env.Type {
		case "meta":
			if env.Meta != nil {
				doc.Meta = *env.Meta
			}
		case "turn":
			if env.Turn != nil {
				doc.Turns = append(doc.Turns, *env.Turn)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if doc.Meta.ID == "" {
		base := strings.TrimSuffix(filepath.Base(path), ".jsonl")
		doc.Meta.ID = base
	}
	if doc.Meta.ActiveBranch == "" {
		doc.Meta.ActiveBranch = "b1"
	}
	if len(doc.Meta.Branches) == 0 {
		doc.Meta.Branches = []BranchMeta{{ID: doc.Meta.ActiveBranch, Name: "main", CreatedAt: doc.Meta.CreatedAt}}
	}
	return doc, nil
}

// LoadSessionByID loads <sessionsDir>/<id>.jsonl with ID sanitization.
func LoadSessionByID(sessionsDir, id string) (*SessionDoc, error) {
	cleanID, err := sanitizeSessionID(id)
	if err != nil {
		return nil, err
	}
	return LoadSession(filepath.Join(sessionsDir, cleanID+".jsonl"))
}

// ListSessions enumerates saved sessions ordered newest-first.
func ListSessions(sessionsDir string) ([]SessionSummary, error) {
	des, err := os.ReadDir(sessionsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []SessionSummary
	for _, de := range des {
		if de.IsDir() || !strings.HasSuffix(strings.ToLower(de.Name()), ".jsonl") {
			continue
		}
		full := filepath.Join(sessionsDir, de.Name())
		doc, lErr := LoadSession(full)
		if lErr != nil {
			continue
		}
		active := doc.ActiveTurns()
		title := strings.TrimSpace(doc.Meta.Title)
		if title == "" {
			title = "(untitled session)"
		}
		snippet := ""
		for _, t := range active {
			if t.Role == "user" && strings.TrimSpace(t.Content) != "" {
				snippet = autoTitle(t.Content)
				break
			}
		}
		out = append(out, SessionSummary{
			ID:          doc.Meta.ID,
			Path:        full,
			Title:       title,
			UpdatedAt:   doc.Meta.UpdatedAt,
			Workspace:   doc.Meta.Workspace,
			Model:       doc.Meta.Model,
			Profile:     doc.Meta.Profile,
			TurnCount:   len(active),
			BranchCount: len(doc.Meta.Branches),
			Snippet:     snippet,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}

// SearchSessions searches saved session titles AND message contents case-insensitively.
func SearchSessions(sessionsDir, query string) ([]SessionSummary, error) {
	q := strings.ToLower(strings.TrimSpace(query))
	all, err := ListSessions(sessionsDir)
	if err != nil || q == "" {
		return all, err
	}
	var matches []SessionSummary
	for _, s := range all {
		doc, lErr := LoadSession(s.Path)
		if lErr != nil {
			continue
		}
		if strings.Contains(strings.ToLower(doc.Meta.Title), q) || strings.Contains(strings.ToLower(doc.Meta.ID), q) {
			matches = append(matches, s)
			continue
		}
		found := false
		for _, t := range doc.Turns {
			if idx := strings.Index(strings.ToLower(t.Content), q); idx >= 0 {
				start := idx - 20
				if start < 0 {
					start = 0
				}
				end := idx + len(q) + 32
				if end > len(t.Content) {
					end = len(t.Content)
				}
				s.Snippet = strings.TrimSpace(strings.ReplaceAll(t.Content[start:end], "\n", " "))
				found = true
				break
			}
		}
		if found {
			matches = append(matches, s)
		}
	}
	return matches, nil
}

// RenameSession updates a persisted session's title atomically.
func RenameSession(sessionsDir, id, newTitle string) error {
	doc, err := LoadSessionByID(sessionsDir, id)
	if err != nil {
		return err
	}
	title := strings.TrimSpace(newTitle)
	if title == "" {
		return errors.New("session title cannot be empty")
	}
	doc.Meta.Title = title
	return SaveSession(sessionsDir, doc)
}

// DeleteSession moves a session JSONL file into <sessionsDir>/.trash/ or removes it safely.
func DeleteSession(sessionsDir, id string) error {
	cleanID, err := sanitizeSessionID(id)
	if err != nil {
		return err
	}
	src := filepath.Join(sessionsDir, cleanID+".jsonl")
	return os.Remove(src)
}

// ExportMarkdown formats the active branch of doc cleanly as readable Markdown.
func ExportMarkdown(doc *SessionDoc) string {
	if doc == nil {
		return ""
	}
	var b strings.Builder
	title := doc.Meta.Title
	if strings.TrimSpace(title) == "" {
		title = doc.Meta.ID
	}
	b.WriteString(fmt.Sprintf("# %s\n\n", title))
	b.WriteString(fmt.Sprintf("- Session ID: `%s`\n", doc.Meta.ID))
	b.WriteString(fmt.Sprintf("- Branch: `%s` (%d total branches)\n", doc.Meta.ActiveBranch, len(doc.Meta.Branches)))
	if doc.Meta.Model != "" {
		b.WriteString(fmt.Sprintf("- Model: `%s`\n", doc.Meta.Model))
	}
	if doc.Meta.Profile != "" {
		b.WriteString(fmt.Sprintf("- Profile: `%s`\n", doc.Meta.Profile))
	}
	if doc.Meta.Workspace != "" {
		b.WriteString(fmt.Sprintf("- Workspace: `%s`\n", doc.Meta.Workspace))
	}
	b.WriteString("\n---\n\n")

	for _, t := range doc.ActiveTurns() {
		switch t.Role {
		case "user":
			b.WriteString("## User\n\n")
			b.WriteString(t.Content)
			b.WriteString("\n\n")
			for _, a := range t.Attachments {
				lr := ""
				if a.LineRange != "" {
					lr = ":" + a.LineRange
				}
				b.WriteString(fmt.Sprintf("- Attachment: `@%s%s` (%d tok)\n", a.Path, lr, a.Tokens))
			}
			if len(t.Attachments) > 0 {
				b.WriteByte('\n')
			}
		case "info":
			b.WriteString("> **Note:** ")
			b.WriteString(strings.ReplaceAll(strings.TrimSpace(t.Content), "\n", "\n> "))
			b.WriteString("\n\n")
		case "assistant":
			b.WriteString("## Assistant\n\n")
			if strings.TrimSpace(t.Reasoning) != "" {
				b.WriteString("<details><summary>Thinking</summary>\n\n")
				b.WriteString(t.Reasoning)
				b.WriteString("\n\n</details>\n\n")
			}
			for _, tc := range t.ToolCalls {
				b.WriteString(fmt.Sprintf("- Tool `%s` (`%s`) — %s\n", tc.Name, tc.Status, tc.Summary))
			}
			if len(t.ToolCalls) > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(t.Content)
			b.WriteString("\n\n")
		}
	}
	return b.String()
}
