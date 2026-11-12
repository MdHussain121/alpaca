package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const defaultMaxCheckpointBytes int64 = 64 * 1024 * 1024

type checkpointEntry struct {
	RelPath   string `json:"rel_path"`
	Existed   bool   `json:"existed"`
	IsDir     bool   `json:"is_dir"`
	SavedFile string `json:"saved_file,omitempty"`
}

type checkpointManifest struct {
	Turn    int               `json:"turn"`
	Created string            `json:"created"`
	Entries []checkpointEntry `json:"entries"`
	seen    map[string]bool   `json:"-"`
}

// CheckpointStore persists per-turn file snapshots under .alpaca/checkpoints/<session>/<turn>/
// so /undo can restore the last mutating turn cleanly and prunes oldest snapshots past MaxBytes.
type CheckpointStore struct {
	WorkspaceRoot string
	SessionID     string
	MaxBytes      int64
}

func NewCheckpointStore(workspaceRoot, sessionID string) *CheckpointStore {
	if strings.TrimSpace(sessionID) == "" {
		sessionID = "default"
	}
	return &CheckpointStore{
		WorkspaceRoot: workspaceRoot,
		SessionID:     sanitizeSegment(sessionID),
		MaxBytes:      defaultMaxCheckpointBytes,
	}
}

func (s *CheckpointStore) sessionDir() string {
	return filepath.Join(s.WorkspaceRoot, ".alpaca", "checkpoints", s.SessionID)
}

func (s *CheckpointStore) turnDir(turn int) string {
	return filepath.Join(s.sessionDir(), fmt.Sprintf("turn-%04d", turn))
}

// CaptureBefore snapshots targetAbs (relative relSlash) for turn BEFORE a mutating tool edits/creates/deletes it.
// Calling CaptureBefore multiple times on the same file within the same turn preserves the pre-turn state.
func (s *CheckpointStore) CaptureBefore(turn int, targetAbs, relSlash string) error {
	if s == nil || s.WorkspaceRoot == "" || relSlash == "" {
		return nil
	}
	dir := s.turnDir(turn)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	manifestPath := filepath.Join(dir, "manifest.json")
	man := s.loadManifest(manifestPath, turn)
	key := strings.ToLower(relSlash)
	if man.seen[key] {
		return nil
	}
	man.seen[key] = true

	st, err := os.Stat(targetAbs)
	if err != nil {
		if os.IsNotExist(err) {
			man.Entries = append(man.Entries, checkpointEntry{
				RelPath: relSlash,
				Existed: false,
			})
			return s.saveManifest(manifestPath, man)
		}
		return err
	}

	if st.IsDir() {
		man.Entries = append(man.Entries, checkpointEntry{
			RelPath: relSlash,
			Existed: true,
			IsDir:   true,
		})
		return s.saveManifest(manifestPath, man)
	}

	data, err := os.ReadFile(targetAbs)
	if err != nil {
		return err
	}
	savedName := fmt.Sprintf("file-%03d.bak", len(man.Entries)+1)
	if err := os.WriteFile(filepath.Join(dir, savedName), data, 0o644); err != nil {
		return err
	}
	man.Entries = append(man.Entries, checkpointEntry{
		RelPath:   relSlash,
		Existed:   true,
		SavedFile: savedName,
	})
	if err := s.saveManifest(manifestPath, man); err != nil {
		return err
	}
	s.pruneOldest()
	return nil
}

// PendingUndoSummary returns the file paths that the latest saved checkpoint turn would revert.
func (s *CheckpointStore) PendingUndoSummary() (turn int, paths []string, ok bool) {
	if s == nil {
		return 0, nil, false
	}
	turns := s.listTurnDirs()
	if len(turns) == 0 {
		return 0, nil, false
	}
	last := turns[len(turns)-1]
	man := s.loadManifest(filepath.Join(last, "manifest.json"), 0)
	if len(man.Entries) == 0 {
		return 0, nil, false
	}
	for _, e := range man.Entries {
		paths = append(paths, e.RelPath)
	}
	return man.Turn, paths, true
}

// UndoLastTurn restores all files modified during the latest checkpointed turn and deletes that checkpoint.
func (s *CheckpointStore) UndoLastTurn() (restored []string, deleted []string, err error) {
	if s == nil {
		return nil, nil, fmt.Errorf("checkpoints unavailable")
	}
	turns := s.listTurnDirs()
	if len(turns) == 0 {
		return nil, nil, fmt.Errorf("nothing to undo")
	}
	lastDir := turns[len(turns)-1]
	man := s.loadManifest(filepath.Join(lastDir, "manifest.json"), 0)
	if len(man.Entries) == 0 {
		_ = os.RemoveAll(lastDir)
		return nil, nil, fmt.Errorf("nothing to undo")
	}

	// Restore in reverse order cleanly.
	for i := len(man.Entries) - 1; i >= 0; i-- {
		e := man.Entries[i]
		target := filepath.Join(s.WorkspaceRoot, filepath.FromSlash(e.RelPath))
		if !e.Existed {
			if remErr := os.RemoveAll(target); remErr != nil && !os.IsNotExist(remErr) {
				return restored, deleted, remErr
			}
			deleted = append(deleted, e.RelPath)
			continue
		}
		if e.IsDir {
			if mkErr := os.MkdirAll(target, 0o755); mkErr != nil {
				return restored, deleted, mkErr
			}
			restored = append(restored, e.RelPath)
			continue
		}
		raw, rErr := os.ReadFile(filepath.Join(lastDir, e.SavedFile))
		if rErr != nil {
			return restored, deleted, rErr
		}
		if mkErr := os.MkdirAll(filepath.Dir(target), 0o755); mkErr != nil {
			return restored, deleted, mkErr
		}
		if wErr := os.WriteFile(target, raw, 0o644); wErr != nil {
			return restored, deleted, wErr
		}
		restored = append(restored, e.RelPath)
	}

	_ = os.RemoveAll(lastDir)
	return restored, deleted, nil
}

func (s *CheckpointStore) listTurnDirs() []string {
	des, err := os.ReadDir(s.sessionDir())
	if err != nil {
		return nil
	}
	var names []string
	for _, de := range des {
		if de.IsDir() && strings.HasPrefix(de.Name(), "turn-") {
			names = append(names, filepath.Join(s.sessionDir(), de.Name()))
		}
	}
	sort.Strings(names)
	return names
}

func (s *CheckpointStore) loadManifest(path string, turn int) checkpointManifest {
	man := checkpointManifest{
		Turn:    turn,
		Created: time.Now().Format(time.RFC3339),
		seen:    make(map[string]bool),
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return man
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		return checkpointManifest{Turn: turn, Created: time.Now().Format(time.RFC3339), seen: make(map[string]bool)}
	}
	man.seen = make(map[string]bool, len(man.Entries))
	for _, e := range man.Entries {
		man.seen[strings.ToLower(e.RelPath)] = true
	}
	return man
}

func (s *CheckpointStore) saveManifest(path string, man checkpointManifest) error {
	b, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func (s *CheckpointStore) pruneOldest() {
	capBytes := s.MaxBytes
	if capBytes <= 0 {
		capBytes = defaultMaxCheckpointBytes
	}
	root := filepath.Join(s.WorkspaceRoot, ".alpaca", "checkpoints")
	for {
		total, dirs := dirTreeSizeAndLeaves(root)
		if total <= capBytes || len(dirs) <= 1 {
			return
		}
		// Never delete the newest turn directory; remove oldest turn directory first.
		_ = os.RemoveAll(dirs[0].path)
	}
}

type sizedDir struct {
	path  string
	mtime int64
}

func dirTreeSizeAndLeaves(root string) (int64, []sizedDir) {
	var total int64
	var turns []sizedDir
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if strings.HasPrefix(d.Name(), "turn-") {
				if info, iErr := d.Info(); iErr == nil {
					turns = append(turns, sizedDir{path: p, mtime: info.ModTime().UnixNano()})
				}
			}
			return nil
		}
		if info, iErr := d.Info(); iErr == nil {
			total += info.Size()
		}
		return nil
	})
	sort.Slice(turns, func(i, j int) bool {
		return turns[i].mtime < turns[j].mtime
	})
	return total, turns
}

func sanitizeSegment(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}
