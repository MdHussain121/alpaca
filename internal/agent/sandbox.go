package agent

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// AccessMode specifies whether a sandboxed path validation is for reading, mutating, or listing.
type AccessMode int

const (
	AccessRead AccessMode = iota
	AccessWrite
	AccessCreate
)

var (
	ErrOutsideWorkspace = errors.New("path escapes workspace sandbox")
	ErrHostilePath      = errors.New("path uses forbidden Windows/NT/device/ADS syntax")
	ErrProtectedPath    = errors.New("path targets protected internal directory")
)

var winReservedNames = map[string]bool{
	"CON": true, "PRN": true, "AUX": true, "NUL": true,
	"COM1": true, "COM2": true, "COM3": true, "COM4": true, "COM5": true, "COM6": true, "COM7": true, "COM8": true, "COM9": true,
	"LPT1": true, "LPT2": true, "LPT3": true, "LPT4": true, "LPT5": true, "LPT6": true, "LPT7": true, "LPT8": true, "LPT9": true,
	"COM¹": true, "COM²": true, "COM³": true, "LPT¹": true, "LPT²": true, "LPT³": true,
}

// ResolveWorkspacePath strictly resolves candidate against workspaceRoot, rejecting all
// hostile path syntaxes (UNC, \\?\, \\.\, drive-relative, ADS, reserved names, trailing dot/space,
// symlink/junction escapes) and forbidding writes to .alpaca/, .llm/, or skills/.
func ResolveWorkspacePath(workspaceRoot, candidate string, mode AccessMode) (resolved string, relSlash string, err error) {
	if strings.TrimSpace(workspaceRoot) == "" {
		return "", "", errors.New("empty workspace root")
	}
	if candidate == "" || strings.TrimSpace(candidate) == "" {
		return "", "", fmt.Errorf("%w: empty path", ErrHostilePath)
	}
	if strings.ContainsRune(candidate, 0) {
		return "", "", fmt.Errorf("%w: NUL byte", ErrHostilePath)
	}
	for _, r := range candidate {
		if r < 0x20 {
			return "", "", fmt.Errorf("%w: control character", ErrHostilePath)
		}
	}

	raw := candidate
	slashNorm := strings.ReplaceAll(raw, `\`, `/`)
	// Reject UNC paths (\\server\share, //server/share) and extended/device namespaces (\\?\, \\.\, \??\).
	if strings.HasPrefix(raw, `\\`) || strings.HasPrefix(raw, `//`) ||
		strings.HasPrefix(slashNorm, `//`) || strings.HasPrefix(raw, `\??\`) || strings.HasPrefix(slashNorm, `/??/`) {
		return "", "", fmt.Errorf("%w: UNC or device namespace %q", ErrHostilePath, raw)
	}
	// Reject root-relative paths on Windows (\foo or /foo) unless cleanly relative/workspace-abs.
	if runtime.GOOS == "windows" {
		if len(raw) >= 1 && (raw[0] == '\\' || raw[0] == '/') {
			return "", "", fmt.Errorf("%w: drive-root relative path %q", ErrHostilePath, raw)
		}
		// Reject drive-relative paths like "C:foo.txt" (drive letter without separator).
		if len(raw) >= 2 && isASCIILetter(raw[0]) && raw[1] == ':' {
			if len(raw) == 2 || (raw[2] != '\\' && raw[2] != '/') {
				return "", "", fmt.Errorf("%w: drive-relative path %q", ErrHostilePath, raw)
			}
			// Reject alternate data streams after drive prefix (C:\a:b).
			if strings.ContainsRune(raw[2:], ':') {
				return "", "", fmt.Errorf("%w: alternate data stream %q", ErrHostilePath, raw)
			}
		} else if strings.ContainsRune(raw, ':') {
			// Relative path with ':' is ADS or alternate volume syntax.
			return "", "", fmt.Errorf("%w: colon/ADS %q", ErrHostilePath, raw)
		}
	} else if strings.ContainsRune(raw, ':') {
		return "", "", fmt.Errorf("%w: colon/ADS %q", ErrHostilePath, raw)
	}

	// Validate each individual segment BEFORE filepath.Clean can strip trailing dots/spaces or normalize devices.
	if err := validateRawSegments(slashNorm); err != nil {
		return "", "", err
	}

	canonRoot, err := canonicalRoot(workspaceRoot)
	if err != nil {
		return "", "", err
	}

	joined := raw
	if !filepath.IsAbs(joined) {
		joined = filepath.Join(canonRoot, joined)
	}
	cleaned := filepath.Clean(joined)

	if err := checkVolumeAndContainment(canonRoot, cleaned); err != nil {
		return "", "", err
	}

	realTarget, err := evalSymlinksWithMissingTail(cleaned)
	if err != nil {
		return "", "", fmt.Errorf("%w: %v", ErrOutsideWorkspace, err)
	}
	if err := checkVolumeAndContainment(canonRoot, realTarget); err != nil {
		return "", "", err
	}

	rel, err := filepath.Rel(canonRoot, realTarget)
	if err != nil {
		return "", "", fmt.Errorf("%w: %v", ErrOutsideWorkspace, err)
	}
	relSlash = filepath.ToSlash(rel)
	if relSlash == ".." || strings.HasPrefix(relSlash, "../") {
		return "", "", fmt.Errorf("%w: %q", ErrOutsideWorkspace, candidate)
	}

	// Protected internal directories: .alpaca/, .llm/, and skills/ cannot be modified directly via file tools.
	if mode == AccessWrite || mode == AccessCreate {
		if err := CheckProtectedWriteRel(relSlash); err != nil {
			return "", "", err
		}
	}

	return realTarget, relSlash, nil
}

// CheckProtectedWriteRel forbids writes to .alpaca/, .llm/, or skills/ (including root itself if delete/move).
func CheckProtectedWriteRel(relSlash string) error {
	low := strings.ToLower(strings.Trim(relSlash, "/"))
	if low == "." {
		return fmt.Errorf("%w: cannot overwrite or delete workspace root", ErrProtectedPath)
	}
	if low == ".alpaca" || strings.HasPrefix(low, ".alpaca/") {
		return fmt.Errorf("%w: cannot write to .alpaca/", ErrProtectedPath)
	}
	if low == ".llm" || strings.HasPrefix(low, ".llm/") {
		return fmt.Errorf("%w: cannot write to .llm/", ErrProtectedPath)
	}
	if low == "skills" || strings.HasPrefix(low, "skills/") {
		return fmt.Errorf("%w: cannot write to skills/ (use save_skill)", ErrProtectedPath)
	}
	return nil
}

// IsHiddenInternal returns true for .alpaca, .llm, and anything beneath them.
func IsHiddenInternal(relSlash string) bool {
	low := strings.ToLower(strings.TrimPrefix(strings.Trim(relSlash, "/"), "./"))
	return low == ".alpaca" || strings.HasPrefix(low, ".alpaca/") ||
		low == ".llm" || strings.HasPrefix(low, ".llm/")
}

func validateRawSegments(slashNorm string) error {
	checkPart := slashNorm
	// Skip "C:/" drive prefix when splitting segments.
	if runtime.GOOS == "windows" && len(checkPart) >= 3 && isASCIILetter(checkPart[0]) && checkPart[1] == ':' && checkPart[2] == '/' {
		checkPart = checkPart[3:]
	}
	for _, seg := range strings.Split(checkPart, "/") {
		if seg == "" || seg == "." || seg == ".." {
			continue
		}
		if strings.HasSuffix(seg, ".") || strings.HasSuffix(seg, " ") {
			return fmt.Errorf("%w: segment %q has trailing dot or space", ErrHostilePath, seg)
		}
		base := seg
		if dot := strings.IndexByte(seg, '.'); dot >= 0 {
			base = seg[:dot]
		}
		if winReservedNames[strings.ToUpper(strings.TrimSpace(base))] {
			return fmt.Errorf("%w: reserved device name %q", ErrHostilePath, seg)
		}
	}
	return nil
}

func canonicalRoot(root string) (string, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	realRoot, err := filepath.EvalSymlinks(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return filepath.Clean(abs), nil
		}
		return "", err
	}
	return filepath.Clean(realRoot), nil
}

func evalSymlinksWithMissingTail(target string) (string, error) {
	target = filepath.Clean(target)
	if real, err := filepath.EvalSymlinks(target); err == nil {
		return filepath.Clean(real), nil
	}
	var tail []string
	cur := target
	for {
		parent := filepath.Dir(cur)
		tail = append(tail, filepath.Base(cur))
		if parent == cur {
			return "", fmt.Errorf("no existing ancestor for %q", target)
		}
		if realParent, err := filepath.EvalSymlinks(parent); err == nil {
			out := realParent
			for i := len(tail) - 1; i >= 0; i-- {
				out = filepath.Join(out, tail[i])
			}
			return filepath.Clean(out), nil
		}
		cur = parent
	}
}

func checkVolumeAndContainment(root, target string) error {
	if runtime.GOOS == "windows" {
		rv := filepath.VolumeName(root)
		tv := filepath.VolumeName(target)
		if !strings.EqualFold(rv, tv) {
			return fmt.Errorf("%w: volume %q != workspace volume %q", ErrOutsideWorkspace, tv, rv)
		}
		rootLow := strings.ToLower(strings.TrimSuffix(root, string(filepath.Separator)))
		tgtLow := strings.ToLower(strings.TrimSuffix(target, string(filepath.Separator)))
		if tgtLow == rootLow {
			return nil
		}
		if !strings.HasPrefix(tgtLow, rootLow+string(filepath.Separator)) {
			return fmt.Errorf("%w: %q is outside %q", ErrOutsideWorkspace, target, root)
		}
		return nil
	}
	rel, err := filepath.Rel(root, target)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("%w: %q is outside %q", ErrOutsideWorkspace, target, root)
	}
	return nil
}

func isASCIILetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}
