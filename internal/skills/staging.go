package skills

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

// StageRequest holds unvalidated save_skill arguments prior to staging.
type StageRequest struct {
	Name          string
	Description   string
	Body          string
	Files         map[string]string
	ModelAuthored bool
}

// StagedBundle describes a validated skill staged inside .alpaca/skills-staging/<name>/.
type StagedBundle struct {
	Name          string
	Description   string
	Body          string
	Files         map[string]string
	StagingDir    string
	RenderedMD    string
	ExistingMD    string
	Overwrites    bool
	ModelAuthored bool
	TotalBytes    int
}

// CommitResult describes a saved skill written after explicit user approval.
type CommitResult struct {
	SkillDir    string
	SkillFile   string
	HistoryPath string
	Overwrote   bool
	TotalBytes  int
}

// ValidateStageRequest checks name, description, body size, and extra file safety & caps.
func ValidateStageRequest(req StageRequest) (cleanName string, cleanDesc string, cleanFiles map[string]string, err error) {
	cleanName, err = ValidateSkillName(req.Name)
	if err != nil {
		return "", "", nil, err
	}
	cleanDesc, err = SanitizeDescription(req.Description)
	if err != nil {
		return "", "", nil, err
	}
	body := strings.TrimSpace(strings.ReplaceAll(req.Body, "\r\n", "\n"))
	if body == "" {
		return "", "", nil, errors.New("skill body cannot be empty")
	}
	if !utf8.ValidString(body) || strings.ContainsRune(body, 0) {
		return "", "", nil, errors.New("skill body must be valid UTF-8 text without NUL bytes")
	}
	if len(body) > MaxBodyBytes {
		return "", "", nil, fmt.Errorf("skill body exceeds %d byte cap (%d bytes)", MaxBodyBytes, len(body))
	}

	if len(req.Files) > MaxExtraFiles {
		return "", "", nil, fmt.Errorf("too many extra files (%d); maximum is %d", len(req.Files), MaxExtraFiles)
	}

	totalBytes := len(RenderSkillMarkdown(cleanName, cleanDesc, body, req.ModelAuthored))
	cleanFiles = make(map[string]string, len(req.Files))
	for rawRel, content := range req.Files {
		rel, vErr := validateExtraFileRel(rawRel)
		if vErr != nil {
			return "", "", nil, vErr
		}
		if !utf8.ValidString(content) || strings.ContainsRune(content, 0) {
			return "", "", nil, fmt.Errorf("file %q must be valid UTF-8 text without NUL bytes", rel)
		}
		if len(content) > MaxExtraFileBytes {
			return "", "", nil, fmt.Errorf("file %q exceeds %d byte cap (%d bytes)", rel, MaxExtraFileBytes, len(content))
		}
		totalBytes += len(content)
		if totalBytes > MaxTotalSkillBytes {
			return "", "", nil, fmt.Errorf("total skill size exceeds %d byte cap (%d bytes)", MaxTotalSkillBytes, totalBytes)
		}
		cleanFiles[rel] = content
	}
	if totalBytes > MaxTotalSkillBytes {
		return "", "", nil, fmt.Errorf("total skill size exceeds %d byte cap (%d bytes)", MaxTotalSkillBytes, totalBytes)
	}
	return cleanName, cleanDesc, cleanFiles, nil
}

func validateExtraFileRel(raw string) (string, error) {
	if strings.ContainsRune(raw, 0) {
		return "", fmt.Errorf("invalid extra file path %q: NUL byte", raw)
	}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", errors.New("extra file path cannot be empty")
	}
	if strings.ContainsAny(trimmed, ":*?\"<>|") {
		return "", fmt.Errorf("invalid extra file path %q: illegal character", raw)
	}
	if strings.HasPrefix(trimmed, `\`) || strings.HasPrefix(trimmed, "/") || filepath.IsAbs(trimmed) {
		return "", fmt.Errorf("invalid extra file path %q: must be relative", raw)
	}
	slash := filepath.ToSlash(trimmed)
	segs := strings.Split(slash, "/")
	if len(segs) > 4 {
		return "", fmt.Errorf("extra file path %q too deeply nested (max 4 segments)", raw)
	}
	for _, seg := range segs {
		if seg == "" || seg == "." || seg == ".." {
			return "", fmt.Errorf("invalid extra file path %q: traversal or empty segment", raw)
		}
		if strings.HasSuffix(seg, ".") || strings.HasSuffix(seg, " ") {
			return "", fmt.Errorf("invalid extra file path %q: trailing dot or space", raw)
		}
		base := seg
		if dot := strings.IndexByte(seg, '.'); dot >= 0 {
			base = seg[:dot]
		}
		if winReserved[strings.ToUpper(strings.TrimSpace(base))] {
			return "", fmt.Errorf("invalid extra file path %q: reserved Windows device name", raw)
		}
	}
	if strings.EqualFold(slash, "SKILL.md") {
		return "", errors.New("files cannot overwrite SKILL.md directly")
	}
	return slash, nil
}

// PrepareStaging writes a validated skill bundle into .alpaca/skills-staging/<name>/
// without touching skills/<name>/ yet.
func PrepareStaging(skillsDir, dataDir string, req StageRequest) (*StagedBundle, error) {
	cleanName, cleanDesc, cleanFiles, err := ValidateStageRequest(req)
	if err != nil {
		return nil, err
	}
	stagingDir := filepath.Join(dataDir, "skills-staging", cleanName)
	_ = os.RemoveAll(stagingDir)
	if err := os.MkdirAll(stagingDir, 0o755); err != nil {
		return nil, err
	}

	rendered := RenderSkillMarkdown(cleanName, cleanDesc, req.Body, req.ModelAuthored)
	if err := os.WriteFile(filepath.Join(stagingDir, "SKILL.md"), []byte(rendered), 0o644); err != nil {
		_ = os.RemoveAll(stagingDir)
		return nil, err
	}
	totalBytes := len(rendered)
	for rel, content := range cleanFiles {
		dst := filepath.Join(stagingDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			_ = os.RemoveAll(stagingDir)
			return nil, err
		}
		if err := os.WriteFile(dst, []byte(content), 0o644); err != nil {
			_ = os.RemoveAll(stagingDir)
			return nil, err
		}
		totalBytes += len(content)
	}

	existingPath := filepath.Join(skillsDir, cleanName, "SKILL.md")
	existingBytes, rErr := os.ReadFile(existingPath)
	overwrites := false
	existingMD := ""
	if rErr == nil {
		overwrites = true
		existingMD = strings.ReplaceAll(string(existingBytes), "\r\n", "\n")
	} else if _, sErr := os.Stat(filepath.Join(skillsDir, cleanName)); sErr == nil {
		overwrites = true
	}

	return &StagedBundle{
		Name:          cleanName,
		Description:   cleanDesc,
		Body:          strings.TrimSpace(strings.ReplaceAll(req.Body, "\r\n", "\n")),
		Files:         cleanFiles,
		StagingDir:    stagingDir,
		RenderedMD:    rendered,
		ExistingMD:    existingMD,
		Overwrites:    overwrites,
		ModelAuthored: req.ModelAuthored,
		TotalBytes:    totalBytes,
	}, nil
}

// CommitStaged moves an approved staged skill into skills/<finalName>/, archiving any
// existing version into .alpaca/skill-history/<finalName>/<ts>/ first.
func CommitStaged(skillsDir, dataDir string, staged *StagedBundle, finalName, finalDesc, finalBody string) (CommitResult, error) {
	if staged == nil {
		return CommitResult{}, errors.New("staged bundle is nil")
	}
	defer func() {
		if staged.StagingDir != "" {
			_ = os.RemoveAll(staged.StagingDir)
		}
	}()

	cleanName, err := ValidateSkillName(finalName)
	if err != nil {
		return CommitResult{}, err
	}
	cleanDesc, err := SanitizeDescription(finalDesc)
	if err != nil {
		return CommitResult{}, err
	}
	body := strings.TrimSpace(strings.ReplaceAll(finalBody, "\r\n", "\n"))
	if body == "" {
		return CommitResult{}, errors.New("skill body cannot be empty")
	}
	if len(body) > MaxBodyBytes {
		return CommitResult{}, fmt.Errorf("skill body exceeds %d byte cap", MaxBodyBytes)
	}

	finalMD := RenderSkillMarkdown(cleanName, cleanDesc, body, staged.ModelAuthored)
	if err := os.WriteFile(filepath.Join(staged.StagingDir, "SKILL.md"), []byte(finalMD), 0o644); err != nil {
		return CommitResult{}, err
	}

	targetDir := filepath.Join(skillsDir, cleanName)
	var historyPath string
	overwrote := false
	if _, err := os.Stat(targetDir); err == nil {
		overwrote = true
		historyPath = filepath.Join(dataDir, "skill-history", cleanName, time.Now().UTC().Format("20060102-150405.000"))
		if err := os.MkdirAll(filepath.Dir(historyPath), 0o755); err != nil {
			return CommitResult{}, err
		}
		if err := os.Rename(targetDir, historyPath); err != nil {
			return CommitResult{}, fmt.Errorf("archive existing skill: %w", err)
		}
	}

	if err := os.MkdirAll(skillsDir, 0o755); err != nil {
		return CommitResult{}, err
	}
	if err := os.Rename(staged.StagingDir, targetDir); err != nil {
		if copyErr := copyDir(staged.StagingDir, targetDir); copyErr != nil {
			return CommitResult{}, fmt.Errorf("commit skill: %w (rename err: %v)", copyErr, err)
		}
	}

	return CommitResult{
		SkillDir:    targetDir,
		SkillFile:   filepath.Join(targetDir, "SKILL.md"),
		HistoryPath: historyPath,
		Overwrote:   overwrote,
		TotalBytes:  staged.TotalBytes,
	}, nil
}

// DiscardStaged removes temporary staging files after rejection or interruption.
func DiscardStaged(staged *StagedBundle) {
	if staged != nil && staged.StagingDir != "" {
		_ = os.RemoveAll(staged.StagingDir)
	}
}

func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, p)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
}
