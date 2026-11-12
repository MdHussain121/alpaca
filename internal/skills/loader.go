package skills

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	MaxBodyBytes       = 16 * 1024
	MaxExtraFiles      = 10
	MaxExtraFileBytes  = 16 * 1024
	MaxTotalSkillBytes = 64 * 1024
	MaxPromptSkills    = 30
	MaxPromptBytes     = 4096
)

var (
	nameRegexp = regexp.MustCompile(`^[a-z0-9-]{1,40}$`)

	winReserved = map[string]bool{
		"CON": true, "PRN": true, "AUX": true, "NUL": true,
		"COM1": true, "COM2": true, "COM3": true, "COM4": true,
		"COM5": true, "COM6": true, "COM7": true, "COM8": true, "COM9": true,
		"LPT1": true, "LPT2": true, "LPT3": true, "LPT4": true,
		"LPT5": true, "LPT6": true, "LPT7": true, "LPT8": true, "LPT9": true,
	}
)

// Skill describes one validated Agent Skill loaded from skills/<name>/SKILL.md.
type Skill struct {
	Name          string
	Description   string
	Body          string
	RawMarkdown   string
	Dir           string
	SkillFile     string
	ExtraFiles    []string
	ModelAuthored bool
	Enabled       bool
	UseCount      int
	LastUsed      time.Time
	TotalBytes    int
}

// InvalidSkill describes a malformed skill directory or SKILL.md without failing startup.
type InvalidSkill struct {
	Name   string
	Dir    string
	Reason string
}

// Catalog holds valid and invalid skills alongside state metadata.
type Catalog struct {
	SkillsDir      string
	DataDir        string
	Valid          []Skill
	Invalid        []InvalidSkill
	TruncatedCount int
}

type skillStateEntry struct {
	Disabled bool      `json:"disabled,omitempty"`
	UseCount int       `json:"use_count,omitempty"`
	LastUsed time.Time `json:"last_used,omitempty"`
}

type skillStateFile struct {
	Skills map[string]skillStateEntry `json:"skills"`
}

// SkillsRoot returns <appRoot>/skills, falling back to <workDir>/skills.
func SkillsRoot(appRoot, workDir string) string {
	base := strings.TrimSpace(appRoot)
	if base == "" {
		base = strings.TrimSpace(workDir)
	}
	if base == "" {
		base = "."
	}
	if abs, err := filepath.Abs(base); err == nil {
		base = abs
	}
	return filepath.Join(base, "skills")
}

// DataRoot returns <skillsParent>/.alpaca so staging, history, state, and trash stay alongside skills/.
func DataRoot(skillsDir, workDir string) string {
	s := strings.TrimSpace(skillsDir)
	if s != "" {
		return filepath.Join(filepath.Dir(s), ".alpaca")
	}
	w := strings.TrimSpace(workDir)
	if w == "" {
		w = "."
	}
	return filepath.Join(w, ".alpaca")
}

// ValidateSkillName enforces ^[a-z0-9-]{1,40}$ plus Windows reserved device name and traversal checks.
func ValidateSkillName(name string) (string, error) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return "", errors.New("skill name cannot be empty")
	}
	if strings.ContainsAny(trimmed, `/\:. `) {
		return "", fmt.Errorf("invalid skill name %q: must contain only lowercase letters, digits, and hyphens", name)
	}
	if !nameRegexp.MatchString(trimmed) {
		return "", fmt.Errorf("invalid skill name %q: must match ^[a-z0-9-]{1,40}$", name)
	}
	if winReserved[strings.ToUpper(trimmed)] {
		return "", fmt.Errorf("invalid skill name %q: reserved Windows device name", name)
	}
	return trimmed, nil
}

// SanitizeDescription collapses line breaks/whitespace into one concise description line (<= 240 chars).
func SanitizeDescription(desc string) (string, error) {
	s := strings.Join(strings.Fields(strings.TrimSpace(desc)), " ")
	if s == "" {
		return "", errors.New("skill description cannot be empty")
	}
	if len(s) > 240 {
		s = strings.TrimSpace(s[:240])
	}
	return s, nil
}

// LoadCatalog discovers skills in skillsDir, keeping invalid entries recorded with reasons.
func LoadCatalog(skillsDir, dataDir string) Catalog {
	cat := Catalog{SkillsDir: skillsDir, DataDir: dataDir}
	if strings.TrimSpace(skillsDir) == "" {
		return cat
	}
	entries, err := os.ReadDir(skillsDir)
	if err != nil {
		return cat
	}
	state := loadStateFile(dataDir)

	sort.Slice(entries, func(i, j int) bool {
		return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
	})

	for _, de := range entries {
		dirName := de.Name()
		if strings.HasPrefix(dirName, ".") {
			continue
		}
		dirPath := filepath.Join(skillsDir, dirName)
		if !de.IsDir() {
			continue
		}
		if _, err := ValidateSkillName(dirName); err != nil {
			cat.Invalid = append(cat.Invalid, InvalidSkill{
				Name:   dirName,
				Dir:    dirPath,
				Reason: err.Error(),
			})
			continue
		}
		sk, err := loadOneSkill(dirPath, dirName)
		if err != nil {
			cat.Invalid = append(cat.Invalid, InvalidSkill{
				Name:   dirName,
				Dir:    dirPath,
				Reason: err.Error(),
			})
			continue
		}
		st := state.Skills[sk.Name]
		sk.Enabled = !st.Disabled
		sk.UseCount = st.UseCount
		sk.LastUsed = st.LastUsed
		cat.Valid = append(cat.Valid, sk)
	}
	return cat
}

func loadOneSkill(dirPath, dirName string) (Skill, error) {
	skillFile := filepath.Join(dirPath, "SKILL.md")
	rawBytes, err := os.ReadFile(skillFile)
	if err != nil {
		return Skill{}, fmt.Errorf("missing or unreadable SKILL.md: %v", err)
	}
	if len(rawBytes) > MaxTotalSkillBytes {
		return Skill{}, fmt.Errorf("SKILL.md exceeds %d byte cap (%d bytes)", MaxTotalSkillBytes, len(rawBytes))
	}
	if !utf8.Valid(rawBytes) {
		return Skill{}, errors.New("SKILL.md is not valid UTF-8")
	}
	raw := strings.ReplaceAll(string(rawBytes), "\r\n", "\n")
	if strings.ContainsRune(raw, 0) {
		return Skill{}, errors.New("SKILL.md contains NUL byte")
	}

	fmName, fmDesc, modelAuthored, body, err := parseFrontMatter(raw)
	if err != nil {
		return Skill{}, err
	}
	if _, err := ValidateSkillName(fmName); err != nil {
		return Skill{}, err
	}
	if fmName != dirName {
		return Skill{}, fmt.Errorf("front matter name %q does not match folder name %q", fmName, dirName)
	}
	desc, err := SanitizeDescription(fmDesc)
	if err != nil {
		return Skill{}, err
	}
	if strings.TrimSpace(body) == "" {
		return Skill{}, errors.New("SKILL.md markdown body cannot be empty")
	}

	var extras []string
	totalBytes := len(rawBytes)
	_ = filepath.WalkDir(dirPath, func(p string, d os.DirEntry, wErr error) error {
		if wErr != nil || d.IsDir() {
			return nil
		}
		rel, rErr := filepath.Rel(dirPath, p)
		if rErr != nil {
			return nil
		}
		relSlash := filepath.ToSlash(rel)
		if strings.EqualFold(relSlash, "SKILL.md") {
			return nil
		}
		if info, iErr := d.Info(); iErr == nil {
			totalBytes += int(info.Size())
		}
		extras = append(extras, relSlash)
		return nil
	})
	sort.Strings(extras)

	return Skill{
		Name:          fmName,
		Description:   desc,
		Body:          strings.TrimSpace(body),
		RawMarkdown:   raw,
		Dir:           dirPath,
		SkillFile:     skillFile,
		ExtraFiles:    extras,
		ModelAuthored: modelAuthored,
		Enabled:       true,
		TotalBytes:    totalBytes,
	}, nil
}

func parseFrontMatter(raw string) (name, desc string, modelAuthored bool, body string, err error) {
	trimmed := strings.TrimPrefix(raw, "\uFEFF")
	if !strings.HasPrefix(trimmed, "---\n") {
		return "", "", false, "", errors.New("SKILL.md must begin with YAML front matter '---'")
	}
	rest := trimmed[4:]
	endIdx := strings.Index(rest, "\n---")
	if endIdx < 0 {
		return "", "", false, "", errors.New("SKILL.md front matter missing closing '---'")
	}
	fmBlock := rest[:endIdx]
	after := rest[endIdx+4:]
	if strings.HasPrefix(after, "\n") {
		after = after[1:]
	} else if after != "" {
		return "", "", false, "", errors.New("closing '---' must be followed by a newline")
	}

	fields := make(map[string]string)
	for _, line := range strings.Split(fmBlock, "\n") {
		ln := strings.TrimSpace(line)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		colon := strings.IndexByte(ln, ':')
		if colon <= 0 {
			return "", "", false, "", fmt.Errorf("invalid front matter line: %q", ln)
		}
		k := strings.ToLower(strings.TrimSpace(ln[:colon]))
		v := unquoteYAML(strings.TrimSpace(ln[colon+1:]))
		fields[k] = v
	}

	name = strings.TrimSpace(fields["name"])
	if name == "" {
		return "", "", false, "", errors.New("front matter missing required 'name'")
	}
	desc = strings.TrimSpace(fields["description"])
	if desc == "" {
		return "", "", false, "", errors.New("front matter missing required 'description'")
	}
	ma := strings.ToLower(strings.TrimSpace(fields["model-authored"]))
	if ma == "" {
		ma = strings.ToLower(strings.TrimSpace(fields["model_authored"]))
	}
	modelAuthored = ma == "true" || ma == "yes" || ma == "1"
	return name, desc, modelAuthored, after, nil
}

func unquoteYAML(s string) string {
	if len(s) >= 2 {
		if s[0] == '"' && s[len(s)-1] == '"' {
			inner := s[1 : len(s)-1]
			inner = strings.ReplaceAll(inner, `\"`, `"`)
			inner = strings.ReplaceAll(inner, `\\`, `\`)
			return inner
		}
		if s[0] == '\'' && s[len(s)-1] == '\'' {
			return strings.ReplaceAll(s[1:len(s)-1], "''", "'")
		}
	}
	return s
}

func yamlQuote(s string) string {
	escaped := strings.ReplaceAll(s, `\`, `\\`)
	escaped = strings.ReplaceAll(escaped, `"`, `\"`)
	return `"` + escaped + `"`
}

// RenderSkillMarkdown builds canonical SKILL.md bytes with YAML front matter and body.
func RenderSkillMarkdown(name, desc, body string, modelAuthored bool) string {
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString("name: ")
	b.WriteString(name)
	b.WriteByte('\n')
	b.WriteString("description: ")
	b.WriteString(yamlQuote(desc))
	b.WriteByte('\n')
	if modelAuthored {
		b.WriteString("model-authored: true\n")
	}
	b.WriteString("---\n\n")
	b.WriteString(strings.TrimSpace(body))
	b.WriteByte('\n')
	return b.String()
}

// Lookup finds an enabled valid skill by name (or all valid skills if includeDisabled is true).
func (c *Catalog) Lookup(name string, includeDisabled bool) (Skill, bool) {
	if c == nil {
		return Skill{}, false
	}
	for _, s := range c.Valid {
		if s.Name == name {
			if !includeDisabled && !s.Enabled {
				return Skill{}, false
			}
			return s, true
		}
	}
	return Skill{}, false
}

// SystemPromptBlock builds the compact progressive-loading skill catalog for the system prompt.
func (c *Catalog) SystemPromptBlock() (block string, tokenEst int, shown int) {
	if c == nil {
		return "", 0, 0
	}
	var enabled []Skill
	for _, s := range c.Valid {
		if s.Enabled {
			enabled = append(enabled, s)
		}
	}
	if len(enabled) == 0 {
		return "", 0, 0
	}

	var b strings.Builder
	b.WriteString("Available skills (call `use_skill` with a skill's exact name to load full instructions; read skill files with `read_file`):\n")
	b.WriteString("<available_skills>\n")
	footer := "</available_skills>"

	for i, s := range enabled {
		if shown >= MaxPromptSkills {
			c.TruncatedCount = len(enabled) - i
			break
		}
		line := fmt.Sprintf("- %s: %s\n", s.Name, s.Description)
		if b.Len()+len(line)+len(footer)+64 > MaxPromptBytes {
			c.TruncatedCount = len(enabled) - i
			break
		}
		b.WriteString(line)
		shown++
	}
	if c.TruncatedCount > 0 {
		b.WriteString(fmt.Sprintf("(%d more skills omitted; open /skills to inspect all)\n", c.TruncatedCount))
	}
	b.WriteString(footer)
	out := b.String()
	return out, (len(out) + 3) / 4, shown
}

func stateFilePath(dataDir string) string {
	return filepath.Join(dataDir, "skill-state.json")
}

func loadStateFile(dataDir string) skillStateFile {
	out := skillStateFile{Skills: make(map[string]skillStateEntry)}
	if strings.TrimSpace(dataDir) == "" {
		return out
	}
	b, err := os.ReadFile(stateFilePath(dataDir))
	if err != nil {
		return out
	}
	if err := json.Unmarshal(b, &out); err != nil || out.Skills == nil {
		out.Skills = make(map[string]skillStateEntry)
	}
	return out
}

func saveStateFile(dataDir string, st skillStateFile) error {
	if strings.TrimSpace(dataDir) == "" {
		return nil
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(stateFilePath(dataDir), b, 0o644)
}

// RecordUse increments usage count and updates LastUsed for a skill.
func RecordUse(dataDir, name string) error {
	st := loadStateFile(dataDir)
	ent := st.Skills[name]
	ent.UseCount++
	ent.LastUsed = time.Now().UTC()
	st.Skills[name] = ent
	return saveStateFile(dataDir, st)
}

// SetEnabled enables or disables a skill in persistent state.
func SetEnabled(dataDir, name string, enabled bool) error {
	st := loadStateFile(dataDir)
	ent := st.Skills[name]
	ent.Disabled = !enabled
	st.Skills[name] = ent
	return saveStateFile(dataDir, st)
}

// CreateSkill writes a validated user-created skill to skills/<name>/SKILL.md.
func CreateSkill(skillsDir, name, desc, body string) error {
	cleanName, err := ValidateSkillName(name)
	if err != nil {
		return err
	}
	cleanDesc, err := SanitizeDescription(desc)
	if err != nil {
		return err
	}
	if strings.TrimSpace(body) == "" {
		return errors.New("skill body cannot be empty")
	}
	if len(body) > MaxBodyBytes {
		return fmt.Errorf("skill body exceeds %d byte cap", MaxBodyBytes)
	}
	targetDir := filepath.Join(skillsDir, cleanName)
	if _, err := os.Stat(targetDir); err == nil {
		return fmt.Errorf("skill %q already exists", cleanName)
	}
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return err
	}
	content := RenderSkillMarkdown(cleanName, cleanDesc, body, false)
	return os.WriteFile(filepath.Join(targetDir, "SKILL.md"), []byte(content), 0o644)
}

// RenameSkill renames an existing valid skill folder, updates its front matter name, and migrates state.
func RenameSkill(skillsDir, dataDir, oldName, newName string) error {
	cleanNew, err := ValidateSkillName(newName)
	if err != nil {
		return err
	}
	if oldName == cleanNew {
		return nil
	}
	oldDir := filepath.Join(skillsDir, oldName)
	newDir := filepath.Join(skillsDir, cleanNew)
	if _, err := os.Stat(oldDir); err != nil {
		return fmt.Errorf("skill %q not found", oldName)
	}
	if _, err := os.Stat(newDir); err == nil {
		return fmt.Errorf("skill %q already exists", cleanNew)
	}
	sk, err := loadOneSkill(oldDir, oldName)
	if err != nil {
		return err
	}
	if err := os.Rename(oldDir, newDir); err != nil {
		return err
	}
	updated := RenderSkillMarkdown(cleanNew, sk.Description, sk.Body, sk.ModelAuthored)
	if err := os.WriteFile(filepath.Join(newDir, "SKILL.md"), []byte(updated), 0o644); err != nil {
		return err
	}
	st := loadStateFile(dataDir)
	if ent, ok := st.Skills[oldName]; ok {
		delete(st.Skills, oldName)
		st.Skills[cleanNew] = ent
		_ = saveStateFile(dataDir, st)
	}
	return nil
}

// DeleteSkill moves a skill (valid or invalid) into .alpaca/trash/<ts>-skill-<name>/ instead of hard deletion.
func DeleteSkill(skillsDir, dataDir, name string) (string, error) {
	srcDir := filepath.Join(skillsDir, name)
	if _, err := os.Stat(srcDir); err != nil {
		return "", err
	}
	trashDir := filepath.Join(dataDir, "trash", fmt.Sprintf("%s-skill-%s", time.Now().UTC().Format("20060102-150405.000"), name))
	if err := os.MkdirAll(filepath.Dir(trashDir), 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(srcDir, trashDir); err != nil {
		return "", err
	}
	return trashDir, nil
}
