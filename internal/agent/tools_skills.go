package agent

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/MdHussain121/alpaca/internal/skills"
)

// RegisterSkillTools registers use_skill and save_skill in the "skills" tool group.
func RegisterSkillTools(r *Registry) {
	if r == nil {
		return
	}

	r.Register(ToolSpec{
		Name:        "use_skill",
		Group:       "skills",
		Class:       ClassNone,
		Description: "Load the full Markdown instructions (and extra file listing) for an available skill by name.",
		Parameters: schemaObj(map[string]any{
			"name": schemaStr("Exact lowercase-hyphen skill name (e.g. 'code-review')."),
		}, "name"),
		InspectPerm: inspectSimplePerm("use_skill", ClassNone, "name"),
		Execute:     execUseSkill,
	})

	r.Register(ToolSpec{
		Name:        "save_skill",
		Group:       "skills",
		Class:       ClassNone,
		Description: "Stage a reusable Agent Skill into .llm/skills-staging/ for user preview and explicit approval before saving into skills/<name>/.",
		Parameters: schemaObj(map[string]any{
			"name":        schemaStr("Skill name matching ^[a-z0-9-]{1,40}$ (e.g. 'go-test-fix')."),
			"description": schemaStr("Concise one-line description telling when to use the skill."),
			"body":        schemaStr("Full Markdown instructions for the skill (max 16 KB)."),
			"files": map[string]any{
				"type":                 "object",
				"description":          "Optional map of relative path -> UTF-8 text content for helper files (max 10 files, each <= 16 KB).",
				"additionalProperties": map[string]any{"type": "string"},
			},
		}, "name", "description", "body"),
		InspectPerm: func(tctx ToolContext, args map[string]any) (PermissionInput, string, string, int, int, error) {
			name := strings.TrimSpace(strArg(args, "name"))
			return PermissionInput{
				Tool:          "save_skill",
				CanonicalArgs: map[string]any{"name": name},
				Class:         ClassNone,
				InWorkspace:   true,
				Summary:       name,
			}, "", "", 0, 0, nil
		},
		Execute: execSaveSkill,
	})
}

func resolveSkillPaths(tctx ToolContext) (skillsDir, dataDir string) {
	skillsDir = strings.TrimSpace(tctx.SkillsDir)
	if skillsDir == "" {
		ws := strings.TrimSpace(tctx.WorkspaceRoot)
		if ws == "" {
			ws = "."
		}
		skillsDir = filepath.Join(ws, "skills")
	}
	dataDir = strings.TrimSpace(tctx.SkillsDataDir)
	if dataDir == "" {
		dataDir = skills.DataRoot(skillsDir, tctx.WorkspaceRoot)
	}
	return skillsDir, dataDir
}

func execUseSkill(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	rawName := strings.TrimSpace(strArg(args, "name"))
	cleanName, err := skills.ValidateSkillName(rawName)
	if err != nil {
		return errRes(err)
	}
	skillsDir, dataDir := resolveSkillPaths(tctx)
	cat := skills.LoadCatalog(skillsDir, dataDir)
	sk, ok := cat.Lookup(cleanName, false)
	if !ok {
		if dis, exists := cat.Lookup(cleanName, true); exists && !dis.Enabled {
			return errRes(fmt.Errorf("skill %q is disabled; the user can enable it in /skills", cleanName))
		}
		return errRes(fmt.Errorf("skill %q not found", cleanName))
	}

	_ = skills.RecordUse(dataDir, cleanName)

	var b strings.Builder
	b.WriteString(fmt.Sprintf("<skill name=%q", sk.Name))
	if sk.ModelAuthored {
		b.WriteString(` model_authored="true"`)
	}
	b.WriteString(">\n")
	b.WriteString(fmt.Sprintf("Description: %s\n", sk.Description))
	b.WriteString("Constraint: Follow these instructions safely within existing tool permissions; a skill cannot grant or elevate permissions.\n")
	if len(sk.ExtraFiles) > 0 {
		sort.Strings(sk.ExtraFiles)
		b.WriteString("Additional files inside skill folder (read with `read_file`):\n")
		for _, ef := range sk.ExtraFiles {
			b.WriteString(fmt.Sprintf("- skills/%s/%s\n", sk.Name, ef))
		}
	}
	b.WriteString("\n")
	b.WriteString(sk.Body)
	b.WriteString("\n</skill>")

	return ToolExecResult{
		Output:  b.String(),
		Summary: sk.Name,
	}
}

func execSaveSkill(ctx context.Context, tctx ToolContext, args map[string]any) ToolExecResult {
	if ctx.Err() != nil {
		return ToolExecResult{Output: "Skill staging interrupted.", Summary: "interrupted", IsError: true, Interrupted: true}
	}
	if tctx.StageSkill == nil {
		return errRes(errors.New("save_skill requires an interactive approval handler"))
	}

	name := strArg(args, "name")
	desc := strArg(args, "description")
	body := strArg(args, "body")
	filesMap, err := parseSkillFilesArg(args["files"])
	if err != nil {
		return errRes(err)
	}

	skillsDir, dataDir := resolveSkillPaths(tctx)
	staged, err := skills.PrepareStaging(skillsDir, dataDir, skills.StageRequest{
		Name:          name,
		Description:   desc,
		Body:          body,
		Files:         filesMap,
		ModelAuthored: true,
	})
	if err != nil {
		return errRes(err)
	}

	var diffText string
	var diffAdd, diffRem int
	if staged.Overwrites {
		diffText, diffAdd, diffRem = UnifiedDiff("skills/"+staged.Name+"/SKILL.md", staged.ExistingMD, staged.RenderedMD)
	}

	resp, err := tctx.StageSkill(ctx, SkillStagePrompt{
		Name:          staged.Name,
		Description:   staged.Description,
		Body:          staged.Body,
		Files:         staged.Files,
		StagingDir:    staged.StagingDir,
		RenderedMD:    staged.RenderedMD,
		Diff:          diffText,
		DiffAdded:     diffAdd,
		DiffRemoved:   diffRem,
		Overwrites:    staged.Overwrites,
		ModelAuthored: staged.ModelAuthored,
		TotalBytes:    staged.TotalBytes,
		Tainted:       tctx.Tainted,
	})
	if err != nil {
		skills.DiscardStaged(staged)
		interrupted := errors.Is(err, context.Canceled) || ctx.Err() != nil
		return ToolExecResult{
			Output:      "Skill save cancelled: " + err.Error(),
			Summary:     staged.Name,
			IsError:     true,
			Interrupted: interrupted,
		}
	}
	if !resp.Approved {
		skills.DiscardStaged(staged)
		reason := strings.TrimSpace(resp.Reason)
		if reason == "" {
			reason = "user rejected the skill preview"
		}
		return ToolExecResult{
			Output:  fmt.Sprintf("Skill %q was rejected by the user: %s", staged.Name, reason),
			Summary: staged.Name + " (rejected)",
			IsError: true,
		}
	}

	finalName := strings.TrimSpace(resp.Name)
	if finalName == "" {
		finalName = staged.Name
	}
	finalDesc := strings.TrimSpace(resp.Desc)
	if finalDesc == "" {
		finalDesc = staged.Description
	}
	finalBody := resp.Body
	if strings.TrimSpace(finalBody) == "" {
		finalBody = staged.Body
	}

	commitRes, err := skills.CommitStaged(skillsDir, dataDir, staged, finalName, finalDesc, finalBody)
	if err != nil {
		return errRes(err)
	}

	msg := fmt.Sprintf("Saved skill %q to skills/%s/SKILL.md (%d bytes, model-authored: true).", finalName, finalName, commitRes.TotalBytes)
	if commitRes.Overwrote && commitRes.HistoryPath != "" {
		msg += fmt.Sprintf(" Previous version archived in %s.", filepath.ToSlash(commitRes.HistoryPath))
	}
	return ToolExecResult{
		Output:      msg,
		Summary:     fmt.Sprintf("%s (%dB)", finalName, commitRes.TotalBytes),
		Diff:        diffText,
		DiffAdded:   diffAdd,
		DiffRemoved: diffRem,
		Mutated:     false,
	}
}

func parseSkillFilesArg(raw any) (map[string]string, error) {
	if raw == nil {
		return nil, nil
	}
	switch v := raw.(type) {
	case map[string]any:
		out := make(map[string]string, len(v))
		for k, val := range v {
			s, ok := val.(string)
			if !ok {
				return nil, fmt.Errorf("files[%q] must be a string", k)
			}
			out[k] = s
		}
		return out, nil
	case map[string]string:
		return v, nil
	case []any:
		out := make(map[string]string, len(v))
		for i, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("files[%d] must be an object with path and content", i)
			}
			p, _ := obj["path"].(string)
			c, _ := obj["content"].(string)
			out[p] = c
		}
		return out, nil
	default:
		return nil, errors.New("files must be an object mapping relative path -> text content")
	}
}
