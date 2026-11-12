package skills

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStarterSkillsSeedAndDeleteDoesNotReseed(t *testing.T) {
	root := t.TempDir()
	skillsDir := filepath.Join(root, "skills")
	dataDir := filepath.Join(root, ".alpaca")

	if err := EnsureStarterSkills(skillsDir, dataDir); err != nil {
		t.Fatalf("EnsureStarterSkills: %v", err)
	}
	cat := LoadCatalog(skillsDir, dataDir)
	if len(cat.Invalid) != 0 {
		t.Fatalf("unexpected invalid starter skills: %+v", cat.Invalid)
	}
	if len(cat.Valid) != 3 {
		t.Fatalf("expected 3 starter skills, got %d", len(cat.Valid))
	}

	// Delete one starter skill to trash and re-run EnsureStarterSkills: it must stay deleted.
	trash, err := DeleteSkill(skillsDir, dataDir, "scaffold-project")
	if err != nil {
		t.Fatalf("DeleteSkill: %v", err)
	}
	if _, err := os.Stat(filepath.Join(trash, "SKILL.md")); err != nil {
		t.Fatalf("expected trashed SKILL.md at %s: %v", trash, err)
	}
	if err := EnsureStarterSkills(skillsDir, dataDir); err != nil {
		t.Fatalf("second EnsureStarterSkills: %v", err)
	}
	cat2 := LoadCatalog(skillsDir, dataDir)
	if len(cat2.Valid) != 2 {
		t.Fatalf("deleted starter skill was resurrected; valid=%d", len(cat2.Valid))
	}
}

func TestValidationRecordsInvalidSkillsWithoutFailing(t *testing.T) {
	root := t.TempDir()
	skillsDir := filepath.Join(root, "skills")
	dataDir := filepath.Join(root, ".alpaca")
	if err := os.MkdirAll(skillsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Valid skill
	if err := CreateSkill(skillsDir, "good-skill", "Does clean things.", "# Steps\n1. Check files."); err != nil {
		t.Fatalf("CreateSkill: %v", err)
	}

	// Invalid cases
	badCases := map[string]string{
		"BadUpper":       "---\nname: BadUpper\ndescription: uppercase\n---\nBody",
		"missing-file":   "", // no SKILL.md created
		"name-mismatch":  "---\nname: other-name\ndescription: mismatch\n---\nBody",
		"empty-desc":     "---\nname: empty-desc\ndescription: \n---\nBody",
		"empty-body":     "---\nname: empty-body\ndescription: ok\n---\n   \n",
		"no-frontmatter": "# Just markdown without frontmatter",
		"unclosed-fm":    "---\nname: unclosed-fm\ndescription: ok\nBody",
	}
	for dirName, content := range badCases {
		d := filepath.Join(skillsDir, dirName)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if content != "" {
			if err := os.WriteFile(filepath.Join(d, "SKILL.md"), []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}

	cat := LoadCatalog(skillsDir, dataDir)
	if len(cat.Valid) != 1 || cat.Valid[0].Name != "good-skill" {
		t.Fatalf("expected 1 valid skill good-skill, got %+v", cat.Valid)
	}
	if len(cat.Invalid) != len(badCases) {
		t.Fatalf("expected %d invalid skills, got %d: %+v", len(badCases), len(cat.Invalid), cat.Invalid)
	}
	for _, inv := range cat.Invalid {
		if strings.TrimSpace(inv.Reason) == "" {
			t.Fatalf("invalid skill %s missing reason", inv.Name)
		}
	}
}

func TestProgressivePromptAndStateTracking(t *testing.T) {
	root := t.TempDir()
	skillsDir := filepath.Join(root, "skills")
	dataDir := filepath.Join(root, ".alpaca")
	if err := CreateSkill(skillsDir, "alpha", "Alpha workflow summary.", "# Alpha Body\nSECRET_ALPHA_BODY"); err != nil {
		t.Fatal(err)
	}
	if err := CreateSkill(skillsDir, "beta", "Beta workflow summary.", "# Beta Body"); err != nil {
		t.Fatal(err)
	}
	if err := SetEnabled(dataDir, "beta", false); err != nil {
		t.Fatal(err)
	}
	if err := RecordUse(dataDir, "alpha"); err != nil {
		t.Fatal(err)
	}

	cat := LoadCatalog(skillsDir, dataDir)
	block, tok, shown := cat.SystemPromptBlock()
	if shown != 1 || tok <= 0 {
		t.Fatalf("expected 1 shown skill & >0 tokens, got shown=%d tok=%d", shown, tok)
	}
	if !strings.Contains(block, "alpha: Alpha workflow summary.") {
		t.Fatalf("missing alpha summary in system prompt block: %s", block)
	}
	if strings.Contains(block, "SECRET_ALPHA_BODY") {
		t.Fatalf("system prompt block must not include full skill body!")
	}
	if strings.Contains(block, "beta") {
		t.Fatalf("disabled skill beta should not appear in system prompt block: %s", block)
	}

	sk, ok := cat.Lookup("alpha", false)
	if !ok || sk.UseCount != 1 || sk.LastUsed.IsZero() {
		t.Fatalf("expected alpha use count 1 & LastUsed set, got ok=%v sk=%+v", ok, sk)
	}
	if _, ok := cat.Lookup("beta", false); ok {
		t.Fatalf("disabled skill should not be returned by Lookup(includeDisabled=false)")
	}
}

func TestStagingHostileNamesOversizedApproveOverwriteReject(t *testing.T) {
	root := t.TempDir()
	skillsDir := filepath.Join(root, "skills")
	dataDir := filepath.Join(root, ".alpaca")

	// Hostile names & oversized inputs must fail closed.
	hostile := []StageRequest{
		{Name: "../escape", Description: "ok", Body: "body"},
		{Name: "UpperCase", Description: "ok", Body: "body"},
		{Name: "has space", Description: "ok", Body: "body"},
		{Name: "con", Description: "ok", Body: "body"},
		{Name: "nul", Description: "ok", Body: "body"},
		{Name: strings.Repeat("a", 41), Description: "ok", Body: "body"},
		{Name: "valid-name", Description: "ok", Body: strings.Repeat("x", MaxBodyBytes+1)},
		{Name: "valid-name", Description: "ok", Body: "body", Files: map[string]string{"../evil.txt": "bad"}},
		{Name: "valid-name", Description: "ok", Body: "body", Files: map[string]string{"SKILL.md": "bad"}},
		{Name: "valid-name", Description: "ok", Body: "body", Files: map[string]string{"CON.txt": "bad"}},
		{Name: "valid-name", Description: "ok", Body: "body", Files: map[string]string{
			"f1": "a", "f2": "a", "f3": "a", "f4": "a", "f5": "a",
			"f6": "a", "f7": "a", "f8": "a", "f9": "a", "f10": "a", "f11": "a",
		}},
	}
	for i, req := range hostile {
		if _, err := PrepareStaging(skillsDir, dataDir, req); err == nil {
			t.Fatalf("case %d (%+v): expected PrepareStaging error, got nil", i, req.Name)
		}
	}

	// Reject cleanup leaves skills/ untouched.
	stagedReject, err := PrepareStaging(skillsDir, dataDir, StageRequest{
		Name:          "temp-skill",
		Description:   "Temp desc",
		Body:          "# Temp v1",
		ModelAuthored: true,
	})
	if err != nil {
		t.Fatalf("PrepareStaging reject case: %v", err)
	}
	if _, err := os.Stat(filepath.Join(skillsDir, "temp-skill")); err == nil {
		t.Fatalf("staged skill must NOT exist in skills/ before approval")
	}
	DiscardStaged(stagedReject)
	if _, err := os.Stat(stagedReject.StagingDir); !os.IsNotExist(err) {
		t.Fatalf("DiscardStaged should clean up staging dir")
	}

	// Stage -> Approve -> Overwrite with .alpaca/skill-history preservation.
	staged1, err := PrepareStaging(skillsDir, dataDir, StageRequest{
		Name:          "deploy-check",
		Description:   "Check deployment readiness.",
		Body:          "# Deploy Check v1\n1. Run tests.",
		Files:         map[string]string{"notes/checklist.md": "- tests pass\n"},
		ModelAuthored: true,
	})
	if err != nil {
		t.Fatalf("PrepareStaging v1: %v", err)
	}
	res1, err := CommitStaged(skillsDir, dataDir, staged1, staged1.Name, staged1.Description, staged1.Body)
	if err != nil {
		t.Fatalf("CommitStaged v1: %v", err)
	}
	if res1.Overwrote {
		t.Fatalf("expected Overwrote=false on initial create")
	}

	cat := LoadCatalog(skillsDir, dataDir)
	sk1, ok := cat.Lookup("deploy-check", false)
	if !ok || !sk1.ModelAuthored || len(sk1.ExtraFiles) != 1 || sk1.ExtraFiles[0] != "notes/checklist.md" {
		t.Fatalf("unexpected v1 skill loaded: ok=%v sk=%+v", ok, sk1)
	}

	// Overwrite with edited body -> previous archived in .alpaca/skill-history/
	staged2, err := PrepareStaging(skillsDir, dataDir, StageRequest{
		Name:          "deploy-check",
		Description:   "Updated readiness checklist.",
		Body:          "# Deploy Check v2\n1. Run tests.\n2. Tag release.",
		ModelAuthored: true,
	})
	if err != nil {
		t.Fatalf("PrepareStaging v2: %v", err)
	}
	if !staged2.Overwrites || !strings.Contains(staged2.ExistingMD, "Deploy Check v1") {
		t.Fatalf("expected staged2.Overwrites=true and ExistingMD containing v1, got %+v", staged2)
	}
	res2, err := CommitStaged(skillsDir, dataDir, staged2, staged2.Name, staged2.Description, staged2.Body)
	if err != nil {
		t.Fatalf("CommitStaged v2: %v", err)
	}
	if !res2.Overwrote || res2.HistoryPath == "" {
		t.Fatalf("expected Overwrote=true and non-empty HistoryPath, got %+v", res2)
	}
	oldHistBytes, err := os.ReadFile(filepath.Join(res2.HistoryPath, "SKILL.md"))
	if err != nil || !strings.Contains(string(oldHistBytes), "Deploy Check v1") {
		t.Fatalf("expected archived v1 in %s, err=%v", res2.HistoryPath, err)
	}

	// Rename round trip
	if err := RenameSkill(skillsDir, dataDir, "deploy-check", "release-check"); err != nil {
		t.Fatalf("RenameSkill: %v", err)
	}
	catAfter := LoadCatalog(skillsDir, dataDir)
	skRenamed, ok := catAfter.Lookup("release-check", false)
	if !ok || !skRenamed.ModelAuthored {
		t.Fatalf("expected renamed release-check skill to remain valid & model-authored: ok=%v sk=%+v", ok, skRenamed)
	}
}
