package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/skills"
)

func TestAgentLoopSkillsUseReadApproveOverwriteReject(t *testing.T) {
	ws := t.TempDir()
	skillsDir := filepath.Join(t.TempDir(), "skills")
	dataDir := skills.DataRoot(skillsDir, ws)

	if err := skills.CreateSkill(skillsDir, "helper-skill", "Helper workflow.", "# Helper\nRead extra file notes/ref.txt."); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(skillsDir, "helper-skill", "notes"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillsDir, "helper-skill", "notes", "ref.txt"), []byte("REF_LINE_1\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var step atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := step.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		writeCall := func(id, name, args string) {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":%q,\"type\":\"function\",\"function\":{\"name\":%q,\"arguments\":%q}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", id, name, args)
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
		switch n {
		case 1:
			writeCall("c1", "use_skill", `{"name":"helper-skill"}`)
		case 2:
			writeCall("c2", "read_file", `{"path":"skills/helper-skill/notes/ref.txt"}`)
		case 3:
			// Direct write_file to skills/ must still be rejected by sandbox!
			writeCall("c3", "write_file", `{"path":"skills/hacked/SKILL.md","content":"evil"}`)
		case 4:
			// Stage a brand-new skill; test rejects first.
			args, _ := json.Marshal(map[string]any{
				"name":        "auto-check",
				"description": "Checks Go tests cleanly.",
				"body":        "# Auto Check v1\nRun go test ./...",
				"files":       map[string]string{"check.md": "extra"},
			})
			writeCall("c4", "save_skill", string(args))
		case 5:
			// Stage again after rejection feedback; test approves.
			args, _ := json.Marshal(map[string]any{
				"name":        "auto-check",
				"description": "Checks Go tests cleanly.",
				"body":        "# Auto Check v2\nRun go test -count=1 ./...",
			})
			writeCall("c5", "save_skill", string(args))
		case 6:
			// Overwrite auto-check; test approves so history is archived.
			args, _ := json.Marshal(map[string]any{
				"name":        "auto-check",
				"description": "Checks Go tests with vet.",
				"body":        "# Auto Check v3\nRun go vet and go test.",
			})
			writeCall("c6", "save_skill", string(args))
		default:
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"SKILLS_FLOW_DONE\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer srv.Close()

	reg := NewDefaultRegistry()
	pe := NewPermissionEngine(ModeWorkspaceAuto)
	var savePrompts []SkillStagePrompt
	stageCalls := 0

	res := RunTurn(context.Background(), LoopRequest{
		BaseURL:     srv.URL,
		History:     []runner.APIMessage{{Role: "user", Content: "exercise skills"}},
		Registry:    reg,
		Permissions: pe,
		ToolCtx: ToolContext{
			WorkspaceRoot: ws,
			SkillsDir:     skillsDir,
			SkillsDataDir: dataDir,
			SessionID:     "s-skill",
			TurnNumber:    1,
			Checkpoints:   NewCheckpointStore(ws, "s-skill"),
			StageSkill: func(ctx context.Context, pr SkillStagePrompt) (SkillStageResponse, error) {
				stageCalls++
				savePrompts = append(savePrompts, pr)
				if stageCalls == 1 {
					return SkillStageResponse{Approved: false, Reason: "add -count=1 flag"}, nil
				}
				return SkillStageResponse{Approved: true, Name: pr.Name, Desc: pr.Description, Body: pr.Body}, nil
			},
		},
		AskPermission: func(ctx context.Context, req PermissionPrompt) (PermissionResponse, error) {
			return PermissionResponse{Choice: PermDeny}, nil
		},
	})

	if res.Err != nil {
		t.Fatalf("RunTurn err: %v", res.Err)
	}
	if !strings.Contains(res.FinalContent, "SKILLS_FLOW_DONE") {
		t.Fatalf("unexpected FinalContent: %q", res.FinalContent)
	}
	if stageCalls != 3 {
		t.Fatalf("expected 3 StageSkill calls (reject, approve v2, overwrite v3), got %d", stageCalls)
	}
	if !savePrompts[2].Overwrites || savePrompts[2].Diff == "" {
		t.Fatalf("expected 3rd StageSkill prompt to carry Overwrites=true and non-empty Diff, got %+v", savePrompts[2])
	}

	// Verify read_file succeeded on extra skill file and direct write_file to skills/ did NOT happen.
	if _, err := os.Stat(filepath.Join(ws, "skills", "hacked")); !os.IsNotExist(err) {
		t.Fatalf("write_file created skills/hacked despite protection!")
	}
	var sawRef, sawRejectedErr, sawProtectedErr bool
	for _, m := range res.MessagesAppended {
		if m.Role == "tool" {
			if strings.Contains(m.Content, "REF_LINE_1") {
				sawRef = true
			}
			if strings.Contains(m.Content, "add -count=1 flag") {
				sawRejectedErr = true
			}
			if strings.Contains(m.Content, "cannot write to skills/") || strings.Contains(m.Content, "protected") {
				sawProtectedErr = true
			}
		}
	}
	if !sawRef || !sawRejectedErr || !sawProtectedErr {
		t.Fatalf("missing expected tool outcomes: sawRef=%v sawRejectedErr=%v sawProtectedErr=%v", sawRef, sawRejectedErr, sawProtectedErr)
	}

	// Verify final saved skill has model-authored: true and .llm/skill-history/ archived v2.
	cat := skills.LoadCatalog(skillsDir, dataDir)
	sk, ok := cat.Lookup("auto-check", false)
	if !ok || !sk.ModelAuthored || !strings.Contains(sk.Body, "Auto Check v3") {
		t.Fatalf("expected saved auto-check v3 model-authored skill, got ok=%v sk=%+v", ok, sk)
	}
	histEntries, err := os.ReadDir(filepath.Join(dataDir, "skill-history", "auto-check"))
	if err != nil || len(histEntries) != 1 {
		t.Fatalf("expected 1 archived version in skill-history/auto-check, err=%v len=%d", err, len(histEntries))
	}
}
