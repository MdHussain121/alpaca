package agent

import "testing"

func TestAlwaysAllowToolCoversDifferentArgsButNotHighRiskOrTaint(t *testing.T) {
	eng := NewPermissionEngine(ModeReadAuto)
	execArgs := func(cmd string) PermissionInput {
		return PermissionInput{
			Tool:          "run_command",
			CanonicalArgs: map[string]any{"command": cmd, "cwd": "."},
			Class:         ClassExec,
			InWorkspace:   true,
		}
	}

	if d, _ := eng.Evaluate(execArgs("go test ./...")); d != DecisionAsk {
		t.Fatalf("expected ask before allowlist, got %s", d)
	}
	eng.AllowToolSession("run_command")
	if d, _ := eng.Evaluate(execArgs("go test ./...")); d != DecisionAllow {
		t.Fatalf("expected allow for same args, got %s", d)
	}
	if d, _ := eng.Evaluate(execArgs("go build ./...")); d != DecisionAllow {
		t.Fatalf("always-allow must cover different args, got %s", d)
	}

	high := execArgs("format c:")
	high.HighRisk = true
	if d, _ := eng.Evaluate(high); d != DecisionAsk {
		t.Fatalf("high-risk must still ask, got %s", d)
	}

	tainted := execArgs("go vet ./...")
	tainted.Tainted = true
	if d, _ := eng.Evaluate(tainted); d != DecisionAsk {
		t.Fatalf("tainted session must still ask, got %s", d)
	}

	entries := eng.Allowlist()
	if len(entries) != 1 || !entries[0].AllArgs || entries[0].Tool != "run_command" {
		t.Fatalf("expected one tool-wide allowlist entry, got %+v", entries)
	}

	eng.ClearAllowlist()
	if d, _ := eng.Evaluate(execArgs("go build ./...")); d != DecisionAsk {
		t.Fatalf("clearing the allowlist must prompt again, got %s", d)
	}
}
