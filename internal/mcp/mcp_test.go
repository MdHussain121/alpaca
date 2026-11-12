package mcp

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MdHussain121/alpaca/internal/agent"
)

func buildEchoMCPBinary(t *testing.T) string {
	t.Helper()
	binDir := t.TempDir()
	binPath := filepath.Join(binDir, "echomcp.exe")
	cmd := exec.Command("go", "build", "-o", binPath, "./testdata/echomcp")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build echomcp failed: %v\n%s", err, string(out))
	}
	return binPath
}

func writeTestMCPToml(t *testing.T, dir, content string) string {
	t.Helper()
	p := filepath.Join(dir, "mcp.toml")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write mcp.toml: %v", err)
	}
	return p
}

func isPIDAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	cmd := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), fmt.Sprintf("%d", pid))
}

func TestMissingConfigIsNoOp(t *testing.T) {
	reg := agent.NewDefaultRegistry()
	mgr := NewManager(t.TempDir(), t.TempDir(), filepath.Join(t.TempDir(), "missing-mcp.toml"), reg)
	defer mgr.Shutdown()

	sum, err := mgr.Reload(context.Background())
	if err != nil {
		t.Fatalf("missing mcp.toml should not error: %v", err)
	}
	if sum.TotalServers != 0 || sum.TotalTools != 0 {
		t.Fatalf("expected 0 servers/tools, got %+v", sum)
	}
}

func TestStdioHandshakeToolsPermissionsTogglesSecretsAndShutdown(t *testing.T) {
	bin := buildEchoMCPBinary(t)
	root := t.TempDir()
	secret := "SUPER_SECRET_TOKEN_987654"
	childPidFile := filepath.Join(root, "child.pid")

	tomlCfg := fmt.Sprintf(`
[servers.echomcp]
command = %q
args = ["-spawn-child", "-child-pid-file", %q]
enabled = true
timeout_s = 1
class = "net"

[servers.echomcp.env]
MCP_SECRET = %q
`, bin, childPidFile, secret)
	cfgPath := writeTestMCPToml(t, root, tomlCfg)

	reg := agent.NewDefaultRegistry()
	mgr := NewManager(root, root, cfgPath, reg)
	defer mgr.Shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	sum, err := mgr.Reload(ctx)
	if err != nil {
		t.Fatalf("Reload failed: %v", err)
	}
	if sum.RunningServers != 1 || sum.TotalEnabledTools != 4 {
		t.Fatalf("expected 1 running server and 4 tools, got %+v", sum)
	}
	if sum.TotalSchemaTokens <= 0 {
		t.Fatalf("expected positive schema token count, got %d", sum.TotalSchemaTokens)
	}

	// Verify secret never appears in Summary or descriptions.
	for _, s := range sum.Servers {
		if strings.Contains(fmt.Sprintf("%+v", s), secret) {
			t.Fatalf("server status leaked env secret!")
		}
	}

	// Verify server.tool naming and default-to-ask + taint across all modes.
	spec, ok := reg.Lookup("echomcp.echo")
	if !ok {
		t.Fatalf("expected echomcp.echo in registry")
	}
	if spec.Group != "mcp" || spec.Class != agent.ClassNet {
		t.Fatalf("unexpected spec Group=%s Class=%s", spec.Group, spec.Class)
	}
	for _, mode := range []agent.PermissionMode{agent.ModeAsk, agent.ModeReadAuto, agent.ModeWorkspaceAuto} {
		pe := agent.NewPermissionEngine(mode)
		in, _, _, _, _, err := spec.InspectPerm(agent.ToolContext{WorkspaceRoot: root}, map[string]any{"text": "hi"})
		if err != nil {
			t.Fatalf("InspectPerm error: %v", err)
		}
		dec, _ := pe.Evaluate(in)
		if dec != agent.DecisionAsk {
			t.Fatalf("mode %s should default MCP tool to ask, got %s", mode, dec)
		}
		in.Tainted = true
		decT, _ := pe.Evaluate(in)
		if decT != agent.DecisionAsk {
			t.Fatalf("tainted %s should require ask, got %s", mode, decT)
		}
	}

	// Call echo tool.
	res := spec.Execute(ctx, agent.ToolContext{}, map[string]any{"text": "hello mcp"})
	if res.IsError || res.Output != "echomcp: hello mcp" {
		t.Fatalf("unexpected echo result: %+v", res)
	}

	// Verify env secret is passed to child yet scrubbed if echoed back.
	probeSpec, ok := reg.Lookup("echomcp.env_probe")
	if !ok {
		t.Fatalf("expected echomcp.env_probe")
	}
	probeRes := probeSpec.Execute(ctx, agent.ToolContext{}, map[string]any{})
	if strings.Contains(probeRes.Output, secret) {
		t.Fatalf("secret leaked in MCP tool output: %q", probeRes.Output)
	}
	if !strings.Contains(probeRes.Output, "[REDACTED]") {
		t.Fatalf("expected [REDACTED] confirming env was passed and redacted, got %q", probeRes.Output)
	}

	// Per-call timeout enforcement (timeout_s = 1, sleep = 2500ms).
	sleepSpec, _ := reg.Lookup("echomcp.sleep")
	sleepRes := sleepSpec.Execute(ctx, agent.ToolContext{}, map[string]any{"ms": 2500})
	if !sleepRes.IsError {
		t.Fatalf("expected timeout error from slow tool call, got %+v", sleepRes)
	}

	// Per-tool toggle.
	en, err := mgr.ToggleTool("echomcp", "add")
	if err != nil || en {
		t.Fatalf("expected echomcp.add disabled, en=%v err=%v", en, err)
	}
	if _, ok := reg.Lookup("echomcp.add"); ok {
		t.Fatalf("echomcp.add should be unregistered when toggled off")
	}

	// Per-server toggle off/on.
	on, err := mgr.ToggleServer(ctx, "echomcp")
	if err != nil || on {
		t.Fatalf("expected echomcp server off, on=%v err=%v", on, err)
	}
	if _, ok := reg.Lookup("echomcp.echo"); ok {
		t.Fatalf("tools should be removed when server disabled")
	}
	on, err = mgr.ToggleServer(ctx, "echomcp")
	if err != nil || !on {
		t.Fatalf("expected echomcp server back on, on=%v err=%v", on, err)
	}
	if _, ok := reg.Lookup("echomcp.echo"); !ok {
		t.Fatalf("echomcp.echo should be back after enabling server")
	}

	// Read grandchild PID spawned by echomcp, then Shutdown and verify 0 orphans!
	pidBytes, err := os.ReadFile(childPidFile)
	if err != nil {
		t.Fatalf("expected child pid file: %v", err)
	}
	var childPID int
	_, _ = fmt.Sscanf(strings.TrimSpace(string(pidBytes)), "%d", &childPID)
	if childPID <= 0 || !isPIDAlive(childPID) {
		t.Fatalf("expected spawned grandchild PID %d to be alive before shutdown", childPID)
	}

	mgr.Shutdown()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if !isPIDAlive(childPID) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if isPIDAlive(childPID) {
		t.Fatalf("orphan grandchild PID %d survived Manager.Shutdown", childPID)
	}
}

func TestManyToolsWarningAndCrashRestartLimit(t *testing.T) {
	bin := buildEchoMCPBinary(t)
	root := t.TempDir()
	crashOnceFile := filepath.Join(root, "crashed.once")
	_ = os.WriteFile(crashOnceFile, []byte("skip"), 0o644) // don't fail initial reload

	tomlCfg := fmt.Sprintf(`
[servers.big]
command = %q
args = ["-many", "12"]
enabled = true

[servers.flaky]
command = %q
args = ["-crash-after-ms", "50"]
enabled = true
class = "exec"
`, bin, bin)
	cfgPath := writeTestMCPToml(t, root, tomlCfg)

	reg := agent.NewDefaultRegistry()
	mgr := NewManager(root, root, cfgPath, reg)
	mgr.maxRestarts = 2
	defer mgr.Shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	sum, err := mgr.Reload(ctx)
	if err != nil {
		t.Fatalf("Reload: %v", err)
	}
	if sum.Warning == "" {
		t.Fatalf("expected many-tools warning when server exposes >= %d tools", ManyToolsThreshold)
	}
	if !sum.Servers[0].WarnManyTools && !sum.Servers[1].WarnManyTools {
		t.Fatalf("expected WarnManyTools flag on big server: %+v", sum.Servers)
	}

	// Wait for flaky server to hit restart limit cleanly.
	deadline := time.Now().Add(8 * time.Second)
	var flakyState string
	for time.Now().Before(deadline) {
		s := mgr.Summary()
		for _, srv := range s.Servers {
			if srv.Name == "flaky" {
				flakyState = srv.State
			}
		}
		if strings.Contains(flakyState, "limit reached") {
			break
		}
		time.Sleep(80 * time.Millisecond)
	}
	if !strings.Contains(flakyState, "limit reached") {
		t.Fatalf("expected flaky server to hit restart limit, final state=%q", flakyState)
	}
}
