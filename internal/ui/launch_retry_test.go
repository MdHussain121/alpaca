package ui

import (
	"errors"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/runner"
)

// templateRejectLog mirrors llama-server's fatal output when a model chat
// template rejects the --reasoning-effort value with a Jinja exception: the
// actionable error sits above the generic final line.
var templateRejectLog = []string{
	"I llama_kv_cache: CUDA0 KV buffer size =    30.38 MiB",
	"E srv          init: init: chat template parsing error: Unable to generate parser for this template.",
	"Error: Jinja Exception: Unexpected reasoning effort max. Supported types are high (default) and low.",
	"E srv          init: operator (): cleaning up before exit...",
	"E srv  server_run_i: exiting due to model loading error",
}

func retryTestModel(t *testing.T) *Model {
	t.Helper()
	m := testUI(t)
	m.screen = screenRun
	m.status = "LOADING"
	m.meta = testMeta(8_000_000_000, 36)
	m.proc = &runner.Process{}
	m.logs = templateRejectLog
	return m
}

func TestExitDuringLoadingRetriesWithoutReasoningEffort(t *testing.T) {
	m := retryTestModel(t)
	m.settings.ReasoningEffort = "max"

	newM, _ := m.Update(exitMsg{err: errors.New("exit status 1")})
	got := newM.(*Model)
	if got.status != "LOADING" {
		t.Fatalf("status = %q, want LOADING (auto-retry)", got.status)
	}
	if got.settings.ReasoningEffort != "default" {
		t.Fatalf("reasoning effort = %q, want default after retry", got.settings.ReasoningEffort)
	}
	if !got.launchRetried {
		t.Fatal("retry flag not set")
	}
	if got.proc != nil {
		t.Fatal("stale process left behind")
	}
	if !strings.Contains(got.toast, "--reasoning-effort max") {
		t.Fatalf("toast = %q, want the retry explanation", got.toast)
	}
}

func TestExitRetryRunsOnlyOnce(t *testing.T) {
	m := retryTestModel(t)
	m.launchRetried = true // first retry already spent

	newM, _ := m.Update(exitMsg{err: errors.New("exit status 1")})
	got := newM.(*Model)
	if got.status != "ERROR" {
		t.Fatalf("status = %q, want ERROR after the retry is exhausted", got.status)
	}
	if !strings.Contains(got.statusDe, "reasoning effort") {
		t.Fatalf("statusDe = %q, want the template error surfaced", got.statusDe)
	}
	if !strings.Contains(got.statusDe, "ctrl+e") {
		t.Fatalf("statusDe = %q, want the fix hint", got.statusDe)
	}
}

func TestExitWithDefaultEffortDoesNotRetry(t *testing.T) {
	m := retryTestModel(t)
	m.settings.ReasoningEffort = "default"

	newM, _ := m.Update(exitMsg{err: errors.New("exit status 1")})
	got := newM.(*Model)
	if got.status != "ERROR" {
		t.Fatalf("status = %q, want ERROR (nothing to drop)", got.status)
	}
	if got.launchRetried {
		t.Fatal("retry must not fire when the effort flag is absent")
	}
}

func TestExitAfterReadyDoesNotRetry(t *testing.T) {
	m := retryTestModel(t)
	m.status = "READY" // crash mid-chat is not a startup failure
	m.settings.ReasoningEffort = "max"

	newM, _ := m.Update(exitMsg{err: errors.New("exit status 1")})
	got := newM.(*Model)
	if got.status != "ERROR" {
		t.Fatalf("status = %q, want ERROR", got.status)
	}
	if got.settings.ReasoningEffort != "max" {
		t.Fatalf("reasoning effort = %q, want max (untouched after READY crash)", got.settings.ReasoningEffort)
	}
	if got.launchRetried {
		t.Fatal("retry flag must stay unset after a READY crash")
	}
}

func TestStartupReasonPrefersTemplateError(t *testing.T) {
	if got := startupReason(templateRejectLog); !strings.Contains(got, "reasoning effort") {
		t.Fatalf("startupReason = %q, want the Jinja error line", got)
	}
	if got := startupReason([]string{"noise", "E srv: exiting due to model loading error", ""}); got != "E srv: exiting due to model loading error" {
		t.Fatalf("startupReason tail = %q", got)
	}
	if got := startupReason(nil); got != "" {
		t.Fatalf("startupReason(nil) = %q", got)
	}
}

func TestEffortRejectedByTemplate(t *testing.T) {
	if !effortRejectedByTemplate(templateRejectLog) {
		t.Fatal("want the Jinja rejection detected")
	}
	if effortRejectedByTemplate([]string{"I srv init: listening on http://127.0.0.1:8080"}) {
		t.Fatal("want no detection on a healthy log")
	}
	if !effortRejectedByTemplate([]string{"note: reasoning_effort rejected"}) {
		t.Fatal("want the underscore variant detected")
	}
}
