//go:build windows

package runner

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestPortFreeAndNextFreePort(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	busy := ln.Addr().(*net.TCPAddr).Port

	if PortFree(busy) {
		t.Fatalf("port %d reported free while bound", busy)
	}
	if next := NextFreePort(busy); next == busy {
		t.Fatalf("NextFreePort returned the busy port %d", busy)
	}
}

func TestPortFromArgs(t *testing.T) {
	if got := portFromArgs([]string{"-m", "x", "--port", "9123"}); got != 9123 {
		t.Fatalf("port = %d", got)
	}
	if got := portFromArgs([]string{"-m", "x"}); got != 8080 {
		t.Fatalf("default port = %d", got)
	}
}

func TestMissingDLLHint(t *testing.T) {
	if got := MissingDLL(nil, "bin/cuda"); got != "" {
		t.Fatalf("nil error produced %q", got)
	}
	hint := MissingDLL(errors.New("The code execution cannot proceed because cudart64_12.dll was not found"), "bin/cuda")
	if !strings.Contains(hint, "cudart") {
		t.Fatalf("hint = %q", hint)
	}
	if got := MissingDLL(errors.New("something else"), "bin/cuda"); got != "" {
		t.Fatalf("unrelated error produced %q", got)
	}
}

func TestStartRejectsMissingExe(t *testing.T) {
	_, err := Start(filepath.Join(t.TempDir(), "nope.exe"), t.TempDir(), nil, "")
	if err == nil {
		t.Fatal("starting a missing exe must fail")
	}
}

func TestHealthSucceedsAgainstReadyServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}

	p := &Process{Port: port, done: make(chan struct{})}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := p.Health(ctx); err != nil {
		t.Fatalf("health: %v", err)
	}
}

func TestHealthFailsWhenTheChildExits(t *testing.T) {
	p := &Process{Port: 1, done: make(chan struct{})}
	close(p.done)
	if err := p.Health(context.Background()); err == nil {
		t.Fatal("expected an error once the server is gone")
	}
}

func TestHealthHonoursContextCancellation(t *testing.T) {
	p := &Process{Port: 1, done: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Health(ctx); err == nil {
		t.Fatal("expected a cancellation error")
	}
}

// A real long-running child must actually be stopped, not left behind.
func TestStopTerminatesLongRunningChild(t *testing.T) {
	exe := filepath.Join(os.Getenv("SystemRoot"), "System32", "ping.exe")
	if _, err := os.Stat(exe); err != nil {
		t.Skip("no ping.exe to run")
	}

	p, err := Start(exe, filepath.Dir(exe), []string{"-n", "60", "127.0.0.1"}, "")
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := p.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if !p.Exited() {
		t.Fatal("child still running after Stop")
	}
}

// A real short-lived llama-server run exercises the job object, the log pump
// and the exit path without needing a model.
func TestStartRunsAndCapturesOutput(t *testing.T) {
	exe := filepath.Join("..", "..", "dist", "bin", "cpu", "llama-server.exe")
	if _, err := os.Stat(exe); err != nil {
		t.Skip("no packaged llama-server.exe to run")
	}
	logPath := filepath.Join(t.TempDir(), "logs", "run.log")

	p, err := Start(exe, filepath.Dir(exe), []string{"--help"}, logPath)
	if err != nil {
		t.Fatalf("start: %v", err)
	}

	deadline := time.After(60 * time.Second)
	sawOutput := false
loop:
	for {
		select {
		case line, ok := <-p.Lines():
			if !ok {
				break loop
			}
			if strings.Contains(line, "--help") || strings.Contains(line, "usage") {
				sawOutput = true
			}
		case <-deadline:
			t.Fatal("server did not finish in time")
		}
	}

	if !sawOutput {
		t.Error("no help output captured from the child")
	}
	// The output stream closes on EOF, a moment before Wait records the exit.
	select {
	case <-p.Done():
	case <-time.After(10 * time.Second):
		t.Fatal("process did not exit after the stream closed")
	}
	if p.PID() == 0 {
		t.Error("no pid")
	}
	_ = p.Stop() // idempotent on an exited process

	if st, err := os.Stat(logPath); err != nil || st.Size() == 0 {
		t.Errorf("log file missing or empty: %v", err)
	}
}
