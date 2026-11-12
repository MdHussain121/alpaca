//go:build windows

// Package runner starts, monitors and stops a llama-server child process.
package runner

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/windows"
)

const (
	createNoWindow        = 0x08000000
	createNewProcessGroup = 0x00000200
	ctrlBreakEvent        = 1
)

// Process is a running llama-server.
type Process struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	job     windows.Handle
	Port    int
	LogPath string
	Started time.Time

	lines  chan string
	done   chan struct{}
	mu     sync.Mutex
	waited bool
	exit   error
}

// Start launches exe with args in workDir, wiring stdout and stderr into one
// stream and into a log file. The child is placed in a Job Object so it dies
// with the launcher.
func Start(exe, workDir string, args []string, logPath string) (*Process, error) {
	// Windows resolves a relative executable path against the child's working
	// directory, so both must be absolute before they are handed to CreateProcess.
	if abs, err := filepath.Abs(exe); err == nil {
		exe = abs
	}
	if workDir != "" {
		if abs, err := filepath.Abs(workDir); err == nil {
			workDir = abs
		}
	}
	if _, err := os.Stat(exe); err != nil {
		return nil, fmt.Errorf("llama-server not found at %s: %w", exe, err)
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(exe, args...)
	cmd.Dir = workDir
	cmd.Stdout = pw
	cmd.Stderr = pw
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow | createNewProcessGroup}

	// Keep stdin open so the child cannot inherit a closed handle.
	stdin, err := cmd.StdinPipe()
	if err != nil {
		pr.Close()
		pw.Close()
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		pr.Close()
		pw.Close()
		return nil, fmt.Errorf("starting llama-server: %w", err)
	}
	// The parent must not hold the write end, or reads never see EOF.
	pw.Close()

	p := &Process{
		cmd:     cmd,
		stdin:   stdin,
		Port:    portFromArgs(args),
		LogPath: logPath,
		Started: time.Now(),
		lines:   make(chan string, 512),
		done:    make(chan struct{}),
	}

	// Without the job object a killed launcher would orphan the server, so a
	// failure here aborts the launch instead of running unprotected.
	h, err := attachJob(uint32(cmd.Process.Pid))
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		pr.Close()
		return nil, fmt.Errorf("could not place llama-server in a job object: %w", err)
	}
	p.job = h

	var logFile *os.File
	if logPath != "" {
		if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err == nil {
			logFile, _ = os.Create(logPath)
		}
	}

	go p.pump(pr, logFile)
	go func() {
		err := cmd.Wait()
		p.mu.Lock()
		p.waited = true
		p.exit = err
		p.mu.Unlock()
		close(p.done)
	}()

	return p, nil
}

func (p *Process) pump(r io.ReadCloser, logFile *os.File) {
	defer r.Close()
	if logFile != nil {
		defer logFile.Close()
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	written := 0
	for sc.Scan() {
		line := sc.Text()
		if logFile != nil {
			fmt.Fprintln(logFile, line)
			// Flushing every line turns a log flood into a flood of USB writes.
			written++
			if written%200 == 0 {
				logFile.Sync()
			}
		}
		select {
		case p.lines <- line:
		default:
			// Never block the child on a slow UI; drop the oldest.
			select {
			case <-p.lines:
			default:
			}
			select {
			case p.lines <- line:
			default:
			}
		}
	}
	if logFile != nil {
		_ = logFile.Sync()
	}
	close(p.lines)
}

func portFromArgs(args []string) int {
	for i := 0; i < len(args); i++ {
		if (args[i] == "--port" || args[i] == "-p") && i+1 < len(args) {
			var port int
			if _, err := fmt.Sscanf(args[i+1], "%d", &port); err == nil && port > 0 {
				return port
			}
		}
		if v, ok := strings.CutPrefix(args[i], "--port="); ok {
			var port int
			if _, err := fmt.Sscanf(v, "%d", &port); err == nil && port > 0 {
				return port
			}
		}
	}
	return 8080
}

// Lines is the merged log stream. It closes when the process exits.
func (p *Process) Lines() <-chan string { return p.lines }

// Done is closed when the process exits.
func (p *Process) Done() <-chan struct{} { return p.done }

// Exited reports whether the process has finished.
func (p *Process) Exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

// ExitError returns the wait error, if the process has exited.
func (p *Process) ExitError() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.exit
}

// PID is the child process id.
func (p *Process) PID() int {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

// Uptime is how long the server has been running.
func (p *Process) Uptime() time.Duration {
	if p == nil || p.Started.IsZero() {
		return 0
	}
	return time.Since(p.Started)
}

// Stop asks the server to shut down gracefully, waits up to 1.5s for it
// to exit cleanly, and then kills the job object.
func (p *Process) Stop() error {
	return p.StopTimeout(1500 * time.Millisecond)
}

// StopTimeout behaves like Stop with a configurable graceful-exit wait;
// zero terminates the job immediately.
func (p *Process) StopTimeout(grace time.Duration) error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	stdin := p.stdin
	p.stdin = nil
	p.mu.Unlock()
	if stdin != nil {
		_ = stdin.Close()
	}
	if p.Exited() {
		p.releaseJob()
		return nil
	}

	if grace > 0 && windows.GenerateConsoleCtrlEvent(ctrlBreakEvent, uint32(p.PID())) == nil {
		select {
		case <-p.done:
			p.releaseJob()
			return nil
		case <-time.After(grace):
		}
	}

	if job := p.jobHandle(); job != 0 {
		_ = windows.TerminateJobObject(job, 1)
	} else if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}

	select {
	case <-p.done:
	case <-time.After(1500 * time.Millisecond):
	}
	p.releaseJob()
	return nil
}

func (p *Process) jobHandle() windows.Handle {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.job
}

func (p *Process) releaseJob() {
	p.mu.Lock()
	job := p.job
	p.job = 0
	p.mu.Unlock()
	if job != 0 {
		_ = windows.CloseHandle(job)
	}
}

// Release drops the job handle without killing the child. Used once the child
// has already exited on its own.
func (p *Process) Release() { p.releaseJob() }

// Health polls GET /health until it returns 200, the context is cancelled, or
// the process exits. Uses adaptive backoff (400ms -> 2s) so long cold loads do not poll aggressively.
func (p *Process) Health(ctx context.Context) error {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", p.Port)
	client := telemetryHTTPClient

	var deadlineC <-chan time.Time
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		deadline := time.NewTimer(10 * time.Minute)
		defer deadline.Stop()
		deadlineC = deadline.C
	}

	interval := 400 * time.Millisecond
	polls := 0
	timer := time.NewTimer(interval)
	defer timer.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-p.done:
			return errors.New("llama-server exited before becoming ready")
		case <-deadlineC:
			return errors.New("timed out waiting for llama-server health")
		case <-timer.C:
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			if err == nil {
				if resp, err := client.Do(req); err == nil {
					code := resp.StatusCode
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
					if code == http.StatusOK {
						return nil
					}
				}
			}
			polls++
			if polls >= 20 {
				interval = 2 * time.Second
			} else if polls >= 10 {
				interval = time.Second
			}
			timer.Reset(interval)
		}
	}
}

// PortFree reports whether a TCP port can be bound on localhost.
func PortFree(port int) bool {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	l, err := netListen(addr)
	if err != nil {
		return false
	}
	l.Close()
	return true
}

// NextFreePort returns port, or the next free port after it.
func NextFreePort(port int) int {
	for i := 0; i < 100; i++ {
		p := port + i
		if p > 65535 {
			p = 1024 + (p % 64000)
		}
		if PortFree(p) {
			return p
		}
	}
	return port
}

// MissingDLL hints at a missing CUDA runtime when the exe cannot start.
func MissingDLL(err error, dir string) string {
	if err == nil {
		return ""
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "cudart") || strings.Contains(msg, "cublas") ||
		strings.Contains(msg, "0xc0000135") || strings.Contains(msg, "dll") {
		return fmt.Sprintf("a CUDA DLL is missing from %s — copy cudart64_*.dll, cublas64_*.dll and cublasLt64_*.dll next to llama-server.exe", dir)
	}
	return ""
}

var errNoListen = errors.New("listen not available")

// netListen is a tiny indirection so PortFree can be tested.
var netListen = func(addr string) (io.Closer, error) {
	return listenTCP(addr)
}
