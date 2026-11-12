// Command alpaca is a portable terminal launcher and chat client for llama.cpp models.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/ui"
)

var (
	version  = "dev"
	commit   = "unknown"
	llamaTag = "b12284"
)

func main() {
	if len(os.Args) > 1 {
		arg := strings.TrimSpace(os.Args[1])
		if arg == "--version" || arg == "-version" || arg == "-V" || arg == "version" {
			fmt.Println(ui.CompactBanner(prefersASCIIBanner()))
			fmt.Printf("alpaca %s (commit %s, llama.cpp %s)\n", version, commit, llamaTag)
			return
		}
	}

	asciiConsole, vtNotice := initConsole()
	root, envNotice := resolveRoot()
	mig := config.MigrateLegacy(root)
	cfg := config.Load(root)
	if asciiConsole && !cfg.ASCII {
		cfg.ASCII = true
	}

	sigCtx, stopSig := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSig()

	model := ui.New(root, cfg)
	model.SetVersionInfo(version, commit, llamaTag)
	var notices []string
	if vtNotice != "" {
		notices = append(notices, vtNotice)
	}
	if envNotice != "" {
		notices = append(notices, envNotice)
	}
	notices = append(notices, mig.Notices...)
	if len(notices) > 0 {
		model.SetStartupNotices(notices)
	}

	prog := tea.NewProgram(
		model,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
		tea.WithContext(sigCtx),
	)

	final, err := prog.Run()
	if m, ok := final.(*ui.Model); ok && m != nil {
		m.Shutdown()
	} else {
		model.Shutdown()
	}
	if err != nil && !errors.Is(err, tea.ErrProgramKilled) {
		fmt.Fprintf(os.Stderr, "alpaca: %v\n", err)
		os.Exit(1)
	}
}

// resolveRoot locates the application folder: ALPACA_ROOT, legacy LLM_ROOT,
// the executable's own directory, dist/ for local development, or cwd.
func resolveRoot() (string, string) {
	if v := os.Getenv("ALPACA_ROOT"); v != "" {
		return v, ""
	}
	if v := os.Getenv("LLM_ROOT"); v != "" {
		return v, "using legacy LLM_ROOT; prefer ALPACA_ROOT"
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if isDir(filepath.Join(dir, "models")) || isDir(filepath.Join(dir, "bin")) || fileExists(filepath.Join(dir, "config.toml")) {
			return dir, ""
		}
	}
	cwd, err := os.Getwd()
	if err != nil {
		return ".", ""
	}
	if isDir(filepath.Join(cwd, "dist")) {
		return filepath.Join(cwd, "dist"), ""
	}
	return cwd, ""
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}
