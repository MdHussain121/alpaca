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

	"github.com/MdHussain121/alpaca/internal/bench"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/eval"
	"github.com/MdHussain121/alpaca/internal/ui"
)

var (
	version  = "dev"
	commit   = "unknown"
	llamaTag = "b12284"
)

func printHelp() {
	fmt.Println(ui.CompactBanner(prefersASCIIBanner()))
	fmt.Printf("alpaca %s (commit %s, llama.cpp %s)\n\n", version, commit, llamaTag)
	fmt.Println("Usage:")
	fmt.Println("  alpaca                    Launch the interactive terminal UI")
	fmt.Println("  alpaca bench [options]    Run throughput, latency, and memory benchmarks")
	fmt.Println("  alpaca eval [options]     Evaluate agent tool-use capabilities and update UI badges")
	fmt.Println("  alpaca --version          Print version and llama.cpp information")
	fmt.Println("  alpaca --help             Show this help message")
	fmt.Println()
	fmt.Println("Benchmark Examples:")
	fmt.Println("  alpaca bench --model models/model.gguf")
	fmt.Println("  alpaca bench --model models/model.gguf --repeats 3 --out results.csv")
	fmt.Println()
	fmt.Println("Evaluation Examples:")
	fmt.Println("  alpaca eval                                 # evaluate all models in models/ (3 attempts)")
	fmt.Println("  alpaca eval --models-dir models --attempts 1 # fast single-pass evaluation")
	fmt.Println("  alpaca eval --filter Qwen3.5                 # evaluate only matching models")
}

func main() {
	if len(os.Args) > 1 {
		arg := strings.ToLower(strings.TrimSpace(os.Args[1]))
		switch arg {
		case "--version", "-version", "-v", "version":
			fmt.Println(ui.CompactBanner(prefersASCIIBanner()))
			fmt.Printf("alpaca %s (commit %s, llama.cpp %s)\n", version, commit, llamaTag)
			return
		case "--help", "-help", "-h", "help", "/?":
			printHelp()
			return
		case "eval":
			if err := eval.Run(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "alpaca eval: %v\n", err)
				os.Exit(1)
			}
			return
		case "bench":
			if err := bench.Run(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "alpaca bench: %v\n", err)
				os.Exit(1)
			}
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
