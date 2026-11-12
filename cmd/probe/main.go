// Command probe prints GGUF metadata and a memory estimate for a file, which is
// the fastest way to validate the parser and the estimator against reality.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/MdHussain121/alpaca/internal/estimate"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: probe <model.gguf> [ctx]")
		os.Exit(2)
	}
	path := os.Args[1]

	meta, err := gguf.Parse(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse %s: %v\n", path, err)
		os.Exit(1)
	}

	info := hw.Detect(nil)

	fmt.Printf("file        %s (%.2f GB)\n", filepath.Base(path), float64(meta.FileSize)/(1<<30))
	fmt.Printf("name        %s\n", meta.DisplayName())
	fmt.Printf("arch        %s\n", meta.Arch)
	fmt.Printf("quant       %s\n", meta.Quant)
	fmt.Printf("params      %s\n", meta.ParamsString())
	fmt.Printf("layers      %d\n", meta.BlockCount)
	fmt.Printf("ctx trained %d\n", meta.ContextLength)
	fmt.Printf("embedding   %d\n", meta.EmbeddingLen)
	fmt.Printf("heads       %d (kv arrays: %d)\n", meta.HeadCount, len(meta.HeadCountKV))
	fmt.Printf("head dim    %d (v %d)\n", meta.HeadDim(), meta.ValueDim())
	if meta.IsMoE() {
		fmt.Printf("moe         %d experts, %d used\n", meta.ExpertCount, meta.ExpertUsed)
	}
	fmt.Printf("weights     %.2f GB (global %.2f GB)\n", float64(meta.TotalBytes)/(1<<30), float64(meta.GlobalBytes)/(1<<30))
	fmt.Printf("hw          %s · vram %.1f GB · ram %.1f GB · %d cores\n",
		info.GPUName, float64(info.VRAMTotal)/(1<<30), float64(info.RAMTotal)/(1<<30), info.PhysCores)

	cpuOnly := false
	ctx := 8192
	smoke := false
	for _, a := range os.Args[2:] {
		switch a {
		case "--smoke":
			smoke = true
		case "--cpu":
			cpuOnly = true
		default:
			_, _ = fmt.Sscanf(a, "%d", &ctx)
		}
	}
	if cpuOnly {
		info.HasCUDA = false
		info.HasVulkan = false
		info.GPUName = ""
		info.VRAMTotal = 0
		info.VRAMFree = 0
		info.Devices = nil
	}
	s := flags.Default(0, 8080)
	if cpuOnly {
		s.GPULayers = 0
		s.NoKVOffload = true
	}
	s.Context = ctx
	s.Normalize()

	resolved, est := estimate.ResolveSettings(s, meta, info)
	fmt.Printf("\ncmd         %s\n", flags.Quote(resolved.Args(path)))
	fmt.Printf("vram        %.2f / %.2f GB (%.0f%%)\n",
		float64(est.VRAMUsed())/(1<<30), float64(est.VRAMAvail)/(1<<30), est.VRAMRatio()*100)
	fmt.Printf("ram         %.2f / %.2f GB (%.0f%%)\n",
		float64(est.RAMUsed())/(1<<30), float64(est.RAMAvail)/(1<<30), est.RAMRatio()*100)
	if est.CommitAvail > 0 {
		fmt.Printf("commit      %.2f / %.2f GB (%.0f%%)\n",
			float64(est.CommitUsed)/(1<<30), float64(est.CommitAvail)/(1<<30), est.CommitRatio()*100)
	}
	fmt.Printf("breakdown   %s\n", est.Breakdown())
	if !est.Fits() {
		fmt.Printf("warning     does not fit (%s)\n", strings.Join(est.Notes, ", "))
	}
	if smoke {
		runSmoke(path, resolved)
	}
}

func runSmoke(modelPath string, s flags.Settings) {
	exe := strings.TrimSpace(os.Getenv("ALPACA_PROBE_EXE"))
	if exe == "" {
		var err error
		exe, err = filepath.Abs(`dist\bin\cuda\llama-server.exe`)
		if err != nil {
			fmt.Fprintf(os.Stderr, "abs exe: %v\n", err)
			os.Exit(1)
		}
		if _, err := os.Stat(exe); err != nil {
			exe, _ = filepath.Abs(`bin\cuda\llama-server.exe`)
		}
	}
	port := runner.NextFreePort(18080)
	s.Port = port
	s.Reasoning = "on"
	s.ReasoningEffort = "low"
	s.MaxTokens = 128

	logPath := filepath.Join(os.TempDir(), "alpaca-probe-smoke.log")
	proc, err := runner.Start(exe, filepath.Dir(exe), s.Args(modelPath), logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "start: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		_ = proc.Stop()
		proc.Release()
	}()

	var mu sync.Mutex
	var metrics runner.Metrics
	doneLog := make(chan struct{})
	go func() {
		defer close(doneLog)
		for line := range proc.Lines() {
			mu.Lock()
			runner.ParseLine(line, &metrics)
			mu.Unlock()
		}
	}()

	t0 := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	if err := proc.Health(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "health: %v (see %s)\n", err, logPath)
		os.Exit(1)
	}
	fmt.Printf("smoke ready elapsed=%s\n", time.Since(t0).Round(time.Millisecond))

	var reasoning, content strings.Builder
	var last runner.ChatDelta
	req := runner.ChatRequest{
		Messages: []runner.APIMessage{
			{Role: "system", Content: "Answer concisely."},
			{Role: "user", Content: "What is 17 * 19? Give just the answer."},
		},
		ReasoningEffort: "low",
		MaxTokens:       128,
		Temperature:     0.2,
	}
	if err := runner.StreamChat(ctx, port, req, func(d runner.ChatDelta) {
		reasoning.WriteString(d.Reasoning)
		content.WriteString(d.Content)
		if d.PromptTokens > 0 || d.CompletionTokens > 0 || d.GenTokPerSec > 0 || d.PromptTokSec > 0 {
			last = d
		}
	}); err != nil {
		fmt.Fprintf(os.Stderr, "stream chat: %v\n", err)
		os.Exit(1)
	}

	tel, telErr := runner.FetchTelemetry(ctx, port)
	ws, priv, ramErr := hw.ProcessRAM(uint32(proc.PID()))
	memTotal, memAvail, _ := hw.MemoryStatus()

	mu.Lock()
	snap := metrics
	mu.Unlock()
	fmt.Printf("\nsmoke log   kv=%.1f MiB gpuKV=%.1f MiB cpuKV=%.1f MiB cpuModel=%.1f MiB gen=%.1f t/s layers=%d/%d\n",
		snap.KVActualMiB, snap.GPUKVMiB, snap.CPUKVMiB, snap.CPUModelMiB, snap.GenTokPerSec,
		snap.LayersOnGPU, snap.TotalLayers)
	fmt.Printf("smoke sse   promptTok=%d compTok=%d gen=%.1f t/s prompt=%.0f t/s\n",
		last.PromptTokens, last.CompletionTokens, last.GenTokPerSec, last.PromptTokSec)
	fmt.Printf("smoke tel   err=%v n_ctx=%d kvRatio=%.4f gen=%.1f t/s prompt=%.0f t/s\n",
		telErr, tel.NCtx, tel.KVCacheRatio, tel.GenTokPerSec, tel.PromptTokSec)
	fmt.Printf("smoke ram   err=%v ws=%.1f MiB priv=%.1f MiB sys=%.2f/%.2f GB\n",
		ramErr, float64(ws)/(1<<20), float64(priv)/(1<<20),
		float64(memTotal-memAvail)/(1<<30), float64(memTotal)/(1<<30))
	fmt.Printf("smoke text  reasoning=%q content=%q\n", reasoning.String(), content.String())
}
