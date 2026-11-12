package bench

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/MdHussain121/alpaca/internal/estimate"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

//go:embed data/prompts.json
var embeddedPromptsJSON []byte

type BenchConfig struct {
	BinPath        string
	ModelPath      string
	Label          string
	Port           int
	Repeats        int
	PromptsPath    string
	Filter         string
	SaveText       bool
	ColdEvictDir   string
	ColdEvictGB    int
	MeasureIdle    bool
	WarmRepeat     bool
	NeedleCtx      int
	OutCSV         string
	EvidenceDir    string
	LogTimestamps  bool
	CachePrompt    bool
	Temp           float64
	Seed           int
	MaxTokens      int
	EnvVars        string
	RawExtra       string
	Ctx            int
	CacheK         string
	CacheV         string
	FlashAttn      string
	GPULayers      int
	NCPUMoE        int
	NoKVOffload    string
	AutoFit        string
	Threads        int
	ThreadsBatch   int
	Batch          int
	UBatch         int
	LoadMode       string
	Warmup         string
	NoRepack       string
	NoWebUI        string
	MoECacheMiB    int
	MoECacheCPU    float64
	CacheReuse     int
	CacheRAM       int
	SpecType       string
	BackendSample  string
	CPUStrict      string
	Prio           int
	Metrics        string
	Slots          string
	LogLevel       int
	HealthTimeoutS int
}

type RunReport struct {
	Label           string           `json:"label"`
	Repeat          int              `json:"repeat"`
	Model           string           `json:"model"`
	ModelBytes      int64            `json:"model_bytes"`
	Bin             string           `json:"bin"`
	Args            []string         `json:"args"`
	Env             []string         `json:"env,omitempty"`
	StartedAt       string           `json:"started_at"`
	Startup         StartupBreakdown `json:"startup"`
	EvictDurationMs float64          `json:"evict_duration_ms"`
	ReadyMs         float64          `json:"ready_ms"`
	RSSMiB          float64          `json:"rss_mib"`
	PrivateMiB      float64          `json:"private_mib"`
	GPUBefore       GPUSample        `json:"gpu_before"`
	GPUAfterReady   GPUSample        `json:"gpu_after_ready"`
	GPUAfterBench   GPUSample        `json:"gpu_after_bench"`
	IdleCPUPct      float64          `json:"idle_cpu_pct"`
	WebUIStatus     int              `json:"webui_status"`
	WebUIBodyBytes  int              `json:"webui_body_bytes"`
	Prompts         []PromptResult   `json:"prompts"`
	WarmRepeat1     *PromptResult    `json:"warm_repeat_1,omitempty"`
	WarmRepeat2     *PromptResult    `json:"warm_repeat_2,omitempty"`
	NeedleOK        *bool            `json:"needle_ok,omitempty"`
	NeedleOutput    string           `json:"needle_output,omitempty"`
	PrefillMedian   float64          `json:"prefill_tps_median"`
	DecodeMedian    float64          `json:"decode_tps_median"`
	TTFTMedianMs    float64          `json:"ttft_ms_median"`
	OutputBundleSHA string           `json:"output_bundle_sha256"`
	EstVRAMBytes    int64            `json:"est_vram_bytes"`
	EstRAMBytes     int64            `json:"est_ram_bytes"`
	EstFits         bool             `json:"est_fits"`
	Error           string           `json:"error,omitempty"`
}

func Run(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	var cfg BenchConfig
	fs.StringVar(&cfg.BinPath, "bin", "", "path to llama-server.exe (auto-detected by default)")
	fs.StringVar(&cfg.ModelPath, "model", "", "path to .gguf model file")
	fs.StringVar(&cfg.Label, "label", "run", "benchmark run label")
	fs.IntVar(&cfg.Port, "port", 18080, "HTTP server port")
	fs.IntVar(&cfg.Repeats, "repeats", 1, "number of repeats")
	fs.StringVar(&cfg.PromptsPath, "prompts", "", "path to fixed prompts JSON (falls back to embedded)")
	fs.StringVar(&cfg.Filter, "filter", "all", "prompt selector: all, none, short, medium, long, or comma-separated IDs")
	fs.BoolVar(&cfg.SaveText, "save-text", false, "store full generated outputs in evidence JSON")
	fs.StringVar(&cfg.ColdEvictDir, "cold-evict-dir", "", "if non-empty, evict standby cache before each repeat using scratch dir")
	fs.IntVar(&cfg.ColdEvictGB, "cold-evict-gb", 14, "GB size for standby cache eviction scratch file")
	fs.BoolVar(&cfg.MeasureIdle, "measure-idle", true, "measure server idle CPU% over 3s and GET / check")
	fs.BoolVar(&cfg.WarmRepeat, "warm-repeat", false, "run 2-turn KV cache prefix hit test")
	fs.IntVar(&cfg.NeedleCtx, "needle-ctx", 0, "if >0, run needle-in-haystack retrieval test with ~N tokens")
	fs.StringVar(&cfg.OutCSV, "out", "", "append per-run & summary rows to CSV")
	fs.StringVar(&cfg.EvidenceDir, "evidence-dir", "", "directory for JSON evidence and server logs")
	fs.BoolVar(&cfg.LogTimestamps, "timestamps", true, "pass --log-timestamps for timing breakdown")
	fs.BoolVar(&cfg.CachePrompt, "cache-prompt", false, "pass cache_prompt=true/false in prompt suite requests")
	fs.Float64Var(&cfg.Temp, "temp", 0, "sampling temperature (0 = greedy deterministic)")
	fs.IntVar(&cfg.Seed, "seed", 42, "sampling seed")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", 0, "override max_tokens if > 0")
	fs.StringVar(&cfg.EnvVars, "env", "", "comma-separated KEY=VAL env overrides")
	fs.StringVar(&cfg.RawExtra, "extra", "", "additional raw flags appended to llama-server")

	fs.IntVar(&cfg.Ctx, "ctx", -1, "context size (-1 = launcher default 8192)")
	fs.StringVar(&cfg.CacheK, "ctk", "", "K cache type (empty = launcher default q8_0)")
	fs.StringVar(&cfg.CacheV, "ctv", "", "V cache type (empty = launcher default f16)")
	fs.StringVar(&cfg.FlashAttn, "fa", "default", "flash attn: default|on|off")
	fs.IntVar(&cfg.GPULayers, "ngl", -2, "GPU layers (-2 = launcher default)")
	fs.IntVar(&cfg.NCPUMoE, "n-cpu-moe", -1, "MoE layers on CPU (-1 = launcher default)")
	fs.StringVar(&cfg.NoKVOffload, "no-kv-offload", "default", "no-kv-offload: default|on|off")
	fs.StringVar(&cfg.AutoFit, "fit", "default", "fit: default|on|off")
	fs.IntVar(&cfg.Threads, "threads", -1, "threads (-1 = hw.PhysCores)")
	fs.IntVar(&cfg.ThreadsBatch, "threads-batch", -1, "prompt batch threads (-1 = unset)")
	fs.IntVar(&cfg.Batch, "batch", -1, "batch size (-1 = launcher default)")
	fs.IntVar(&cfg.UBatch, "ubatch", -1, "ubatch size (-1 = launcher default)")
	fs.StringVar(&cfg.LoadMode, "load-mode", "default", "load mode: default|auto|mmap|none|dio|mlock")
	fs.StringVar(&cfg.Warmup, "warmup", "default", "warmup: default|on|off")
	fs.StringVar(&cfg.NoRepack, "no-repack", "default", "no-repack: default|on|off")
	fs.StringVar(&cfg.NoWebUI, "no-webui", "default", "no-webui: default|on|off")
	fs.IntVar(&cfg.MoECacheMiB, "moe-cache-mib", -1, "MoE cache size in MiB (-1 = unset)")
	fs.Float64Var(&cfg.MoECacheCPU, "moe-cache-cpu-share", -1, "MoE cache CPU share (-1 = unset)")
	fs.IntVar(&cfg.CacheReuse, "cache-reuse", -1, "--cache-reuse threshold (-1 = unset)")
	fs.IntVar(&cfg.CacheRAM, "cram", -1, "--cache-ram MiB (-1 = unset)")
	fs.StringVar(&cfg.SpecType, "spec-type", "", "speculative decoding type (e.g. ngram-mod)")
	fs.StringVar(&cfg.BackendSample, "bs", "default", "backend sampling: default|on|off")
	fs.StringVar(&cfg.CPUStrict, "cpu-strict", "default", "cpu-strict: default|on|off")
	fs.IntVar(&cfg.Prio, "prio", -99, "process/thread priority (-99 = unset)")
	fs.StringVar(&cfg.Metrics, "metrics", "default", "metrics flag: default|on|off")
	fs.StringVar(&cfg.Slots, "slots", "default", "slots endpoint: default|on|off")
	fs.IntVar(&cfg.LogLevel, "lv", -99, "log verbosity (-99 = launcher default)")
	fs.IntVar(&cfg.HealthTimeoutS, "health-timeout", 900, "max seconds waiting for /health")
	if err := fs.Parse(args); err != nil {
		return err
	}

	hwInfo := hw.Detect(nil)

	if cfg.BinPath == "" {
		cfg.BinPath = autoDetectServer(hwInfo)
	}
	if cfg.BinPath == "" {
		return fmt.Errorf("could not locate llama-server.exe; use --bin <path>")
	}
	if _, err := os.Stat(cfg.BinPath); err != nil {
		return fmt.Errorf("server binary %q: %w", cfg.BinPath, err)
	}

	if cfg.ModelPath == "" {
		cfg.ModelPath = autoDetectModel()
	}
	if cfg.ModelPath == "" {
		return fmt.Errorf("--model is required (e.g. --model models\\model-name.gguf)")
	}
	if abs, err := filepath.Abs(cfg.ModelPath); err == nil {
		cfg.ModelPath = abs
	}
	st, err := os.Stat(cfg.ModelPath)
	if err != nil {
		return fmt.Errorf("model %q: %w", cfg.ModelPath, err)
	}

	envList := applyEnv(cfg.EnvVars)
	meta, parseErr := gguf.Parse(cfg.ModelPath)
	if parseErr != nil {
		fmt.Fprintf(os.Stderr, "warning: gguf.Parse failed: %v\n", parseErr)
	}

	settings, extraArgs := buildSettings(cfg, hwInfo)
	var est estimate.Result
	if meta != nil {
		est = estimate.Compute(meta, settings, hwInfo)
	}

	prompts, err := loadAndFilterPrompts(cfg.PromptsPath, cfg.Filter, cfg.MaxTokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading prompts: %v\n", err)
		return fmt.Errorf("invalid arguments")
	}

	if cfg.EvidenceDir != "" {
		if err := os.MkdirAll(cfg.EvidenceDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "error creating evidence dir: %v\n", err)
			return fmt.Errorf("invalid arguments")
		}
	}

	fmt.Printf("=== alpaca-bench [%s] model=%s size=%.2fGB repeats=%d prompts=%d ===\n",
		cfg.Label, filepath.Base(cfg.ModelPath), float64(st.Size())/1e9, cfg.Repeats, len(prompts))
	fmt.Printf("HW: GPU=%s VRAM=%dMiB PhysCores=%d LogicCores=%d RAM=%.1fGB\n",
		hwInfo.GPUName, hwInfo.VRAMTotal>>20, hwInfo.PhysCores, hwInfo.LogicCores, float64(hwInfo.RAMTotal)/(1<<30))

	var reports []RunReport
	var readyVals, ttftVals, prefillVals, decodeVals, rssVals []float64
	hashes := make(map[string]int)

	for r := 1; r <= cfg.Repeats; r++ {
		rep := runOnce(cfg, settings, extraArgs, envList, meta, est, st.Size(), prompts, r)
		reports = append(reports, rep)
		if rep.Error != "" {
			fmt.Fprintf(os.Stderr, "repeat %d error: %s\n", r, rep.Error)
			continue
		}
		readyVals = append(readyVals, rep.ReadyMs)
		rssVals = append(rssVals, rep.RSSMiB)
		if len(prompts) > 0 {
			ttftVals = append(ttftVals, rep.TTFTMedianMs)
			prefillVals = append(prefillVals, rep.PrefillMedian)
			decodeVals = append(decodeVals, rep.DecodeMedian)
			hashes[rep.OutputBundleSHA]++
		}
		fmt.Printf("  repeat %d/%d: ready=%.1fms rss=%.0fMiB vram=%.0fMiB ttft_med=%.1fms prefill_med=%.2ftok/s decode_med=%.2ftok/s idle_cpu=%.2f%% webui=%d(%dB) sha=%s\n",
			r, cfg.Repeats, rep.ReadyMs, rep.RSSMiB, rep.GPUAfterReady.VRAMUsedMiB,
			rep.TTFTMedianMs, rep.PrefillMedian, rep.DecodeMedian, rep.IdleCPUPct,
			rep.WebUIStatus, rep.WebUIBodyBytes, shortHash(rep.OutputBundleSHA))
	}

	readySum := summarize(readyVals)
	prefSum := summarize(prefillVals)
	decSum := summarize(decodeVals)
	ttftSum := summarize(ttftVals)
	rssSum := summarize(rssVals)

	fmt.Printf("SUMMARY [%s] n=%d:\n", cfg.Label, len(readyVals))
	fmt.Printf("  ready_ms:   min=%.1f  median=%.1f  max=%.1f  sd=%.1f\n", readySum.Min, readySum.Median, readySum.Max, readySum.StdDev)
	if len(prompts) > 0 {
		fmt.Printf("  prefill:    min=%.2f median=%.2f max=%.2f tok/s (cv=%.2f%%)\n", prefSum.Min, prefSum.Median, prefSum.Max, pctCV(prefSum))
		fmt.Printf("  decode:     min=%.2f median=%.2f max=%.2f tok/s (cv=%.2f%%)\n", decSum.Min, decSum.Median, decSum.Max, pctCV(decSum))
		fmt.Printf("  ttft_ms:    min=%.1f  median=%.1f  max=%.1f\n", ttftSum.Min, ttftSum.Median, ttftSum.Max)
		fmt.Printf("  unique output bundle hashes across %d repeats: %d\n", len(readyVals), len(hashes))
	}
	fmt.Printf("  rss_mib:    min=%.1f  median=%.1f  max=%.1f\n", rssSum.Min, rssSum.Median, rssSum.Max)

	if cfg.OutCSV != "" {
		if err := appendCSV(cfg.OutCSV, reports); err != nil {
			fmt.Fprintf(os.Stderr, "error writing CSV: %v\n", err)
			return fmt.Errorf("benchmark run failed")
		}
	}
	if cfg.EvidenceDir != "" {
		sumObj := map[string]any{
			"label":       cfg.Label,
			"model":       filepath.Base(cfg.ModelPath),
			"repeats":     cfg.Repeats,
			"ready_ms":    readySum,
			"prefill_tps": prefSum,
			"decode_tps":  decSum,
			"ttft_ms":     ttftSum,
			"rss_mib":     rssSum,
			"runs":        reports,
		}
		if b, err := json.MarshalIndent(sumObj, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(cfg.EvidenceDir, sanitizeFile(cfg.Label)+"_summary.json"), b, 0o644)
		}
	}
	if len(readyVals) == 0 {
		return fmt.Errorf("benchmark run failed")
	}
	return nil
}

func autoDetectServer(info hw.Info) string {
	candidates := []string{}
	if info.HasCUDA {
		candidates = append(candidates,
			"alpaca/bin/cuda/llama-server.exe",
			"bin/cuda/llama-server.exe",
			"dist/bin/cuda/llama-server.exe",
		)
	}
	candidates = append(candidates,
		"alpaca/bin/cpu/llama-server.exe",
		"bin/cpu/llama-server.exe",
		"dist/bin/cpu/llama-server.exe",
		"alpaca/bin/cpu-compat/llama-server.exe",
		"bin/cpu-compat/llama-server.exe",
		"dist/bin/cpu-compat/llama-server.exe",
	)
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if info.HasCUDA {
			candidates = append(candidates,
				filepath.Join(dir, "alpaca", "bin", "cuda", "llama-server.exe"),
				filepath.Join(dir, "bin", "cuda", "llama-server.exe"),
			)
		}
		candidates = append(candidates,
			filepath.Join(dir, "alpaca", "bin", "cpu", "llama-server.exe"),
			filepath.Join(dir, "bin", "cpu", "llama-server.exe"),
			filepath.Join(dir, "alpaca", "bin", "cpu-compat", "llama-server.exe"),
			filepath.Join(dir, "bin", "cpu-compat", "llama-server.exe"),
		)
	}
	for _, c := range candidates {
		if abs, err := filepath.Abs(c); err == nil {
			if _, statErr := os.Stat(abs); statErr == nil {
				return abs
			}
		}
	}
	return ""
}

func autoDetectModel() string {
	for _, d := range []string{"alpaca/models", "models", "."} {
		entries, err := os.ReadDir(d)
		if err == nil {
			for _, e := range entries {
				if strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
					return filepath.Join(d, e.Name())
				}
			}
		}
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		for _, d := range []string{filepath.Join(dir, "alpaca", "models"), filepath.Join(dir, "models")} {
			entries, err := os.ReadDir(d)
			if err == nil {
				for _, e := range entries {
					if strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
						return filepath.Join(d, e.Name())
					}
				}
			}
		}
	}
	return ""
}

func runOnce(
	cfg BenchConfig,
	s flags.Settings,
	extra []string,
	envList []string,
	meta *gguf.Metadata,
	est estimate.Result,
	modelBytes int64,
	prompts []PromptItem,
	repeatIdx int,
) RunReport {
	rep := RunReport{
		Label:        cfg.Label,
		Repeat:       repeatIdx,
		Model:        filepath.Base(cfg.ModelPath),
		ModelBytes:   modelBytes,
		Bin:          cfg.BinPath,
		Env:          envList,
		StartedAt:    time.Now().Format(time.RFC3339),
		EstVRAMBytes: int64(est.VRAMUsed()),
		EstRAMBytes:  int64(est.RAMUsed()),
		EstFits:      est.Fits(),
	}

	if cfg.ColdEvictDir != "" {
		dur, err := evictStandbyCache(cfg.ColdEvictDir, cfg.ColdEvictGB)
		if err != nil {
			rep.Error = fmt.Sprintf("cold evict: %v", err)
			return rep
		}
		rep.EvictDurationMs = float64(dur.Microseconds()) / 1000.0
	}

	if g, ok := sampleGPU(); ok {
		rep.GPUBefore = g
	}

	// Build exact flags via launcher's flags.Settings.Args plus controlled benchmark flags.
	runSettings := s
	if len(extra) > 0 {
		joined := strings.TrimSpace(strings.Join(extra, " "))
		if runSettings.ExtraArgs != "" {
			runSettings.ExtraArgs += " " + joined
		} else {
			runSettings.ExtraArgs = joined
		}
	}
	args := runSettings.Args(cfg.ModelPath)
	args = postFilterBenchFlags(args, cfg)
	rep.Args = args

	var logPath string
	if cfg.EvidenceDir != "" {
		logPath = filepath.Join(cfg.EvidenceDir, fmt.Sprintf("%s_r%d_server.log", sanitizeFile(cfg.Label), repeatIdx))
	} else {
		logPath = filepath.Join(os.TempDir(), fmt.Sprintf("alpaca-bench-%d.log", time.Now().UnixNano()))
		defer os.Remove(logPath)
	}

	startWall := time.Now()
	proc, err := runner.Start(cfg.BinPath, filepath.Dir(cfg.BinPath), args, logPath)
	if err != nil {
		rep.Error = fmt.Sprintf("runner.Start: %v", err)
		return rep
	}
	defer proc.StopTimeout(3 * time.Second)

	// Observe startup milestones asynchronously from proc.Lines().
	doneLogs := make(chan StartupBreakdown, 1)
	go func() {
		var sb StartupBreakdown
		for line := range proc.Lines() {
			updateMilestones(&sb, time.Since(startWall), line)
		}
		doneLogs <- sb
	}()

	ctxHealth, cancelHealth := context.WithTimeout(context.Background(), time.Duration(cfg.HealthTimeoutS)*time.Second)
	err = proc.Health(ctxHealth)
	cancelHealth()
	rep.ReadyMs = float64(time.Since(startWall).Microseconds()) / 1000.0

	if err != nil {
		rep.Error = fmt.Sprintf("health: %v", err)
		_ = proc.StopTimeout(3 * time.Second)
		rep.Startup = <-doneLogs
		return rep
	}

	if ws, priv, err := hw.ProcessRAM(uint32(proc.PID())); err == nil {
		rep.RSSMiB = float64(ws) / (1 << 20)
		rep.PrivateMiB = float64(priv) / (1 << 20)
	}
	if g, ok := sampleGPU(); ok {
		rep.GPUAfterReady = g
	}

	client := newBenchClient(cfg.Port)
	if cfg.MeasureIdle {
		if idlePct, err := measureIdleCPU(proc.PID(), 3*time.Second); err == nil {
			rep.IdleCPUPct = idlePct
		}
		if stCode, sz, err := client.checkRoot(context.Background()); err == nil {
			rep.WebUIStatus = stCode
			rep.WebUIBodyBytes = sz
		}
	}

	var prefillList, decodeList, ttftList []float64
	bundleHash := sha256.New()
	for _, p := range prompts {
		pr := client.runPrompt(context.Background(), p, cfg.Temp, cfg.Seed, cfg.CachePrompt, cfg.SaveText)
		rep.Prompts = append(rep.Prompts, pr)
		if pr.Error != "" {
			rep.Error = fmt.Sprintf("prompt %s: %s", p.ID, pr.Error)
			break
		}
		bundleHash.Write([]byte(p.ID + ":" + pr.OutputSHA + "\n"))
		ttftList = append(ttftList, pr.TTFTMs)
		if pr.Timings.PromptPerSecond > 0 {
			prefillList = append(prefillList, pr.Timings.PromptPerSecond)
		}
		if pr.Timings.PredictedPerSecond > 0 {
			decodeList = append(decodeList, pr.Timings.PredictedPerSecond)
		}
	}
	rep.OutputBundleSHA = hex.EncodeToString(bundleHash.Sum(nil))
	rep.TTFTMedianMs = summarize(ttftList).Median
	rep.PrefillMedian = summarize(prefillList).Median
	rep.DecodeMedian = summarize(decodeList).Median

	if rep.Error == "" && cfg.WarmRepeat {
		prefix := buildNeedleFiller(950)
		p1 := PromptItem{ID: "warm1", MaxTokens: 32, Text: prefix + "\n\nSummarize the paragraph above in one short sentence."}
		p2 := PromptItem{ID: "warm2", MaxTokens: 32, Text: prefix + "\n\nNow extract three key technical nouns from that same text."}
		r1 := client.runPrompt(context.Background(), p1, 0, cfg.Seed, true, false)
		r2 := client.runPrompt(context.Background(), p2, 0, cfg.Seed, true, false)
		rep.WarmRepeat1 = &r1
		rep.WarmRepeat2 = &r2
	}

	if rep.Error == "" && cfg.NeedleCtx > 0 {
		needleText := buildNeedlePrompt(cfg.NeedleCtx, "739418")
		np := PromptItem{ID: "needle", MaxTokens: 64, Text: needleText}
		nr := client.runPrompt(context.Background(), np, 0, cfg.Seed, false, true)
		ok := strings.Contains(nr.OutputText, "739418")
		rep.NeedleOK = &ok
		rep.NeedleOutput = nr.OutputText
	}

	if ws, priv, err := hw.ProcessRAM(uint32(proc.PID())); err == nil {
		if rss := float64(ws) / (1 << 20); rss > rep.RSSMiB {
			rep.RSSMiB = rss
		}
		if pv := float64(priv) / (1 << 20); pv > rep.PrivateMiB {
			rep.PrivateMiB = pv
		}
	}
	if g, ok := sampleGPU(); ok {
		rep.GPUAfterBench = g
	}

	_ = proc.StopTimeout(3 * time.Second)
	sb := <-doneLogs
	sb.ReadyHTTPMs = rep.ReadyMs
	rep.Startup = sb
	return rep
}

func buildSettings(cfg BenchConfig, hwInfo hw.Info) (flags.Settings, []string) {
	threads := hwInfo.PhysCores
	if threads <= 0 {
		threads = 8
	}
	s := flags.Default(threads, cfg.Port)
	s.Jinja = true
	if cfg.Ctx > 0 {
		s.Context = cfg.Ctx
	}
	if cfg.CacheK != "" {
		s.KCache = cfg.CacheK
	}
	if cfg.CacheV != "" {
		s.VCache = cfg.CacheV
	}
	switch cfg.FlashAttn {
	case "on":
		s.FlashAttn = true
	case "off":
		s.FlashAttn = false
	}
	if cfg.GPULayers != -2 {
		s.GPULayers = cfg.GPULayers
	}
	if cfg.NCPUMoE >= 0 {
		s.NCpuMoe = cfg.NCPUMoE
	}
	switch cfg.NoKVOffload {
	case "on":
		s.NoKVOffload = true
	case "off":
		s.NoKVOffload = false
	}
	switch cfg.AutoFit {
	case "on":
		s.AutoFit = true
	case "off":
		s.AutoFit = false
	}
	if cfg.Threads > 0 {
		s.Threads = cfg.Threads
	}
	if cfg.Batch > 0 {
		s.Batch = cfg.Batch
	}
	if cfg.UBatch > 0 {
		s.UBatch = cfg.UBatch
	}
	if cfg.LoadMode != "default" {
		s.LoadMode = cfg.LoadMode
	}
	if cfg.NoWebUI == "on" {
		s.NoWebUI = true
	} else if cfg.NoWebUI == "off" {
		s.NoWebUI = false
	}
	if cfg.RawExtra != "" {
		s.ExtraArgs = cfg.RawExtra
	}

	var extra []string
	if cfg.Warmup == "off" {
		extra = append(extra, "--no-warmup")
	}
	if cfg.NoRepack == "on" {
		extra = append(extra, "--no-repack")
	}
	if cfg.ThreadsBatch > 0 {
		extra = append(extra, "-tb", strconv.Itoa(cfg.ThreadsBatch))
	}
	if cfg.MoECacheMiB >= 0 {
		extra = append(extra, "--moe-cache-mib", strconv.Itoa(cfg.MoECacheMiB))
	}
	if cfg.MoECacheCPU >= 0 {
		extra = append(extra, "--moe-cache-cpu-share", strconv.FormatFloat(cfg.MoECacheCPU, 'f', 2, 64))
	}
	if cfg.CacheReuse >= 0 {
		extra = append(extra, "--cache-reuse", strconv.Itoa(cfg.CacheReuse))
	}
	if cfg.CacheRAM >= 0 {
		extra = append(extra, "-cram", strconv.Itoa(cfg.CacheRAM))
	}
	if cfg.SpecType != "" {
		extra = append(extra, "--spec-type", cfg.SpecType)
	}
	if cfg.BackendSample == "on" {
		extra = append(extra, "-bs")
	}
	if cfg.CPUStrict == "on" {
		extra = append(extra, "--cpu-strict", "1")
	}
	if cfg.Prio != -99 {
		extra = append(extra, "--prio", strconv.Itoa(cfg.Prio))
	}
	if cfg.Slots == "off" {
		extra = append(extra, "--no-slots")
	} else if cfg.Slots == "on" {
		extra = append(extra, "--slots")
	}
	if cfg.LogLevel != -99 {
		extra = append(extra, "-lv", strconv.Itoa(cfg.LogLevel))
	}
	if cfg.LogTimestamps {
		extra = append(extra, "--log-timestamps")
	}
	return s, extra
}

// postFilterBenchFlags allows explicit ablations (e.g. --metrics off or --ngl -99
// to omit -ngl for clean --fit testing) without breaking v1 flags.Args behavior.
func postFilterBenchFlags(args []string, cfg BenchConfig) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		if cfg.Metrics == "off" && args[i] == "--metrics" {
			continue
		}
		if cfg.GPULayers == -99 && args[i] == "-ngl" && i+1 < len(args) {
			i++
			continue
		}
		out = append(out, args[i])
	}
	return out
}

func loadAndFilterPrompts(path, filter string, maxTokensOverride int) ([]PromptItem, error) {
	if filter == "none" {
		return nil, nil
	}
	var b []byte
	var err error
	if path != "" {
		b, err = os.ReadFile(path)
	} else {
		for _, tryPath := range []string{
			"perf/data/prompts.json",
			"cmd/alpaca-bench/testdata/prompts.json",
		} {
			if data, readErr := os.ReadFile(tryPath); readErr == nil {
				b = data
				err = nil
				break
			}
		}
	}
	if len(b) == 0 && len(embeddedPromptsJSON) > 0 {
		b = embeddedPromptsJSON
		err = nil
	}
	if err != nil {
		return nil, err
	}
	if len(b) == 0 {
		return nil, fmt.Errorf("no prompt dataset available")
	}
	var all []PromptItem
	if err := json.Unmarshal(trimBOM(b), &all); err != nil {
		return nil, err
	}
	var want map[string]bool
	if filter != "" && filter != "all" && filter != "short" && filter != "medium" && filter != "long" {
		want = make(map[string]bool)
		for _, part := range strings.Split(filter, ",") {
			if id := strings.TrimSpace(part); id != "" {
				want[id] = true
			}
		}
	}
	var out []PromptItem
	for _, p := range all {
		if maxTokensOverride > 0 {
			p.MaxTokens = maxTokensOverride
		}
		switch {
		case filter == "" || filter == "all":
			out = append(out, p)
		case filter == "short" || filter == "medium" || filter == "long":
			if p.Tier == filter {
				out = append(out, p)
			}
		case want != nil:
			if want[p.ID] {
				out = append(out, p)
			}
		}
	}
	return out, nil
}

func applyEnv(spec string) []string {
	if strings.TrimSpace(spec) == "" {
		return nil
	}
	var applied []string
	for _, pair := range strings.Split(spec, ",") {
		k, v, ok := strings.Cut(strings.TrimSpace(pair), "=")
		if ok && k != "" {
			_ = os.Setenv(k, v)
			applied = append(applied, k+"="+v)
		}
	}
	sort.Strings(applied)
	return applied
}

func appendCSV(path string, runs []RunReport) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	_, statErr := os.Stat(path)
	needHeader := os.IsNotExist(statErr)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	if needHeader {
		_ = w.Write([]string{
			"label", "repeat", "model", "model_bytes", "ready_ms", "backend_init_ms",
			"fit_ms", "load_start_ms", "tensors_done_ms", "kv_alloc_ms", "warmup_ms", "listening_ms",
			"ttft_med_ms", "prefill_med_tps", "decode_med_tps",
			"rss_mib", "priv_mib", "vram_ready_mib", "vram_peak_mib",
			"gpu_clk_mhz", "gpu_power_w", "gpu_temp_c", "idle_cpu_pct",
			"webui_status", "webui_bytes", "bundle_sha", "est_vram_mib", "est_fits", "error",
		})
	}
	for _, r := range runs {
		_ = w.Write([]string{
			r.Label,
			strconv.Itoa(r.Repeat),
			r.Model,
			strconv.FormatInt(r.ModelBytes, 10),
			formatFloatCSV(r.ReadyMs),
			formatFloatCSV(r.Startup.BackendInitMs),
			formatFloatCSV(nonNegDiff(r.Startup.FitDoneMs, r.Startup.FitStartMs)),
			formatFloatCSV(r.Startup.ModelLoadStartMs),
			formatFloatCSV(r.Startup.LoadTensorsDoneMs),
			formatFloatCSV(r.Startup.KVAllocDoneMs),
			formatFloatCSV(nonNegDiff(r.Startup.WarmupDoneMs, r.Startup.WarmupStartMs)),
			formatFloatCSV(r.Startup.ListeningMs),
			formatFloatCSV(r.TTFTMedianMs),
			formatFloatCSV(r.PrefillMedian),
			formatFloatCSV(r.DecodeMedian),
			formatFloatCSV(r.RSSMiB),
			formatFloatCSV(r.PrivateMiB),
			formatFloatCSV(r.GPUAfterReady.VRAMUsedMiB),
			formatFloatCSV(r.GPUAfterBench.VRAMUsedMiB),
			formatFloatCSV(r.GPUAfterBench.ClockMHz),
			formatFloatCSV(r.GPUAfterBench.PowerW),
			formatFloatCSV(r.GPUAfterBench.TempC),
			formatFloatCSV(r.IdleCPUPct),
			strconv.Itoa(r.WebUIStatus),
			strconv.Itoa(r.WebUIBodyBytes),
			shortHash(r.OutputBundleSHA),
			strconv.FormatInt(r.EstVRAMBytes>>20, 10),
			strconv.FormatBool(r.EstFits),
			r.Error,
		})
	}
	return nil
}

func nonNegDiff(a, b float64) float64 {
	if a <= 0 || b <= 0 || a < b {
		return 0
	}
	return a - b
}

func pctCV(s MetricSummary) float64 {
	if s.Mean <= 0 {
		return 0
	}
	return 100.0 * s.StdDev / s.Mean
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

func sanitizeFile(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func buildNeedleFiller(targetTokens int) string {
	sentences := []string{
		"Modular storage adapters measure latency across contiguous block transfers under buffered page access.",
		"Deterministic vector kernels align inner matrix multiplication tiles to cache-line multiples.",
		"Scheduler barriers synchronize worker threads at predictable sequence step boundaries.",
		"Telemetry probes inspect queue depth, thermal margins, and heap retention windows periodically.",
		"Checksum manifests validate executable integrity across removable exFAT flash partitions.",
	}
	var b strings.Builder
	for i := 0; i < targetTokens/14; i++ {
		b.WriteString(sentences[i%len(sentences)])
		b.WriteByte(' ')
		if i%6 == 5 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func buildNeedlePrompt(targetTokens int, secret string) string {
	half := targetTokens / 2
	var b strings.Builder
	b.WriteString("Read the following archival engineering log carefully.\n\n")
	b.WriteString(buildNeedleFiller(half))
	b.WriteString(fmt.Sprintf("\n\n[CRITICAL VERIFICATION CODE: %s]\n\n", secret))
	b.WriteString(buildNeedleFiller(half))
	b.WriteString("\n\nQuestion: What is the exact 6-digit CRITICAL VERIFICATION CODE stated in the log above? Reply with only the 6-digit number.")
	return b.String()
}

