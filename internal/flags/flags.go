// Package flags is the only place llama.cpp command-line flag names live.
package flags

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Cache element sizes in bytes, matching llama.cpp's KV cache formats.
var cacheElemBytes = map[string]float64{
	"f16":  2.0,
	"bf16": 2.0,
	"q8_0": 1.0625,
	"q4_0": 0.5625,
}

// AllCount is the sentinel -ngl value meaning "every layer".
const AllCount = 999

// Settings is every value the launcher configures for llama-server and TUI chat.
type Settings struct {
	Context         int     `toml:"ctx"`
	KCache          string  `toml:"k_cache"`
	VCache          string  `toml:"v_cache"`
	AutoKV          bool    `toml:"auto_kv"`
	FlashAttn       bool    `toml:"flash_attn"`
	GPULayers       int     `toml:"gpu_layers"`
	NCpuMoe         int     `toml:"n_cpu_moe"`
	MoeCacheMiB     int     `toml:"moe_cache_mib"`
	NoKVOffload     bool    `toml:"no_kv_offload"`
	AutoFit         bool    `toml:"auto_fit"`
	Threads         int     `toml:"threads"`
	ThreadsBatch    int     `toml:"threads_batch"`
	Prio            string  `toml:"prio"`
	Batch           int     `toml:"batch"`
	UBatch          int     `toml:"ubatch"`
	LoadMode        string  `toml:"load_mode"`
	Warmup          bool    `toml:"warmup"`
	SpecType        string  `toml:"spec_type"`
	Parallel        int     `toml:"parallel"`
	Temp            float64 `toml:"temp"`
	ToolTemp        float64 `toml:"tool_temp"`
	TopP            float64 `toml:"top_p"`
	TopK            int     `toml:"top_k"`
	MinP            float64 `toml:"min_p"`
	Reasoning       string  `toml:"reasoning"`
	ReasoningEffort string  `toml:"reasoning_effort"`
	ReasoningBudget string  `toml:"reasoning_budget"`
	Jinja           bool    `toml:"jinja"`
	NoWebUI         bool    `toml:"no_webui"`
	LaunchMode      string  `toml:"launch_mode"`
	SystemPrompt    string  `toml:"system_prompt"`
	MaxTokens       int     `toml:"max_tokens"`
	HideThinking    bool    `toml:"hide_thinking"`
	MMProj          string  `toml:"mmproj"`
	Port            int     `toml:"port"`
	LAN             bool    `toml:"lan"`
	ExtraArgs       string  `toml:"extra_args"`
}

// Default returns sensible settings for a machine with the given cores.
// Passing threads <= 0 selects mode-aware auto thread defaults during resolution.
func Default(threads, port int) Settings {
	if threads < 0 {
		threads = 0
	}
	if port <= 0 {
		port = 8080
	}
	s := Settings{
		Context:         8192,
		KCache:          "q8_0",
		VCache:          "f16",
		AutoKV:          true,
		FlashAttn:       true,
		GPULayers:       AllCount,
		MoeCacheMiB:     0,
		AutoFit:         true,
		Threads:         threads,
		Prio:            "auto",
		Batch:           2048,
		UBatch:          512,
		LoadMode:        "auto",
		Warmup:          true,
		SpecType:        "none",
		Parallel:        1,
		Temp:            0.7,
		ToolTemp:        0.25,
		TopP:            0.95,
		TopK:            40,
		MinP:            0.0,
		Reasoning:       "auto",
		ReasoningEffort: "default",
		ReasoningBudget: "auto",
		Jinja:           true,
		NoWebUI:         true,
		LaunchMode:      "chat",
		Port:            port,
	}
	s.Normalize()
	return s
}

// Normalize enforces the invariants llama.cpp requires, so the command preview
// never shows something that would fail to start.
func (s *Settings) Normalize() {
	if !isValidCache(s.KCache) {
		s.KCache = "q8_0"
	}
	if !isValidCache(s.VCache) {
		s.VCache = "f16"
	}
	// A quantized V cache is only supported with flash attention.
	if s.VCache != "f16" && s.VCache != "bf16" {
		s.FlashAttn = true
	}
	if s.GPULayers < 0 {
		s.GPULayers = 0
	}
	if s.GPULayers > AllCount {
		s.GPULayers = AllCount
	}
	if s.NCpuMoe < 0 {
		s.NCpuMoe = 0
	}
	if s.MoeCacheMiB < 0 {
		s.MoeCacheMiB = 0
	}
	if s.Threads < 0 {
		s.Threads = 0
	}
	if s.ThreadsBatch < 0 {
		s.ThreadsBatch = 0
	}
	s.Prio = normalizePrio(s.Prio)
	if !containsString(SpecTypeOptions(), s.SpecType) {
		s.SpecType = "none"
	}
	if s.Batch < 1 {
		s.Batch = 1
	}
	if s.UBatch < 1 {
		s.UBatch = 1
	}
	if s.UBatch > s.Batch {
		s.UBatch = s.Batch
	}
	if s.Parallel < 1 {
		s.Parallel = 1
	}
	if s.Context < 1 {
		s.Context = 2048
	}
	if !containsString(LoadModeOptions(), s.LoadMode) {
		s.LoadMode = "auto"
	}
	if !containsString(ReasoningOptions(), s.Reasoning) {
		s.Reasoning = "auto"
	}
	if !containsString(ReasoningEffortOptions(), s.ReasoningEffort) {
		s.ReasoningEffort = "default"
	}
	if strings.TrimSpace(s.ReasoningBudget) == "" {
		s.ReasoningBudget = "auto"
	}
	if !containsString(LaunchModeOptions(), s.LaunchMode) {
		s.LaunchMode = "chat"
	}
	if s.MaxTokens < 0 {
		s.MaxTokens = 0
	}
	if s.Port < 1 || s.Port > 65535 {
		s.Port = 8080
	}
	if s.TopK < 0 {
		s.TopK = 0
	}
	s.ClampTemp()
}

// ClampTemp keeps sampling values inside the ranges llama.cpp accepts.
func (s *Settings) ClampTemp() {
	if s.Temp < 0 {
		s.Temp = 0
	}
	if s.Temp > 5 {
		s.Temp = 5
	}
	if s.ToolTemp <= 0 {
		s.ToolTemp = 0.25
	}
	if s.ToolTemp > 1 {
		s.ToolTemp = 1
	}
	if s.TopP < 0 {
		s.TopP = 0
	}
	if s.TopP > 1 {
		s.TopP = 1
	}
	if s.MinP < 0 {
		s.MinP = 0
	}
	if s.MinP > 1 {
		s.MinP = 1
	}
}

// FlashAttnLocked reports whether the UI must force flash attention on.
func (s *Settings) FlashAttnLocked() bool { return s.VCache != "f16" && s.VCache != "bf16" }

// Host is the bind address for llama-server.
func (s *Settings) Host() string {
	if s.LAN {
		return "0.0.0.0"
	}
	return "127.0.0.1"
}

// Endpoint is the OpenAI-compatible base URL the UI shows.
func (s *Settings) Endpoint() string {
	return fmt.Sprintf("http://127.0.0.1:%d/v1", s.Port)
}

// Args builds the llama-server argument list. The model path is passed in so a
// preset only has to store a file name.
func (s *Settings) Args(modelPath string) []string {
	s.Normalize()
	extraToks := SplitArgs(strings.TrimSpace(s.ExtraArgs))

	args := []string{
		"-m", modelPath,
		"-c", strconv.Itoa(s.Context),
		"-ctk", s.KCache,
		"-ctv", s.VCache,
	}

	if s.FlashAttn {
		args = append(args, "-fa", "on")
	} else {
		args = append(args, "-fa", "off")
	}

	// Upstream common_fit_params refuses to run --fit when -ngl is explicitly
	// passed on the CLI ("n_gpu_layers already set by user to -2, abort").
	// Omit -ngl when AutoFit is enabled with AllCount so --fit can pick layers;
	// emit it when AutoFit is off or when the user chose an explicit layer cap.
	if !s.AutoFit || s.GPULayers < AllCount {
		if s.GPULayers >= AllCount {
			args = append(args, "-ngl", "all")
		} else {
			args = append(args, "-ngl", strconv.Itoa(s.GPULayers))
		}
	}

	if s.NCpuMoe > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(s.NCpuMoe))
	}
	if s.MoeCacheMiB > 0 && !hasFlag(extraToks, "--moe-cache-mib") {
		args = append(args, "--moe-cache-mib", strconv.Itoa(s.MoeCacheMiB))
	}
	if s.NoKVOffload {
		args = append(args, "-nkvo")
	}

	args = append(args, "--fit", onOff(s.AutoFit))
	if !s.Warmup && !hasFlag(extraToks, "--warmup", "--no-warmup") {
		args = append(args, "--no-warmup")
	}

	if s.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(s.Threads))
	}
	if s.ThreadsBatch > 0 && !hasFlag(extraToks, "-tb", "--threads-batch") {
		args = append(args, "-tb", strconv.Itoa(s.ThreadsBatch))
	}
	if p, ok := prioArg(s.Prio); ok && !hasFlag(extraToks, "--prio") {
		args = append(args, "--prio", p)
	}
	args = append(args,
		"-b", strconv.Itoa(s.Batch),
		"-ub", strconv.Itoa(s.UBatch),
		"-np", strconv.Itoa(s.Parallel),
	)

	if s.SpecType != "" && s.SpecType != "none" && !hasFlag(extraToks, "--spec-type") {
		args = append(args, "--spec-type", s.SpecType)
	}

	if s.LoadMode != "" && s.LoadMode != "auto" {
		args = append(args, "-lm", s.LoadMode)
	}

	args = append(args,
		"--temp", trimFloat(s.Temp),
		"--top-p", trimFloat(s.TopP),
		"--top-k", strconv.Itoa(s.TopK),
		"--min-p", trimFloat(s.MinP),
	)

	if s.Reasoning != "" && s.Reasoning != "auto" {
		args = append(args, "--reasoning", s.Reasoning)
	}
	if s.ReasoningEffort != "" && s.ReasoningEffort != "default" {
		args = append(args, "--reasoning-effort", s.ReasoningEffort)
	}
	if n, ok := ParseReasoningBudget(s.ReasoningBudget); ok {
		args = append(args, "--reasoning-budget", strconv.Itoa(n))
	}

	if s.Jinja {
		args = append(args, "--jinja")
	}

	args = append(args,
		"--host", s.Host(),
		"--port", strconv.Itoa(s.Port),
	)

	// Web UI is disabled by default (--no-webui) because alpaca runs native TUI chat
	// and avoids serving unused HTML/JS assets; users can still toggle it on.
	if s.NoWebUI {
		args = append(args, "--no-webui")
	}

	// Disable unused /slots monitoring endpoint unless overridden in ExtraArgs;
	// /metrics is off by default in llama-server and omitted unless requested.
	if !hasFlag(extraToks, "--slots", "--no-slots") {
		args = append(args, "--no-slots")
	}

	// A multimodal projector is a separate file; it is passed through so vision
	// models work with or without the web UI.
	if strings.TrimSpace(s.MMProj) != "" {
		args = append(args, "--mmproj", s.MMProj)
	}

	// Log per-device tensor/KV buffer sizes (-lv 4) unless overridden in ExtraArgs.
	if !hasFlag(extraToks, "-lv", "--log-verbosity") {
		args = append(args, "-lv", "4")
	}
	args = append(args, extraToks...)
	return args
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

func trimFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

// SplitArgs splits free-text arguments, honouring single and double quotes.
func SplitArgs(s string) []string {
	var out []string
	var cur strings.Builder
	inSingle, inDouble, has := false, false, false

	flush := func() {
		if has {
			out = append(out, cur.String())
			cur.Reset()
			has = false
		}
	}
	for _, r := range s {
		switch {
		case r == '\'' && !inDouble:
			inSingle = !inSingle
			has = true
		case r == '"' && !inSingle:
			inDouble = !inDouble
			has = true
		case (r == ' ' || r == '\t') && !inSingle && !inDouble:
			flush()
		default:
			cur.WriteRune(r)
			has = true
		}
	}
	flush()
	return out
}

// Quote renders an argv slice as a copy-pasteable Windows command line, using
// the same quoting rules CreateProcess applies to argv.
func Quote(args []string) string {
	parts := make([]string, 0, len(args))
	for _, a := range args {
		parts = append(parts, quoteArg(a))
	}
	return "llama-server " + strings.Join(parts, " ")
}

// quoteArg quotes one argument exactly the way Windows parses it back.
func quoteArg(a string) string {
	if a != "" && !strings.ContainsAny(a, " \t\"") {
		return a
	}
	var b strings.Builder
	b.WriteByte('"')
	slashes := 0
	for _, r := range a {
		switch r {
		case '\\':
			slashes++
			continue
		case '"':
			// 2n+1 backslashes then the quote.
			b.WriteString(strings.Repeat(`\`, slashes*2+1))
			b.WriteRune(r)
			slashes = 0
			continue
		}
		if slashes > 0 {
			b.WriteString(strings.Repeat(`\`, slashes))
			slashes = 0
		}
		b.WriteRune(r)
	}
	// Trailing backslashes must be doubled so the closing quote is not escaped.
	if slashes > 0 {
		b.WriteString(strings.Repeat(`\`, slashes*2))
	}
	b.WriteByte('"')
	return b.String()
}

// CacheOptions are the KV cache types offered in the UI.
func CacheOptions() []string { return []string{"f16", "q8_0", "q4_0"} }

// PrioOptions are the --prio levels offered in the UI.
func PrioOptions() []string { return []string{"auto", "low", "normal", "medium", "high", "realtime"} }

// SpecTypeOptions are the speculative decoding modes offered in the UI.
func SpecTypeOptions() []string { return []string{"none", "ngram-mod"} }

// LoadModeOptions are the -lm values offered in the UI.
func LoadModeOptions() []string { return []string{"auto", "mmap", "mlock", "none", "dio"} }

// ReasoningOptions are the --reasoning modes offered in the UI.
func ReasoningOptions() []string { return []string{"auto", "on", "off"} }

// ReasoningEffortOptions are the --reasoning-effort levels supported by llama-server.
func ReasoningEffortOptions() []string {
	return []string{"default", "minimal", "low", "medium", "high", "xhigh", "max"}
}

// ReasoningBudgetOptions are preset stops for --reasoning-budget.
func ReasoningBudgetOptions() []string {
	return []string{"auto", "0", "256", "512", "1024", "2048", "4096", "8192", "16384"}
}

// LaunchModeOptions are the terminal launch views selectable before starting.
func LaunchModeOptions() []string { return []string{"chat", "webui", "log"} }

// LaunchModeLabel renders a concise label for a launch mode.
func LaunchModeLabel(mode string) string {
	switch mode {
	case "chat":
		return "TUI chat"
	case "webui":
		return "Web UI"
	case "log":
		return "Logs only"
	default:
		return "TUI chat"
	}
}

// ParseReasoningBudget parses a non-auto reasoning budget into a numeric token cap.
func ParseReasoningBudget(v string) (int, bool) {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" || v == "auto" {
		return -1, false
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return -1, false
	}
	return n, true
}

func isValidCache(c string) bool {
	_, ok := cacheElemBytes[c]
	return ok
}

func normalizePrio(p string) string {
	switch strings.ToLower(strings.TrimSpace(p)) {
	case "", "auto":
		return "auto"
	case "low", "-1":
		return "low"
	case "normal", "0":
		return "normal"
	case "medium", "1":
		return "medium"
	case "high", "2":
		return "high"
	case "realtime", "3":
		return "realtime"
	default:
		return "auto"
	}
}

func prioArg(p string) (string, bool) {
	switch normalizePrio(p) {
	case "low":
		return "-1", true
	case "medium":
		return "1", true
	case "high":
		return "2", true
	case "realtime":
		return "3", true
	default:
		return "", false
	}
}

func containsString(list []string, v string) bool {
	for _, x := range list {
		if x == v {
			return true
		}
	}
	return false
}

func hasFlag(args []string, names ...string) bool {
	for _, a := range args {
		for _, n := range names {
			if a == n || strings.HasPrefix(a, n+"=") {
				return true
			}
		}
	}
	return false
}

// ElemBytes returns the per-element storage size for a cache type.
func ElemBytes(cacheType string) float64 {
	if v, ok := cacheElemBytes[cacheType]; ok {
		return v
	}
	return 2.0
}

// ContextStops are the context sizes offered by the slider.
func ContextStops() []int {
	return []int{2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144}
}

// NextStop returns the next context stop in direction dir (+1/-1), clamped.
func NextStop(cur int, dir int) int {
	stops := ContextStops()
	if dir > 0 {
		for _, s := range stops {
			if s > cur {
				return s
			}
		}
		return stops[len(stops)-1]
	}
	for i := len(stops) - 1; i >= 0; i-- {
		if stops[i] < cur {
			return stops[i]
		}
	}
	return stops[0]
}

// Keys are the serialised field names, used by presets and tests.
func (s Settings) Keys() []string {
	keys := []string{
		"ctx", "k_cache", "v_cache", "auto_kv", "flash_attn", "gpu_layers", "n_cpu_moe",
		"moe_cache_mib", "no_kv_offload", "auto_fit", "threads", "threads_batch", "prio",
		"batch", "ubatch", "load_mode", "warmup", "spec_type", "parallel", "temp", "top_p",
		"top_k", "min_p", "reasoning", "reasoning_effort", "reasoning_budget", "jinja",
		"no_webui", "launch_mode", "system_prompt", "max_tokens", "hide_thinking",
		"mmproj", "port", "lan", "extra_args",
	}
	sort.Strings(keys)
	return keys
}
