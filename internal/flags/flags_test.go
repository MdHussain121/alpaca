package flags

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestArgsFullOffloadFP16(t *testing.T) {
	s := Default(8, 8080)
	s.Context = 16384
	s.KCache = "q8_0"
	s.VCache = "f16"
	s.NCpuMoe = 0
	s.Jinja = false
	s.ExtraArgs = ""

	got := s.Args("/m/model.gguf")
	want := []string{
		"-m", "/m/model.gguf",
		"-c", "16384",
		"-ctk", "q8_0",
		"-ctv", "f16",
		"-fa", "on",
		"--fit", "on",
		"-t", "8",
		"-b", "2048",
		"-ub", "512",
		"-np", "1",
		"--temp", "0.7",
		"--top-p", "0.95",
		"--top-k", "40",
		"--min-p", "0",
		"--host", "127.0.0.1",
		"--port", "8080",
		"--no-webui",
		"--no-slots",
		"-lv", "4",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv mismatch\n got: %s\nwant: %s", strings.Join(got, " "), strings.Join(want, " "))
	}

	// Threads==0 leaves -t/-tb/--prio default/omitted and omits --metrics unless opted in; optional flags emit when set.
	auto := Default(0, 8080)
	autoArgs := strings.Join(auto.Args("/m/model.gguf"), " ")
	for _, forbid := range []string{" -t ", " -tb ", "--prio", "--no-warmup", "--moe-cache-mib", "--spec-type", "--metrics"} {
		if strings.Contains(" "+autoArgs+" ", forbid) {
			t.Fatalf("default auto GPU args should omit %q: %s", forbid, autoArgs)
		}
	}
	auto.Warmup = false
	auto.Prio = "high"
	auto.Threads = 6
	auto.ThreadsBatch = 12
	auto.MoeCacheMiB = 3072
	auto.SpecType = "ngram-mod"
	optArgs := strings.Join(auto.Args("/m/model.gguf"), " ")
	for _, req := range []string{"--no-warmup", "--prio 2", "-t 6", "-tb 12", "--moe-cache-mib 3072", "--spec-type ngram-mod"} {
		if !strings.Contains(optArgs, req) {
			t.Fatalf("missing %q in %s", req, optArgs)
		}
	}
}

func TestAutoFitOffEmitsNglAllAndWarmupToggle(t *testing.T) {
	s := Default(8, 8080)
	s.AutoFit = false
	joinedOn := strings.Join(s.Args("m.gguf"), " ")
	if !strings.Contains(joinedOn, "-ngl all --fit off") {
		t.Fatalf("fit off must emit -ngl all: %s", joinedOn)
	}
	if strings.Contains(joinedOn, "--no-warmup") {
		t.Fatalf("warmup=true default must omit --no-warmup: %s", joinedOn)
	}
	s.Warmup = false
	if !strings.Contains(strings.Join(s.Args("m.gguf"), " "), "--no-warmup") {
		t.Fatalf("warmup=false must emit --no-warmup")
	}
}

func TestQuantizedVForcesFlashAttn(t *testing.T) {
	s := Default(8, 8080)
	s.VCache = "q4_0"
	s.FlashAttn = false
	s.Normalize()
	if !s.FlashAttn {
		t.Fatal("flash attention must be forced on for a quantized V cache")
	}
	if !s.FlashAttnLocked() {
		t.Fatal("expected the lock")
	}

	s.VCache = "f16"
	if s.FlashAttnLocked() {
		t.Fatal("f16 V cache must not lock flash attention")
	}
}

func TestMoEAndLoadModeArgs(t *testing.T) {
	s := Default(10, 8999)
	s.NCpuMoe = 24
	s.LoadMode = "none"
	s.LAN = true
	s.Jinja = true
	s.GPULayers = 31

	args := s.Args("/m/moe.gguf")
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--n-cpu-moe 24",
		"-lm none",
		"--host 0.0.0.0",
		"--jinja",
		"-ngl 31",
		"--port 8999",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in: %s", want, joined)
		}
	}
}

func TestAutoLoadModeIsOmitted(t *testing.T) {
	s := Default(8, 8080)
	s.LoadMode = "auto"
	joined := strings.Join(s.Args("m.gguf"), " ")
	if strings.Contains(joined, "-lm") {
		t.Fatalf("auto load mode should not emit -lm: %s", joined)
	}
}

func TestExtraArgsSplit(t *testing.T) {
	s := Default(8, 8080)
	s.ExtraArgs = `--cache-reuse 256 --chat-template-file "my template.jinja"`
	joined := strings.Join(s.Args("m.gguf"), " ")
	if !strings.Contains(joined, "--cache-reuse 256") {
		t.Errorf("missing extra flag: %s", joined)
	}
	if !strings.Contains(joined, "my template.jinja") {
		t.Errorf("quoted extra arg not preserved: %s", joined)
	}
}

func TestNormalizeKeepsValuesSane(t *testing.T) {
	s := Settings{Context: 0, Threads: -3, ThreadsBatch: -1, MoeCacheMiB: -99, KCache: "f8_e4m3", VCache: "f8_e4m3", Prio: "2", SpecType: "bogus", Batch: 0, UBatch: 99999, Parallel: 0, Port: 99999, Temp: 99, TopP: 3}
	s.Normalize()
	if s.Context < 1 || s.Threads != 0 || s.ThreadsBatch != 0 || s.Batch < 1 || s.Parallel < 1 {
		t.Fatalf("normalize invalid zero/negative bounds: %+v", s)
	}
	if s.KCache != "q8_0" || s.VCache != "f16" || s.Prio != "high" || s.SpecType != "none" || s.MoeCacheMiB != 0 {
		t.Fatalf("normalize failed rejects/aliases: %+v", s)
	}
	if s.UBatch > s.Batch {
		t.Fatalf("ubatch %d > batch %d", s.UBatch, s.Batch)
	}
	if s.Port != 8080 {
		t.Fatalf("port = %d", s.Port)
	}
	if s.Temp > 5 || s.TopP > 1 {
		t.Fatalf("sampling not clamped: %+v", s)
	}
}

// Every K/V pair must produce a legal command: a quantized V cache forces
// flash attention on, and no combination may emit an unknown flag.
func TestCacheCombinationsAreAllLegal(t *testing.T) {
	for _, k := range CacheOptions() {
		for _, v := range CacheOptions() {
			s := Default(4, 8080)
			s.KCache, s.VCache = k, v
			s.FlashAttn = false
			args := strings.Join(s.Args("m.gguf"), " ")
			if v != "f16" {
				if !strings.Contains(args, "-fa on") {
					t.Fatalf("K=%s V=%s emitted without -fa on: %s", k, v, args)
				}
			}
			if !strings.Contains(args, "-ctk "+k) || !strings.Contains(args, "-ctv "+v) {
				t.Fatalf("cache types missing: %s", args)
			}
		}
	}
}

func TestContextStopsAreAscending(t *testing.T) {
	stops := ContextStops()
	if len(stops) < 2 {
		t.Fatal("no context stops")
	}
	for i := 1; i < len(stops); i++ {
		if stops[i] <= stops[i-1] {
			t.Fatalf("stops not ascending: %v", stops)
		}
		if got := NextStop(stops[i-1], 1); got != stops[i] {
			t.Fatalf("NextStop(%d) = %d, want %d", stops[i-1], got, stops[i])
		}
	}
}

func TestSplitArgsHonoursBothQuoteStyles(t *testing.T) {
	got := SplitArgs(`--a "b c" --d 'e f'  --g`)
	want := []string{"--a", "b c", "--d", "e f", "--g"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v, want %#v", got, want)
	}
}

func TestKeysCoverEverySetting(t *testing.T) {
	if got := len(Default(4, 8080).Keys()); got != 36 {
		t.Fatalf("keys = %d", got)
	}
}

// Web UI is disabled by default (--no-webui) and can be toggled on;
// a vision projector is passed through either way.
func TestWebUIToggleAndProjectorFlags(t *testing.T) {
	s := Default(4, 8080)
	if !s.NoWebUI || !strings.Contains(strings.Join(s.Args("m.gguf"), " "), "--no-webui") {
		t.Fatal("--no-webui must be enabled by default")
	}

	s.NoWebUI = false
	if strings.Contains(strings.Join(s.Args("m.gguf"), " "), "--no-webui") {
		t.Fatal("web UI enable toggle did not drop --no-webui")
	}
	s.NoWebUI = true

	proj := filepath.Join(t.TempDir(), "mmproj-f16.gguf")
	s.MMProj = proj
	joined := strings.Join(s.Args("m.gguf"), " ")
	if !strings.Contains(joined, "--mmproj "+proj) {
		t.Fatalf("projector missing: %s", joined)
	}
}

func TestNextStop(t *testing.T) {
	if got := NextStop(8192, 1); got != 16384 {
		t.Errorf("up from 8192 = %d", got)
	}
	if got := NextStop(8192, -1); got != 4096 {
		t.Errorf("down from 8192 = %d", got)
	}
	if got := NextStop(262144, 1); got != 262144 {
		t.Errorf("clamped top = %d", got)
	}
	if got := NextStop(2048, -1); got != 2048 {
		t.Errorf("clamped bottom = %d", got)
	}
	if got := NextStop(10000, -1); got != 8192 {
		t.Errorf("non-stop value down = %d", got)
	}
}

func TestQuoteRoundTrip(t *testing.T) {
	q := Quote([]string{"-m", "C:\\a b\\m.gguf", "-c", "8192"})
	if !strings.Contains(q, `"C:\a b\m.gguf"`) {
		t.Fatalf("path not quoted: %s", q)
	}
	if !strings.HasPrefix(q, "llama-server ") {
		t.Fatalf("missing program: %s", q)
	}
}

// The preview must be copy-pasteable: Windows quoting, not shell escaping.
func TestQuoteUsesWindowsQuoting(t *testing.T) {
	got := Quote([]string{"--chat-template", `a "quoted" name.jinja`})
	want := `llama-server --chat-template "a \"quoted\" name.jinja"`
	if got != want {
		t.Fatalf("quote = %s, want %s", got, want)
	}
}

func TestQuoteKeepsSimpleArgsBare(t *testing.T) {
	got := Quote([]string{"-m", "m.gguf", "-c", "8192"})
	if got != "llama-server -m m.gguf -c 8192" {
		t.Fatalf("quote = %s", got)
	}
}

func TestElemBytesCoversEveryCacheOption(t *testing.T) {
	for _, c := range CacheOptions() {
		if ElemBytes(c) <= 0 {
			t.Fatalf("%s has no element size", c)
		}
	}
	if got := Quote([]string{""}); got != `llama-server ""` {
		t.Fatalf("empty arg = %s", got)
	}
}

func TestOptionListsAndEmptyExtraArgs(t *testing.T) {
	if len(CacheOptions()) != 3 {
		t.Fatalf("cache options = %v", CacheOptions())
	}
	for _, c := range CacheOptions() {
		if c == "f8_e4m3" {
			t.Fatalf("rejected f8_e4m3 still present in CacheOptions: %v", CacheOptions())
		}
	}
	if len(LoadModeOptions()) != 5 {
		t.Fatalf("load modes = %v", LoadModeOptions())
	}
	if got := SplitArgs("   "); len(got) != 0 {
		t.Fatalf("blank extra args = %#v", got)
	}
	s := Default(4, 8080)
	s.ExtraArgs = "  "
	if strings.Contains(strings.Join(s.Args("m.gguf"), " "), "  ") {
		t.Fatal("blank extra args leaked into the argv")
	}
}

func TestElemBytes(t *testing.T) {
	if ElemBytes("f16") != 2.0 {
		t.Error("f16")
	}
	if ElemBytes("f8_e4m3") != 2.0 {
		t.Error("rejected f8_e4m3 should fall back to default 2.0")
	}
	if ElemBytes("q8_0") != 1.0625 {
		t.Error("q8_0")
	}
	if ElemBytes("q4_0") != 0.5625 {
		t.Error("q4_0")
	}
	if ElemBytes("nonsense") != 2.0 {
		t.Error("fallback")
	}
}

func TestReasoningAndOffloadFlags(t *testing.T) {
	s := Default(8, 8080)
	s.NoKVOffload = true
	s.Reasoning = "on"
	s.ReasoningEffort = "high"
	s.ReasoningBudget = "1024"
	joined := strings.Join(s.Args("m.gguf"), " ")
	for _, want := range []string{
		"-nkvo",
		"--reasoning on",
		"--reasoning-effort high",
		"--reasoning-budget 1024",
		"--no-webui",
		"--no-slots",
		"-lv 4",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in: %s", want, joined)
		}
	}

	// ExtraArgs overriding -lv suppresses duplicate -lv 4.
	s.ExtraArgs = "-lv 2 --metrics"
	args := s.Args("m.gguf")
	lvCount := 0
	for _, a := range args {
		if a == "-lv" {
			lvCount++
		}
	}
	if lvCount != 1 {
		t.Fatalf("-lv emitted %d times: %v", lvCount, args)
	}

	// Zero-value struct from legacy TOML must safely normalize cache & mode fields.
	var legacy Settings
	legacy.Normalize()
	if legacy.KCache != "q8_0" || legacy.VCache != "f16" || legacy.LoadMode != "auto" || legacy.ReasoningEffort != "default" || legacy.LaunchMode != "chat" {
		t.Fatalf("legacy Normalize failed: %+v", legacy)
	}
}
