//go:build livecheck

package estimate

import (
	"context"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

var (
	reLogKV = regexp.MustCompile(`CUDA0 KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB`)
	reLogRS = regexp.MustCompile(`CUDA0 RS buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB`)
)

func queryVRAMUsedMiB(t *testing.T) float64 {
	t.Helper()
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		t.Fatalf("nvidia-smi: %v", err)
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		t.Fatalf("parse nvidia-smi %q: %v", string(out), err)
	}
	return v
}

func TestEstimatorAccuracyLive(t *testing.T) {
	modelsDir := strings.TrimSpace(os.Getenv("ALPACA_TEST_MODELS_DIR"))
	if modelsDir == "" {
		t.Skip("set ALPACA_TEST_MODELS_DIR to run live estimator accuracy test")
	}
	root := filepath.Join("..", "..")
	exe, err := filepath.Abs(filepath.Join(root, "dist", "bin", "cuda", "llama-server.exe"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(exe); err != nil {
		t.Skipf("missing %s", exe)
	}

	info := hw.Detect(nil)
	if info.VRAMTotal == 0 {
		t.Skip("no CUDA GPU detected")
	}

	type modelCase struct {
		file    string
		ncpuMoe int
		autoFit bool
	}
	models := []modelCase{
		{file: filepath.Join(modelsDir, "Spark-X2.6-4B-abliterated-Q4_K_M.gguf"), ncpuMoe: 0, autoFit: false},
		{file: filepath.Join(modelsDir, "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf"), ncpuMoe: 0, autoFit: false},
		{file: filepath.Join(modelsDir, "LFM2.5-8B-A1B-Q4_0.gguf"), ncpuMoe: 0, autoFit: false},
		{file: filepath.Join(modelsDir, "Tiel-Coder-35B-A3B-UD-IQ4_XS.gguf"), ncpuMoe: 40, autoFit: false},
	}
	kvPairs := [][2]string{
		{"f16", "f16"},
		{"q8_0", "q8_0"},
		{"q8_0", "q4_0"},
		{"q4_0", "q4_0"},
	}

	evDir := filepath.Join(t.TempDir(), "evidence")
	if err := os.MkdirAll(evDir, 0o755); err != nil {
		t.Fatal(err)
	}
	csvPath := filepath.Join(evDir, "estimator_accuracy.csv")
	cf, err := os.Create(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	defer cf.Close()
	w := csv.NewWriter(cf)
	_ = w.Write([]string{
		"model", "kv_pair", "ctx",
		"est_kv_mib", "actual_kv_mib", "actual_rs_mib", "kv_err_pct",
		"est_vram_mib", "actual_vram_delta_mib", "vram_err_pct",
	})

	for _, mc := range models {
		if _, err := os.Stat(mc.file); err != nil {
			t.Fatalf("missing model %s: %v", mc.file, err)
		}
		meta, err := gguf.Parse(mc.file)
		if err != nil {
			t.Fatalf("parse %s: %v", mc.file, err)
		}
		baseName := filepath.Base(mc.file)

		for _, kv := range kvPairs {
			s := flags.Default(0, 8080)
			s.Context = 8192
			s.AutoKV = false
			s.KCache = kv[0]
			s.VCache = kv[1]
			s.FlashAttn = true
			s.AutoFit = mc.autoFit
			s.GPULayers = flags.AllCount
			s.NCpuMoe = mc.ncpuMoe
			s.Warmup = false
			s.LoadMode = "mmap"
			s.Port = runner.NextFreePort(18200)
			s.Normalize()

			res := Compute(meta, s, info)
			estKVMiB := float64(res.KVVRAM) / (1 << 20)
			estVRAMMiB := float64(res.VRAMUsed()) / (1 << 20)

			vramBefore := queryVRAMUsedMiB(t)
			logPath := filepath.Join(evDir, fmt.Sprintf("est_%s_%s_%s.log", strings.TrimSuffix(baseName, ".gguf"), kv[0], kv[1]))
			_ = os.Remove(logPath)
			proc, err := runner.Start(exe, filepath.Dir(exe), s.Args(mc.file), logPath)
			if err != nil {
				t.Fatalf("start %s (%s/%s): %v", baseName, kv[0], kv[1], err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			ready := false
			if err := proc.Health(ctx); err == nil {
				ready = true
			}
			cancel()
			vramAfter := queryVRAMUsedMiB(t)
			_ = proc.StopTimeout(2 * time.Second)
			proc.Release()
			time.Sleep(400 * time.Millisecond)

			if !ready {
				t.Fatalf("%s (%s/%s) failed to reach /health; see %s", baseName, kv[0], kv[1], logPath)
			}

			raw, err := os.ReadFile(logPath)
			if err != nil {
				t.Fatal(err)
			}
			txt := string(raw)
			actualKVMiB := 0.0
			for _, m := range reLogKV.FindAllStringSubmatch(txt, -1) {
				if len(m) == 2 {
					v, _ := strconv.ParseFloat(m[1], 64)
					actualKVMiB += v
				}
			}
			actualRSMiB := 0.0
			for _, m := range reLogRS.FindAllStringSubmatch(txt, -1) {
				if len(m) == 2 {
					v, _ := strconv.ParseFloat(m[1], 64)
					actualRSMiB += v
				}
			}
			actualVRAMDelta := vramAfter - vramBefore
			if actualVRAMDelta < 0 {
				actualVRAMDelta = vramAfter
			}

			kvErrPct := 0.0
			if actualKVMiB > 0 {
				kvErrPct = math.Abs(estKVMiB-actualKVMiB) / actualKVMiB * 100.0
			}
			vramErrPct := 0.0
			if actualVRAMDelta > 0 {
				vramErrPct = math.Abs(estVRAMMiB-actualVRAMDelta) / actualVRAMDelta * 100.0
			}

			t.Logf("%-38s %s/%s  KV est=%.1fMiB act=%.1fMiB (RS=%.1fMiB) err=%.1f%% | VRAM est=%.0fMiB act=%.0fMiB err=%.1f%%",
				baseName, kv[0], kv[1], estKVMiB, actualKVMiB, actualRSMiB, kvErrPct, estVRAMMiB, actualVRAMDelta, vramErrPct)

			_ = w.Write([]string{
				baseName,
				kv[0] + "/" + kv[1],
				"8192",
				fmt.Sprintf("%.2f", estKVMiB),
				fmt.Sprintf("%.2f", actualKVMiB),
				fmt.Sprintf("%.2f", actualRSMiB),
				fmt.Sprintf("%.2f", kvErrPct),
				fmt.Sprintf("%.1f", estVRAMMiB),
				fmt.Sprintf("%.1f", actualVRAMDelta),
				fmt.Sprintf("%.2f", vramErrPct),
			})

			if actualKVMiB > 0 && kvErrPct > 15.0 {
				t.Errorf("%s %s/%s KV error %.1f%% > 15%% (est=%.1f act=%.1f)", baseName, kv[0], kv[1], kvErrPct, estKVMiB, actualKVMiB)
			}
			if vramErrPct > 15.0 {
				t.Errorf("%s %s/%s VRAM error %.1f%% > 15%% (est=%.0f act=%.0f)", baseName, kv[0], kv[1], vramErrPct, estVRAMMiB, actualVRAMDelta)
			}
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		t.Fatal(err)
	}
}
