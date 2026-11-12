package bench

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/windows"
)

// MetricSummary aggregates min, median, max across repeated benchmark runs.
type MetricSummary struct {
	Min    float64 `json:"min"`
	Median float64 `json:"median"`
	Max    float64 `json:"max"`
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"stddev"`
}

func summarize(vals []float64) MetricSummary {
	if len(vals) == 0 {
		return MetricSummary{}
	}
	cp := slices.Clone(vals)
	slices.Sort(cp)
	n := len(cp)
	var med float64
	if n%2 == 1 {
		med = cp[n/2]
	} else {
		med = 0.5 * (cp[n/2-1] + cp[n/2])
	}
	var sum float64
	for _, v := range cp {
		sum += v
	}
	mean := sum / float64(n)
	var sq float64
	for _, v := range cp {
		d := v - mean
		sq += d * d
	}
	var sd float64
	if n > 1 {
		sd = math.Sqrt(sq / float64(n-1))
	}
	return MetricSummary{
		Min:    cp[0],
		Median: med,
		Max:    cp[n-1],
		Mean:   mean,
		StdDev: sd,
	}
}

// GPUSample captures instantaneous nvidia-smi telemetry.
type GPUSample struct {
	VRAMUsedMiB float64 `json:"vram_used_mib"`
	GPUUtilPct  float64 `json:"gpu_util_pct"`
	ClockMHz    float64 `json:"clock_mhz"`
	PowerW      float64 `json:"power_w"`
	TempC       float64 `json:"temp_c"`
}

func sampleGPU() (GPUSample, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=memory.used,utilization.gpu,clocks.current.graphics,power.draw,temperature.gpu",
		"--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return GPUSample{}, false
	}
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	parts := strings.Split(line, ",")
	if len(parts) < 5 {
		return GPUSample{}, false
	}
	parse := func(s string) float64 {
		v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
		return v
	}
	return GPUSample{
		VRAMUsedMiB: parse(parts[0]),
		GPUUtilPct:  parse(parts[1]),
		ClockMHz:    parse(parts[2]),
		PowerW:      parse(parts[3]),
		TempC:       parse(parts[4]),
	}, true
}

// processCPUTime returns total user+kernel CPU time consumed by pid.
func processCPUTime(pid int) (time.Duration, error) {
	if pid <= 0 {
		return 0, fmt.Errorf("invalid pid %d", pid)
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return 0, err
	}
	defer windows.CloseHandle(h)
	var cTime, eTime, kTime, uTime windows.Filetime
	if err := windows.GetProcessTimes(h, &cTime, &eTime, &kTime, &uTime); err != nil {
		return 0, err
	}
	ns := (kTime.Nanoseconds() + uTime.Nanoseconds())
	return time.Duration(ns) * time.Nanosecond, nil
}

// measureIdleCPU measures process CPU utilization (%) over dur while idle.
func measureIdleCPU(pid int, dur time.Duration) (float64, error) {
	t0, err := processCPUTime(pid)
	if err != nil {
		return 0, err
	}
	w0 := time.Now()
	time.Sleep(dur)
	t1, err := processCPUTime(pid)
	if err != nil {
		return 0, err
	}
	wall := time.Since(w0)
	if wall <= 0 {
		return 0, nil
	}
	return 100.0 * float64(t1-t0) / float64(wall), nil
}

// evictStandbyCache writes and reads a disposable multi-GB buffered file on an NVMe
// scratch directory to evict cached USB/model pages from the Windows Standby List.
func evictStandbyCache(dir string, sizeGB int) (time.Duration, error) {
	if sizeGB <= 0 {
		return 0, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, err
	}
	tmpPath := filepath.Join(dir, fmt.Sprintf("evict-%d.bin", time.Now().UnixNano()))
	defer os.Remove(tmpPath)

	start := time.Now()
	chunk := make([]byte, 8<<20) // 8 MiB pattern
	for i := range chunk {
		chunk[i] = byte((i * 131) ^ (i >> 8))
	}
	totalBytes := int64(sizeGB) << 30

	f, err := os.Create(tmpPath)
	if err != nil {
		return 0, err
	}
	for written := int64(0); written < totalBytes; {
		n := len(chunk)
		if rem := totalBytes - written; rem < int64(n) {
			n = int(rem)
		}
		if _, err := f.Write(chunk[:n]); err != nil {
			f.Close()
			return 0, err
		}
		written += int64(n)
	}
	if err := f.Close(); err != nil {
		return 0, err
	}

	// Sequential read pass ensures the file pages populate the Standby/Active lists.
	rf, err := os.Open(tmpPath)
	if err != nil {
		return 0, err
	}
	defer rf.Close()
	buf := make([]byte, 8<<20)
	for {
		_, rerr := rf.Read(buf)
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return 0, rerr
		}
	}
	return time.Since(start), nil
}

// StartupBreakdown records wall-clock elapsed milliseconds from process spawn
// to key llama-server startup milestones observed in logs.
type StartupBreakdown struct {
	BackendInitMs      float64 `json:"backend_init_ms"`
	FitStartMs         float64 `json:"fit_start_ms"`
	FitDoneMs          float64 `json:"fit_done_ms"`
	ModelLoadStartMs   float64 `json:"model_load_start_ms"`
	LoadTensorsStartMs float64 `json:"load_tensors_start_ms"`
	LoadTensorsDoneMs  float64 `json:"load_tensors_done_ms"`
	LoadAllDoneMs      float64 `json:"load_all_done_ms"`
	KVAllocDoneMs      float64 `json:"kv_alloc_done_ms"`
	GraphReserveMs     float64 `json:"graph_reserve_ms"`
	WarmupStartMs      float64 `json:"warmup_start_ms"`
	WarmupDoneMs       float64 `json:"warmup_done_ms"`
	ListeningMs        float64 `json:"listening_ms"`
	ReadyHTTPMs        float64 `json:"ready_http_ms"`
}

// milestoneObserver updates StartupBreakdown as lines arrive from runner.Process.
func updateMilestones(sb *StartupBreakdown, elapsed time.Duration, line string) {
	ms := float64(elapsed.Microseconds()) / 1000.0
	low := strings.ToLower(line)
	switch {
	case sb.BackendInitMs == 0 && (strings.Contains(low, "device_info:") || strings.Contains(low, "ggml_cuda_init:") || strings.Contains(low, "load_backend: loaded")):
		sb.BackendInitMs = ms
	case sb.FitStartMs == 0 && strings.Contains(low, "common_fit_a: fitting params"):
		sb.FitStartMs = ms
	case sb.FitDoneMs == 0 && strings.Contains(low, "common_fit_params: fitting params to free memory took"):
		sb.FitDoneMs = ms
	case sb.ModelLoadStartMs == 0 && strings.Contains(low, "llama_model_loader: loaded meta data"):
		sb.ModelLoadStartMs = ms
	case sb.LoadTensorsStartMs == 0 && strings.Contains(low, "load_tensors: loading model tensors"):
		sb.LoadTensorsStartMs = ms
	case sb.LoadTensorsDoneMs == 0 && strings.Contains(low, "model buffer size ="):
		sb.LoadTensorsDoneMs = ms
	case sb.LoadAllDoneMs == 0 && strings.Contains(low, "llama_context: constructing llama_context"):
		sb.LoadAllDoneMs = ms
	case strings.Contains(low, "llama_kv_cache:") && strings.Contains(low, "kv buffer size"):
		sb.KVAllocDoneMs = ms
	case sb.GraphReserveMs == 0 && strings.Contains(low, "sched_reserve: reserve took"):
		sb.GraphReserveMs = ms
	case sb.WarmupStartMs == 0 && strings.Contains(low, "warming up the model"):
		sb.WarmupStartMs = ms
	case sb.WarmupDoneMs == 0 && sb.WarmupStartMs > 0 && strings.Contains(low, "srv    load_model: initializing"):
		sb.WarmupDoneMs = ms
	case sb.ListeningMs == 0 && (strings.Contains(low, "listening on http") || strings.Contains(low, "server is listening") || strings.Contains(low, "starting the main loop")):
		sb.ListeningMs = ms
	}
}

var _ = regexp.MustCompile // keep import clean if unused

func formatFloatCSV(v float64) string {
	return strconv.FormatFloat(v, 'f', 3, 64)
}

func trimBOM(b []byte) []byte {
	return bytes.TrimPrefix(b, []byte{0xEF, 0xBB, 0xBF})
}

