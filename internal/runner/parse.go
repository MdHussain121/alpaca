package runner

import (
	"regexp"
	"strconv"
	"strings"
)

// Every llama.cpp log pattern lives here so a format change is a one-file fix.
var (
	reKVCache     = regexp.MustCompile(`llama_kv_cache:\s*size\s*=\s*([0-9.]+)\s*MiB`)
	reKVCacheKV   = regexp.MustCompile(`K \([^)]*\):\s*([0-9.]+)\s*MiB,\s*V \([^)]*\):\s*([0-9.]+)\s*MiB`)
	reGPUKVBuf    = regexp.MustCompile(`llama_kv_cache:\s*(CUDA\d*|Vulkan\d*|ROCm\d*|Metal\d*)\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB`)
	reCPUKVBuf    = regexp.MustCompile(`llama_kv_cache:\s*(?:CPU(?:_Mapped|_Host)?|CUDA_Host)\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB`)
	reGenEval     = regexp.MustCompile(`(?:^|\s)eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*[a-z]+.*?([0-9.]+)\s*tokens per second`)
	rePrompt      = regexp.MustCompile(`prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*[a-z]+.*?([0-9.]+)\s*tokens per second`)
	reVRAMBuf     = regexp.MustCompile(`(?i)(CUDA\d*|Vulkan\d*|ROCm\d*|Metal\d*)\s+model buffer size\s*=\s*([0-9.]+)\s*MiB`)
	reVRAMCompute = regexp.MustCompile(`(?i)(CUDA\d*|Vulkan\d*|ROCm\d*|Metal\d*)\s+compute buffer size\s*=\s*([0-9.]+)\s*MiB`)
	reCPUModel    = regexp.MustCompile(`(?i)(?:CPU(?:_Mapped|_REPACK|_AARCH64)?|CUDA_Host)\s+model buffer size\s*=\s*([0-9.]+)\s*MiB`)
	reCPUCompute  = regexp.MustCompile(`(?i)(?:CPU(?:_Mapped)?|CUDA_Host)\s+compute buffer size\s*=\s*([0-9.]+)\s*MiB`)
	reLoadFail    = regexp.MustCompile(`(?i)(error loading model|failed to load model|unknown model architecture|out of memory|CUDA error|cannot allocate)`)
	reListening   = regexp.MustCompile(`server is listening on`)
	reLayers      = regexp.MustCompile(`offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU`)
	reNCtx        = regexp.MustCompile(`llama_context:\s*n_ctx\s*=\s*([0-9]+)`)
)

// Metrics is the live state parsed out of the llama-server log.
type Metrics struct {
	KVActualMiB   float64
	KMiB          float64
	VMiB          float64
	GPUKVMiB      float64
	CPUKVMiB      float64
	GenTokPerSec  float64
	PromptTokSec  float64
	GPUModelMiB   float64
	GPUComputeMiB float64
	CPUModelMiB   float64
	CPUComputeMiB float64
	LayersOnGPU   int
	TotalLayers   int
	NCtx          int
	Listening     bool
}

// ParseLine folds one log line into m, reporting whether anything changed.
func ParseLine(line string, m *Metrics) bool {
	changed := false

	if sub := reKVCache.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[1], 64); err == nil {
			m.KVActualMiB += v
			changed = true
		}
	}
	if sub := reKVCacheKV.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[1], 64); err == nil {
			m.KMiB += v
			changed = true
		}
		if v, err := strconv.ParseFloat(sub[2], 64); err == nil {
			m.VMiB += v
			changed = true
		}
	}
	if sub := reGPUKVBuf.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[2], 64); err == nil {
			m.GPUKVMiB += v
			changed = true
		}
	}
	if sub := reCPUKVBuf.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[1], 64); err == nil {
			m.CPUKVMiB += v
			changed = true
		}
	}
	if sub := rePrompt.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[3], 64); err == nil {
			m.PromptTokSec = v
			changed = true
		}
	}
	if sub := reGenEval.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[3], 64); err == nil {
			m.GenTokPerSec = v
			changed = true
		}
	}
	if sub := reVRAMBuf.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[2], 64); err == nil {
			m.GPUModelMiB += v
			changed = true
		}
	}
	if sub := reVRAMCompute.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[2], 64); err == nil {
			m.GPUComputeMiB += v
			changed = true
		}
	}
	if sub := reCPUModel.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[1], 64); err == nil {
			m.CPUModelMiB += v
			changed = true
		}
	}
	if sub := reCPUCompute.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.ParseFloat(sub[1], 64); err == nil {
			m.CPUComputeMiB += v
			changed = true
		}
	}
	if sub := reLayers.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.Atoi(sub[1]); err == nil {
			m.LayersOnGPU = v
			changed = true
		}
		if v, err := strconv.Atoi(sub[2]); err == nil {
			m.TotalLayers = v
			changed = true
		}
	}
	if sub := reNCtx.FindStringSubmatch(line); sub != nil {
		if v, err := strconv.Atoi(sub[1]); err == nil && v > 0 {
			m.NCtx = v
			changed = true
		}
	}
	if reListening.MatchString(line) {
		m.Listening = true
		changed = true
	}
	return changed
}

// IsErrorLine reports whether a log line looks like a fatal problem.
func IsErrorLine(line string) bool {
	return reLoadFail.MatchString(line) || strings.HasPrefix(strings.ToLower(strings.TrimSpace(line)), "error")
}

// ProgressFromLine extracts a load percentage if llama.cpp logged one.
func ProgressFromLine(line string) (float64, bool) {
	// e.g. "llama_model_loader: - kv  12/  36" or "load_tensors: ... 45%"
	if i := strings.Index(line, "load_tensors:"); i >= 0 {
		rest := line[i:]
		if j := strings.Index(rest, "%"); j > 0 {
			start := j
			for start > 0 && (rest[start-1] >= '0' && rest[start-1] <= '9') {
				start--
			}
			if v, err := strconv.ParseFloat(rest[start:j], 64); err == nil {
				return v / 100, true
			}
		}
	}
	return 0, false
}
