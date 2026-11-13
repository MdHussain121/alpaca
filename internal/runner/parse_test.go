package runner

import "testing"

func TestParseKVAndLayerLines(t *testing.T) {
	m := Metrics{}
	lines := []string{
		"llama_kv_cache: size = 512.00 MiB (8192 cells, 36 layers, 1/1 seqs), K (q8_0): 256.00 MiB, V (f16): 256.00 MiB",
		"llm_load_tensors: offloaded 36/36 layers to GPU",
		"llama_perf_context_print: prompt eval time =   120.00 ms /    64 tokens (    1.88 ms per token,   533.33 tokens per second)",
		"llama_perf_context_print:        eval time =  1000.00 ms /   128 runs   (    7.81 ms per token,   128.00 tokens per second)",
	}
	for _, l := range lines {
		if !ParseLine(l, &m) {
			t.Fatalf("line not recognised: %s", l)
		}
	}
	if m.KVActualMiB != 512 {
		t.Errorf("KV = %v", m.KVActualMiB)
	}
	if m.KMiB != 256 || m.VMiB != 256 {
		t.Errorf("K/V = %v/%v", m.KMiB, m.VMiB)
	}
	if m.LayersOnGPU != 36 || m.TotalLayers != 36 {
		t.Errorf("layers = %d/%d", m.LayersOnGPU, m.TotalLayers)
	}
	if m.PromptTokSec != 533.33 {
		t.Errorf("prompt t/s = %v", m.PromptTokSec)
	}
	if m.GenTokPerSec != 128 {
		t.Errorf("gen t/s = %v", m.GenTokPerSec)
	}
}

func TestParseLoadMode(t *testing.T) {
	m := Metrics{}
	if !ParseLine("load_tensors: loading model tensors, this can take a while... (load_mode = mmap)", &m) {
		t.Fatal("load mode line not recognised")
	}
	if m.LoadMode != "mmap" {
		t.Fatalf("LoadMode = %q, want mmap", m.LoadMode)
	}
}

func TestParseIgnoresUnrelatedLines(t *testing.T) {
	m := Metrics{}
	if ParseLine("llama_model_loader: loaded meta data", &m) {
		t.Fatal("unrelated line reported a change")
	}
	if !IsErrorLine("error loading model: bad magic") {
		t.Fatal("error line not detected")
	}
	if IsErrorLine("llama_model_loader: loaded meta data") {
		t.Fatal("normal line flagged as an error")
	}
}

func TestProgressFromLine(t *testing.T) {
	if v, ok := ProgressFromLine("load_tensors: loading model tensors  45%"); !ok || v != 0.45 {
		t.Fatalf("progress = %v, %v", v, ok)
	}
	if _, ok := ProgressFromLine("no percent here"); ok {
		t.Fatal("reported progress without a percentage")
	}
}

func TestParseHybridKVAndCPURAMBuffers(t *testing.T) {
	m := Metrics{}
	lines := []string{
		"load_tensors:   CPU_Mapped model buffer size =   262.50 MiB",
		"load_tensors:        CUDA0 model buffer size =  2474.63 MiB",
		"llama_kv_cache:      CUDA0 KV buffer size =   153.00 MiB",
		"llama_kv_cache: size =  153.00 MiB (8192 cells, 36 layers, 1/1 seqs), K (q8_0): 76.50 MiB, V (q8_0): 76.50 MiB",
		"llama_kv_cache:      CUDA0 KV buffer size =    57.38 MiB",
		"llama_kv_cache: size =   57.38 MiB (1024 cells, 36 layers, 1/1 seqs), K (q8_0): 28.69 MiB, V (q8_0): 28.69 MiB",
		"sched_reserve:      CUDA0 compute buffer size =    97.02 MiB",
		"sched_reserve:  CUDA_Host compute buffer size =    29.03 MiB",
		"llama_kv_cache:        CPU KV buffer size =    64.00 MiB",
	}
	for _, l := range lines {
		ParseLine(l, &m)
	}
	if m.KVActualMiB < 210.3 || m.KVActualMiB > 210.4 {
		t.Fatalf("KVActualMiB = %v, want 210.38", m.KVActualMiB)
	}
	if m.CPUModelMiB != 262.50 || m.GPUModelMiB != 2474.63 {
		t.Fatalf("model buffers CPU=%v GPU=%v", m.CPUModelMiB, m.GPUModelMiB)
	}
	if m.GPUComputeMiB != 97.02 || m.CPUComputeMiB != 29.03 || m.CPUKVMiB != 64.00 {
		t.Fatalf("compute/kv host buffers mismatch: %+v", m)
	}
}
