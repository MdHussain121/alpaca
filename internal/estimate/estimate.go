// Package estimate predicts how much VRAM and system RAM a configuration needs.
package estimate

import (
	"fmt"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
)

const (
	// MaxQ4KVContext is the maximum context length where q4_0 KV cache is allowed
	// by auto KV selection without degrading long-context needle recall.
	MaxQ4KVContext = 16384

	// ReserveVRAMMiB is the VRAM safety headroom preserved for display and runtime.
	ReserveVRAMMiB = 1024

	// MoECacheRAMReserveGiB is the free system RAM headroom required before
	// allowing manual --moe-cache-mib (prevents OS file-cache starvation).
	MoECacheRAMReserveGiB   = 4
	MoECacheRAMReserveBytes = uint64(MoECacheRAMReserveGiB) << 30

	// CPUDenseDefaultPrio boosts throughput for CPU-only dense inference.
	CPUDenseDefaultPrio          = "high"
	HybridMoEDefaultThreadsBatch = 0
)

// Result is the predicted memory split, all values in bytes.
type Result struct {
	WeightsVRAM uint64
	WeightsRAM  uint64
	KVVRAM      uint64
	KVRAM       uint64
	ComputeVRAM uint64
	ComputeRAM  uint64

	VRAMTotal uint64
	VRAMAvail uint64
	VRAMFree  uint64
	RAMTotal  uint64
	RAMAvail  uint64

	AllOffloaded  bool
	IsMoE         bool
	Approximate   bool
	UnifiedMemory bool
	GPUCount      int
	CommitAvail   uint64
	CommitUsed    uint64
	Notes         []string

	MoeCacheCapMiB  int
	MoeCacheReason  string
	KVReason        string
	KVQualityNote   string
	ThreadsReason   string
	LoadModeWarning string
}

// TotalFootprint is total predicted weights + KV + compute across VRAM and RAM.
func (r Result) TotalFootprint() uint64 {
	return r.WeightsVRAM + r.WeightsRAM + r.KVVRAM + r.KVRAM + r.ComputeVRAM + r.ComputeRAM
}

// EffectiveVRAMFree returns VRAMFree if known, otherwise VRAMAvail.
func (r Result) EffectiveVRAMFree() uint64 {
	if r.VRAMFree > 0 {
		return r.VRAMFree
	}
	return r.VRAMAvail
}

// VRAMUsed is everything predicted to live on the GPU.
func (r Result) VRAMUsed() uint64 { return r.WeightsVRAM + r.KVVRAM + r.ComputeVRAM }

// RAMUsed is everything predicted to live in system memory.
func (r Result) RAMUsed() uint64 { return r.WeightsRAM + r.KVRAM + r.ComputeRAM }

// VRAMRatio is used/available, where 1.0 exactly fills the card.
func (r Result) VRAMRatio() float64 {
	if r.VRAMAvail == 0 {
		return 0
	}
	return float64(r.VRAMUsed()) / float64(r.VRAMAvail)
}

// RAMRatio is used/available for system memory (including GPU buffers when using
// unified/integrated graphics that share system RAM).
func (r Result) RAMRatio() float64 {
	if r.RAMAvail == 0 {
		return 0
	}
	used := r.RAMUsed()
	if r.UnifiedMemory {
		used += r.VRAMUsed()
	}
	return float64(used) / float64(r.RAMAvail)
}

// CommitRatio returns estimated OS virtual commit over available commit limit.
func (r Result) CommitRatio() float64 {
	if r.CommitAvail == 0 {
		return 0
	}
	return float64(r.CommitUsed) / float64(r.CommitAvail)
}

// Fits reports whether VRAM, physical RAM, and OS commit budgets are respected.
func (r Result) Fits() bool {
	if r.VRAMRatio() > 1.0 || r.RAMRatio() > 1.0 {
		return false
	}
	if r.CommitAvail > 0 && r.CommitUsed > r.CommitAvail {
		return false
	}
	return true
}

// Compute predicts the memory split for meta under s on info.
func Compute(meta *gguf.Metadata, s flags.Settings, info hw.Info) Result {
	s.Normalize()

	vramTotal := info.VRAMTotal
	vramFree := info.VRAMFree
	gpuCount := 0
	if len(info.Devices) > 0 {
		var sumT, sumF uint64
		for _, d := range info.Devices {
			sumT += d.VRAMTotal
			sumF += d.VRAMFree
			gpuCount++
		}
		if sumT > 0 {
			vramTotal = sumT
			vramFree = sumF
		}
	} else if vramTotal > 0 {
		gpuCount = 1
	}
	if vramFree == 0 {
		vramFree = vramTotal
	}
	unified := info.HasUnifiedMemory() && vramTotal > 0
	res := Result{
		VRAMTotal:     vramTotal,
		VRAMAvail:     vramTotal,
		VRAMFree:      vramFree,
		RAMTotal:      info.RAMTotal,
		RAMAvail:      info.RAMAvail,
		CommitAvail:   info.RAMCommitAvail,
		UnifiedMemory: unified,
		GPUCount:      gpuCount,
	}
	if res.RAMAvail == 0 {
		res.RAMAvail = info.RAMTotal
	}
	if gpuCount > 1 {
		res.Notes = append(res.Notes, fmt.Sprintf("%d GPUs pooled", gpuCount))
	}

	layers := int(meta.BlockCount)
	if layers == 0 {
		layers = len(meta.Layers)
	}

	ngl := s.GPULayers
	if vramTotal == 0 {
		ngl = 0
	} else if ngl > layers {
		ngl = layers
	}
	allOffloaded := layers > 0 && ngl >= layers

	res.IsMoE = meta.IsMoE() || s.NCpuMoe > 0
	res.AllOffloaded = allOffloaded && s.NCpuMoe == 0
	res.KVQualityNote = KVQualityNote(s.KCache, s.VCache)

	// Weights, layer by layer (with fallback when per-layer tensor table is absent).
	var fallbackLayer uint64
	if layers > 0 && len(meta.Layers) == 0 && meta.TotalBytes > meta.GlobalBytes {
		fallbackLayer = (meta.TotalBytes - meta.GlobalBytes) / uint64(layers)
		res.Approximate = true
	}
	for l := 0; l < layers; l++ {
		var lb gguf.LayerBytes
		if l < len(meta.Layers) {
			lb = meta.Layers[l]
		} else if fallbackLayer > 0 {
			lb = gguf.LayerBytes{Total: fallbackLayer}
		}
		if l >= ngl {
			res.WeightsRAM += lb.Total
			continue
		}
		experts := lb.Expert
		dense := lb.Total - experts
		res.WeightsVRAM += dense
		if l < s.NCpuMoe {
			res.WeightsRAM += experts
		} else {
			res.WeightsVRAM += experts
		}
	}

	// Tensors outside the repeating layers (embeddings stay on host/CPU mmap,
	// output head moves to GPU when all layers are offloaded).
	embed := meta.EmbedBytes
	if embed > meta.GlobalBytes {
		embed = meta.GlobalBytes
	}
	res.WeightsRAM += embed
	if ngl > 0 && embed > 0 {
		res.Notes = append(res.Notes, "embed on RAM")
	}
	otherGlobal := meta.GlobalBytes - embed
	if allOffloaded && ngl > 0 {
		res.WeightsVRAM += otherGlobal
	} else {
		res.WeightsRAM += otherGlobal
	}

	// KV cache.
	kBytes := flags.ElemBytes(s.KCache)
	vBytes := flags.ElemBytes(s.VCache)
	headDim := float64(meta.HeadDim())
	valueDim := float64(meta.ValueDim())
	if headDim == 0 {
		res.Approximate = true
		res.Notes = append(res.Notes, "head dim unknown")
	}
	if meta.PerLayerKV() {
		res.Notes = append(res.Notes, "hybrid/SWA KV")
	}

	// llama-server allocates one KV cache per slot and -np sets the slot count,
	// so the cache the server really allocates is slots times this size.
	slots := s.Parallel
	if slots < 1 {
		slots = 1
	}
	if slots > 1 {
		res.Notes = append(res.Notes, fmt.Sprintf("%d slots", slots))
	}
	if s.NoKVOffload {
		res.Notes = append(res.Notes, "KV on RAM")
	}

	var kvVRAM, kvRAM float64
	for l := 0; l < layers; l++ {
		if !meta.IsKVLayer(l) {
			continue
		}
		heads := float64(meta.KVHeadsAt(l))
		if heads == 0 {
			continue
		}
		cells := float64(s.Context)
		if meta.IsSWALayer(l) {
			swaCells := int(meta.SlidingWindow) + s.UBatch
			if swaCells > 0 && swaCells < s.Context {
				cells = float64(swaCells)
			}
		}
		perLayer := cells * float64(slots) * heads * (headDim*kBytes + valueDim*vBytes)
		if !s.NoKVOffload && l < ngl {
			kvVRAM += perLayer
		} else {
			kvRAM += perLayer
		}
	}
	res.KVVRAM = uint64(kvVRAM)
	res.KVRAM = uint64(kvRAM)

	// Compute buffers: a few hundred MiB on the active execution backend, plus a
	// small host compute buffer whenever CPU/host tensors or KV are active.
	computeMiB := 256.0 + float64(s.UBatch)*0.5
	res.Notes = append(res.Notes, "compute buffer estimated")
	if ngl > 0 {
		res.ComputeVRAM = uint64(computeMiB * 1024 * 1024)
		if !allOffloaded || s.NCpuMoe > 0 || s.NoKVOffload || meta.EmbedBytes > 0 {
			res.ComputeRAM = uint64(32 * 1024 * 1024)
		}
	} else {
		res.ComputeRAM = uint64(computeMiB * 1024 * 1024)
	}

	if meta.UnknownTensorTypes != nil && len(meta.UnknownTensorTypes) > 0 {
		res.Approximate = true
		res.Notes = append(res.Notes, "unknown tensor types")
	}
	if unified && ngl > 0 {
		res.Notes = append(res.Notes, "shared iGPU memory")
	}
	// Virtual memory commit estimate: KV + compute allocations always commit;
	// non-mmap load mode (-lm none) also commits model weights into heap/VRAM.
	res.CommitUsed = res.KVRAM + res.KVVRAM + res.ComputeRAM + res.ComputeVRAM
	if s.LoadMode == "none" {
		res.CommitUsed += res.WeightsRAM + res.WeightsVRAM
	} else if unified {
		res.CommitUsed += res.WeightsVRAM
	}
	if res.CommitAvail > 0 && res.CommitUsed > res.CommitAvail {
		res.Notes = append(res.Notes, "exceeds OS commit limit")
	}
	if res.IsMoE {
		res.MoeCacheCapMiB, res.MoeCacheReason = MoECacheCapMiB(res, meta, info)
	}
	return res
}

// Breakdown is a one-line human summary showing where model weights, KV cache,
// and compute memory live (GPU VRAM vs CPU RAM).
func (r Result) Breakdown() string {
	return fmt.Sprintf("%s · %s · %s",
		fmtMemSplit("weights", r.WeightsVRAM, r.WeightsRAM),
		fmtMemSplit("KV", r.KVVRAM, r.KVRAM),
		fmtMemSplit("comp", r.ComputeVRAM, r.ComputeRAM))
}

func fmtMemSplit(label string, vram, ram uint64) string {
	total := vram + ram
	switch {
	case vram > 0 && ram > 0:
		return fmt.Sprintf("%s %s (%s GPU+%s RAM)", label, fmtMemCompact(total), fmtMemCompact(vram), fmtMemCompact(ram))
	case ram > 0:
		return fmt.Sprintf("%s %s RAM", label, fmtMemCompact(ram))
	default:
		return fmt.Sprintf("%s %s GPU", label, fmtMemCompact(vram))
	}
}

func fmtMemCompact(b uint64) string {
	if b == 0 {
		return "0"
	}
	if b >= 100<<20 {
		return fmt.Sprintf("%.1fG", float64(b)/(1<<30))
	}
	return fmt.Sprintf("%.0fM", float64(b)/(1<<20))
}

// Level classifies a ratio for colouring: 0 ok, 1 warning, 2 over budget.
func Level(ratio float64) int {
	switch {
	case ratio > 1.0:
		return 2
	case ratio > 0.85:
		return 1
	default:
		return 0
	}
}

func modelSizeBytes(meta *gguf.Metadata, res Result) uint64 {
	if meta != nil {
		if meta.FileSize > 0 {
			return uint64(meta.FileSize)
		}
		if meta.TotalBytes > 0 {
			return meta.TotalBytes
		}
	}
	return res.WeightsVRAM + res.WeightsRAM
}

// MoECacheAllowed reports whether manual --moe-cache-mib should be offered:
// requires MoE and available RAM >= model size + cache + 4 GiB.
func MoECacheAllowed(meta *gguf.Metadata, res Result, info hw.Info, cacheMiB int) bool {
	if meta == nil || !meta.IsMoE() {
		return false
	}
	ramAvail := info.RAMAvail
	if ramAvail == 0 {
		ramAvail = info.RAMTotal
	}
	if ramAvail == 0 {
		return false
	}
	need := modelSizeBytes(meta, res) + MoECacheRAMReserveBytes
	if cacheMiB > 0 {
		need += uint64(cacheMiB) << 20
	}
	return ramAvail >= need
}

// MoECacheCapMiB computes the safe --moe-cache-mib upper bound as the minimum
// of the VRAM headroom cap and the RAM page-cache headroom cap:
//
//	vramCap = floor((freeVRAM - estGPU - 1024 MiB) / 256) * 256
//	ramCap  = floor((availRAM - modelSize - 4 GiB) / 256) * 256
func MoECacheCapMiB(res Result, meta *gguf.Metadata, info hw.Info) (int, string) {
	freeVRAM := info.VRAMFree
	if freeVRAM == 0 {
		freeVRAM = info.VRAMTotal
	}
	if freeVRAM == 0 {
		return 0, "0M cap (no GPU VRAM)"
	}
	estGPU := res.WeightsVRAM + res.KVVRAM + res.ComputeVRAM
	reserve := uint64(ReserveVRAMMiB) << 20
	vramCap := 0
	if freeVRAM > estGPU+reserve {
		slackMiB := int((freeVRAM - estGPU - reserve) >> 20)
		vramCap = (slackMiB / 256) * 256
	}

	ramAvail := info.RAMAvail
	if ramAvail == 0 {
		ramAvail = info.RAMTotal
	}
	modelBytes := modelSizeBytes(meta, res)
	ramCap := 0
	if ramAvail > modelBytes+MoECacheRAMReserveBytes {
		slackRAMMiB := int((ramAvail - modelBytes - MoECacheRAMReserveBytes) >> 20)
		ramCap = (slackRAMMiB / 256) * 256
	}

	if vramCap == 0 {
		return 0, fmt.Sprintf("0M cap (GPU %dM + %dM res >= %dM free)",
			estGPU>>20, ReserveVRAMMiB, freeVRAM>>20)
	}
	if ramCap == 0 {
		return 0, fmt.Sprintf("0M cap (RAM %.1fG < model %.1fG + %dG)",
			float64(ramAvail)/(1<<30), float64(modelBytes)/(1<<30), MoECacheRAMReserveGiB)
	}
	if ramCap < vramCap {
		return ramCap, fmt.Sprintf("%dM cap (RAM %.1fG - model %.1fG - %dG)",
			ramCap, float64(ramAvail)/(1<<30), float64(modelBytes)/(1<<30), MoECacheRAMReserveGiB)
	}
	return vramCap, fmt.Sprintf("%dM cap (free %dM - GPU %dM - %dM res)",
		vramCap, freeVRAM>>20, estGPU>>20, ReserveVRAMMiB)
}

// KVQualityNote warns when any q4_0 KV cache is active so users see the tradeoff.
func KVQualityNote(kCache, vCache string) string {
	if kCache == "q4_0" || vCache == "q4_0" {
		return "q4_0 trades recall/detail for VRAM fit"
	}
	return ""
}

// SmartKV selects the first fitting KV cache pair from:
// f16/f16 -> q8_0/q8_0 -> q8_0/q4_0 -> q4_0/q4_0.
// Any pair containing q4_0 is allowed only when Context <= MaxQ4KVContext (needle verified).
func SmartKV(meta *gguf.Metadata, s flags.Settings, info hw.Info) (k, v, reason, qualityNote string) {
	type kvPair struct{ k, v string }
	candidates := []kvPair{
		{"f16", "f16"},
		{"q8_0", "q8_0"},
		{"q8_0", "q4_0"},
		{"q4_0", "q4_0"},
	}
	q4Allowed := s.Context <= MaxQ4KVContext
	freeVRAM := info.VRAMFree
	if freeVRAM == 0 {
		freeVRAM = info.VRAMTotal
	}
	ramAvail := info.RAMAvail
	if ramAvail == 0 {
		ramAvail = info.RAMTotal
	}
	reserve := uint64(ReserveVRAMMiB) << 20

	// CPU-only or -nkvo: KV cache resides in system RAM.
	if info.VRAMTotal == 0 || s.NoKVOffload {
		ramBudget := ramAvail
		if ramBudget > reserve {
			ramBudget -= reserve
		}
		for _, p := range candidates {
			if (p.k == "q4_0" || p.v == "q4_0") && !q4Allowed {
				continue
			}
			cs := s
			cs.KCache, cs.VCache = p.k, p.v
			cs.Normalize()
			cr := Compute(meta, cs, info)
			if ramAvail == 0 || cr.RAMUsed() <= ramBudget {
				return p.k, p.v, fmt.Sprintf("auto: %s/%s fits in RAM", p.k, p.v), KVQualityNote(p.k, p.v)
			}
		}
		if q4Allowed {
			return "q4_0", "q4_0", "auto: q4_0/q4_0 tight RAM fit", KVQualityNote("q4_0", "q4_0")
		}
		return "q8_0", "q8_0", fmt.Sprintf("auto: q8_0/q8_0 (q4_0 gated above %dK)", MaxQ4KVContext/1024), ""
	}

	// Pass 1: preserve >= 1024 MiB free VRAM headroom (and fit RAM).
	if freeVRAM > reserve {
		headroomBudget := freeVRAM - reserve
		for _, p := range candidates {
			if (p.k == "q4_0" || p.v == "q4_0") && !q4Allowed {
				continue
			}
			cs := s
			cs.KCache, cs.VCache = p.k, p.v
			cs.Normalize()
			cr := Compute(meta, cs, info)
			if cr.VRAMUsed() <= headroomBudget && (ramAvail == 0 || cr.RAMUsed() <= ramAvail) {
				freeLeft := float64(freeVRAM-cr.VRAMUsed()) / (1 << 30)
				return p.k, p.v, fmt.Sprintf("auto: %s/%s (%.1fG free)", p.k, p.v, freeLeft), KVQualityNote(p.k, p.v)
			}
		}
	}

	// Pass 2: fits inside freeVRAM without the full 1024 MiB slack.
	for _, p := range candidates {
		if (p.k == "q4_0" || p.v == "q4_0") && !q4Allowed {
			continue
		}
		cs := s
		cs.KCache, cs.VCache = p.k, p.v
		cs.Normalize()
		cr := Compute(meta, cs, info)
		if cr.VRAMUsed() <= freeVRAM && (ramAvail == 0 || cr.RAMUsed() <= ramAvail) {
			return p.k, p.v, fmt.Sprintf("auto: %s/%s (tight VRAM fit)", p.k, p.v), KVQualityNote(p.k, p.v)
		}
	}

	// Weights exceed VRAM (offload / --fit split): keep q8_0/q8_0 to preserve GPU layer headroom without q4_0 loss.
	if !q4Allowed {
		return "q8_0", "q8_0", fmt.Sprintf("auto: q8_0/q8_0 (q4_0 blocked >%dK — try -nkvo)", MaxQ4KVContext/1024), ""
	}
	return "q8_0", "q8_0", "auto: q8_0/q8_0 (preserves offload budget)", ""
}

// SmartThreadsAndPrio returns topology-derived mode-aware defaults:
// full GPU -> 0/normal (llama.cpp default),
// CPU-only dense -> P-core physical threads + --prio 2,
// hybrid MoE -> P-core logical threads,
// unknown topology -> 0 (falls back to llama.cpp default).
func SmartThreadsAndPrio(s flags.Settings, res Result, info hw.Info) (threads, threadsBatch int, prio, reason string) {
	isCPUOnly := info.VRAMTotal == 0 || s.GPULayers == 0
	isHybridMoE := res.IsMoE && (!res.AllOffloaded || s.NCpuMoe > 0 || !res.Fits())

	switch {
	case !res.IsMoE && isCPUOnly:
		if info.PCores <= 0 {
			return 0, 0, CPUDenseDefaultPrio, "CPU dense: default --prio 2 (unknown topology)"
		}
		t := info.PCores
		return t, 0, CPUDenseDefaultPrio, fmt.Sprintf("CPU dense: -t %d --prio 2", t)
	case isHybridMoE:
		if info.PLogicCores <= 0 {
			return 0, 0, "normal", "hybrid MoE: default (unknown topology)"
		}
		t := info.PLogicCores
		tb := HybridMoEDefaultThreadsBatch
		if tb > 0 && t < tb {
			tb = t
		}
		return t, tb, "normal", fmt.Sprintf("hybrid MoE: -t %d", t)
	default:
		return 0, 0, "normal", "GPU offload: default"
	}
}

// LoadModeUnsafeReason returns a non-empty reason if -lm none or -lm dio is
// unsafe (MoE, partial/CPU offload, or model + KV + overhead > 50% available RAM).
func LoadModeUnsafeReason(res Result, info hw.Info) string {
	if res.IsMoE {
		return "unsafe for MoE (use mmap to avoid commit stall)"
	}
	if info.VRAMTotal == 0 || !res.AllOffloaded {
		return "unsafe for CPU/partial offload (use mmap)"
	}
	ramAvail := info.RAMAvail
	if ramAvail == 0 {
		ramAvail = info.RAMTotal
	}
	if ramAvail == 0 || res.TotalFootprint() > ramAvail/2 {
		return "model+KV+compute > 50% avail RAM (use mmap)"
	}
	return ""
}

// LoadModeWarning warns if the user explicitly forces none or dio.
// -lm dio always warns because bypassing the OS page cache destroys warm restart speed.
func LoadModeWarning(mode string, res Result, info hw.Info) string {
	if mode != "none" && mode != "dio" {
		return ""
	}
	r := LoadModeUnsafeReason(res, info)
	if mode == "dio" {
		if r != "" {
			return fmt.Sprintf("! -lm dio %s; bypasses OS page cache (slow warm restarts)", r)
		}
		return "! -lm dio bypasses OS page cache (slow warm restarts; prefer auto/mmap)"
	}
	if r != "" {
		return fmt.Sprintf("! -lm %s %s", mode, r)
	}
	return ""
}

// SmartLoadMode chooses the concrete -lm mode for Settings with LoadMode=="auto":
// full-GPU dense models that fit VRAM and stay within 50% available RAM use
// "none", while MoE, partial/CPU offload, or large-RAM models use "mmap".
func SmartLoadMode(res Result, info hw.Info) string {
	if LoadModeUnsafeReason(res, info) != "" {
		return "mmap"
	}
	if info.VRAMTotal > 0 && res.AllOffloaded && res.Fits() {
		return "none"
	}
	return "mmap"
}

// NeedsAutoFit reports whether --fit on is warranted (VRAM tight >90%,
// overflowing, or partial/MoE host split).
func NeedsAutoFit(res Result) bool {
	if res.VRAMAvail == 0 {
		return false
	}
	return !res.Fits() || res.VRAMRatio() > 0.90 || !res.AllOffloaded
}

// ResolveSettings computes res for (meta, s, info) and returns a copy of s with
// auto KV, mode-based threads/prio, MoE cache cap, and smart LoadMode resolved.
func ResolveSettings(s flags.Settings, meta *gguf.Metadata, info hw.Info) (flags.Settings, Result) {
	s.Normalize()
	if meta == nil {
		return s, Result{}
	}
	var kvReason, kvQuality string
	if s.AutoKV {
		s.KCache, s.VCache, kvReason, kvQuality = SmartKV(meta, s, info)
		s.Normalize()
	} else {
		kvQuality = KVQualityNote(s.KCache, s.VCache)
	}

	res := Compute(meta, s, info)
	res.KVReason = kvReason
	res.KVQualityNote = kvQuality

	modeT, modeTB, modePrio, modeReason := SmartThreadsAndPrio(s, res, info)
	if s.Threads == 0 {
		s.Threads = modeT
		if s.ThreadsBatch == 0 {
			s.ThreadsBatch = modeTB
		}
	}
	if s.Prio == "auto" {
		s.Prio = modePrio
	}
	res.ThreadsReason = modeReason

	// Never auto-enable --moe-cache-mib; it is strictly manual-opt-in and
	// constrained by both VRAM and RAM page-cache caps.
	if !res.IsMoE || s.MoeCacheMiB < 0 {
		s.MoeCacheMiB = 0
		if !res.IsMoE {
			res.MoeCacheReason = ""
		}
	}
	if res.IsMoE && s.MoeCacheMiB > 0 {
		if !MoECacheAllowed(meta, res, info, s.MoeCacheMiB) || s.MoeCacheMiB > res.MoeCacheCapMiB {
			res.MoeCacheReason = fmt.Sprintf("! %dM > %s", s.MoeCacheMiB, res.MoeCacheReason)
		}
	}

	if s.LoadMode == "auto" {
		s.LoadMode = SmartLoadMode(res, info)
	} else {
		res.LoadModeWarning = LoadModeWarning(s.LoadMode, res, info)
	}
	return s, res
}
