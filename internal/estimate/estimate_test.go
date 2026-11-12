package estimate

import (
	"math"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
)

func testMeta(layers int, expertBytes, denseBytes uint64) *gguf.Metadata {
	var exp uint64
	if expertBytes > 0 {
		exp = 8
	}
	m := &gguf.Metadata{
		BlockCount:    uint64(layers),
		ContextLength: 32768,
		EmbeddingLen:  4096,
		HeadCount:     32,
		KeyLength:     128,
		ValueLength:   128,
		ExpertCount:   exp,
		GlobalBytes:   1 << 30,
	}
	m.HeadCountKV = []uint64{8}
	for i := 0; i < layers; i++ {
		m.Layers = append(m.Layers, gguf.LayerBytes{Total: denseBytes + expertBytes, Expert: expertBytes})
	}
	return m
}

func testHW() hw.Info {
	return hw.Info{
		VRAMTotal: 8 << 30,
		VRAMFree:  8 << 30,
		RAMTotal:  16 << 30,
		RAMAvail:  12 << 30,
	}
}

func TestKVFormulaMatchesHandComputation(t *testing.T) {
	meta := testMeta(4, 0, 1<<20)
	s := flags.Default(8, 8080)
	s.Context = 4096
	s.KCache = "f16"
	s.VCache = "f16"
	s.GPULayers = flags.AllCount

	res := Compute(meta, s, testHW())

	// 4 layers * 4096 ctx * 8 kv heads * 128 head dim * 2 bytes * 2 (K and V)
	want := uint64(4 * 4096 * 8 * 128 * 2 * 2)
	if res.KVVRAM != want {
		t.Fatalf("KV VRAM = %d, want %d", res.KVVRAM, want)
	}
}

func TestQuantizedCacheShrinksKV(t *testing.T) {
	meta := testMeta(4, 0, 1<<20)
	base := flags.Default(8, 8080)
	base.Context = 4096
	base.KCache, base.VCache = "f16", "f16"
	f16 := Compute(meta, base, testHW())

	q := base
	q.KCache, q.VCache = "q8_0", "q4_0"
	q.Normalize()
	quant := Compute(meta, q, testHW())

	if quant.KVVRAM >= f16.KVVRAM {
		t.Fatalf("quantized KV (%d) should be smaller than f16 (%d)", quant.KVVRAM, f16.KVVRAM)
	}
	// q8_0 + q4_0 is (1.0625+0.5625)/4 = 40.6% of f16+f16.
	ratio := float64(quant.KVVRAM) / float64(f16.KVVRAM)
	if math.Abs(ratio-0.40625) > 0.001 {
		t.Fatalf("ratio = %.4f, want 0.40625", ratio)
	}
}

// llama-server allocates one KV cache per slot, so -np must be in the estimate.
func TestKVScalesWithParallelSlots(t *testing.T) {
	meta := testMeta(4, 0, 1<<20)
	s := flags.Default(8, 8080)
	s.Context = 4096
	s.KCache, s.VCache = "f16", "f16"
	s.Parallel = 1
	one := Compute(meta, s, testHW())

	s.Parallel = 4
	four := Compute(meta, s, testHW())

	if four.KVVRAM != one.KVVRAM*4 {
		t.Fatalf("4 slots KV = %d, want 4x %d", four.KVVRAM, one.KVVRAM)
	}
}

func TestExpertSplitFollowsNCpuMoe(t *testing.T) {
	const layers = 4
	const expert = uint64(100 << 20)
	const dense = uint64(10 << 20)
	meta := testMeta(layers, expert, dense)

	s := flags.Default(8, 8080)
	s.GPULayers = flags.AllCount
	s.NCpuMoe = 0
	allGPU := Compute(meta, s, testHW())

	s.NCpuMoe = 3
	someOff := Compute(meta, s, testHW())

	if someOff.WeightsRAM != allGPU.WeightsRAM+3*expert {
		t.Fatalf("RAM weights = %d, want %d", someOff.WeightsRAM, allGPU.WeightsRAM+3*expert)
	}
	if someOff.WeightsVRAM != allGPU.WeightsVRAM-3*expert {
		t.Fatalf("VRAM weights = %d, want %d", someOff.WeightsVRAM, allGPU.WeightsVRAM-3*expert)
	}
}

func TestPartialOffloadMovesLayersToRAM(t *testing.T) {
	meta := testMeta(10, 0, 100<<20)
	s := flags.Default(8, 8080)
	s.GPULayers = 5

	res := Compute(meta, s, testHW())

	if want := uint64(5 * 100 << 20); res.WeightsVRAM != want {
		t.Fatalf("VRAM weights = %d, want %d", res.WeightsVRAM, want)
	}
	if res.WeightsRAM < uint64(5*100<<20) {
		t.Fatalf("RAM weights = %d, expected the unoffloaded layers", res.WeightsRAM)
	}
	if res.KVRAM == 0 {
		t.Fatal("unoffloaded layers should contribute KV to RAM")
	}
}

func TestPerLayerKVIsNotedAndComputed(t *testing.T) {
	meta := testMeta(4, 0, 1<<20)
	meta.HeadCountKV = []uint64{8, 8, 2, 2}
	res := Compute(meta, flags.Default(8, 8080), testHW())
	if !strings.Contains(strings.Join(res.Notes, ","), "hybrid/SWA KV") {
		t.Fatalf("expected hybrid/SWA note, got %v", res.Notes)
	}
}

func TestResultHelpers(t *testing.T) {
	r := Result{
		WeightsVRAM: 1 << 30, KVVRAM: 1 << 30, ComputeVRAM: 1 << 30,
		WeightsRAM: 2 << 30, KVRAM: 1 << 30, ComputeRAM: 1 << 30,
		VRAMTotal: 8 << 30, VRAMAvail: 8 << 30,
		RAMTotal: 16 << 30, RAMAvail: 8 << 30,
	}
	if r.VRAMUsed() != 3<<30 {
		t.Errorf("VRAMUsed = %d", r.VRAMUsed())
	}
	if r.RAMUsed() != 4<<30 {
		t.Errorf("RAMUsed = %d", r.RAMUsed())
	}
	if math.Abs(r.VRAMRatio()-3.0/8.0) > 1e-9 {
		t.Errorf("VRAMRatio = %v", r.VRAMRatio())
	}
	if !r.Fits() {
		t.Error("3/8 VRAM and 4/8 RAM should fit")
	}
	if !strings.Contains(r.Breakdown(), "weights") {
		t.Errorf("breakdown = %q", r.Breakdown())
	}

	var zero Result
	if zero.VRAMRatio() != 0 || zero.RAMRatio() != 0 {
		t.Error("zero result must not divide by zero")
	}
}

func TestEstimateExplainsItsApproximations(t *testing.T) {
	s := flags.Default(4, 8080)
	s.Parallel = 4
	res := Compute(testMeta(4, 0, 1<<20), s, testHW())
	joined := strings.Join(res.Notes, ", ")
	if !strings.Contains(joined, "slots") {
		t.Errorf("slot count not explained: %v", res.Notes)
	}
	if !strings.Contains(joined, "compute") {
		t.Errorf("compute buffer overhead not explained: %v", res.Notes)
	}
}

func TestHeadDimUnknownIsApproximate(t *testing.T) {
	meta := testMeta(4, 0, 1<<20)
	meta.KeyLength = 0
	meta.HeadCount = 0
	meta.EmbeddingLen = 0
	res := Compute(meta, flags.Default(4, 8080), testHW())
	if !res.Approximate {
		t.Fatal("an unknown head dimension must mark the estimate approximate")
	}
}

func TestFitsAndLevel(t *testing.T) {
	meta := testMeta(8, 0, 4<<30)
	s := flags.Default(8, 8080)
	res := Compute(meta, s, testHW())
	if res.Fits() {
		t.Fatal("a 32GB model must not fit an 8GB card")
	}
	if Level(1.2) != 2 {
		t.Error("over budget should be level 2")
	}
	if Level(0.9) != 1 {
		t.Error("90% should be a warning")
	}
	if Level(0.5) != 0 {
		t.Error("50% should be ok")
	}
}

func TestNoKVOffloadSWAAndCPUOnly(t *testing.T) {
	// 4-layer SWA model: 3 SWA layers (window 512, ubatch 512 -> 1024 cells) + 1 full layer (8192 cells).
	meta := testMeta(4, 0, 10<<20)
	meta.SlidingWindow = 512
	meta.SlidingWindowPat = []bool{true, true, true, false}
	meta.HeadCountKV = []uint64{4}
	meta.KeyLength = 128
	meta.ValueLength = 128
	meta.EmbedBytes = 256 << 20

	s := flags.Default(8, 8080)
	s.Context = 8192
	s.UBatch = 512
	s.KCache, s.VCache = "f16", "f16"

	res := Compute(meta, s, testHW())
	// KV = (3*1024 + 1*8192) * 4 * (128*2 + 128*2) = 11264 * 4 * 512
	wantKV := uint64((3*1024 + 8192) * 4 * 512)
	if res.KVVRAM != wantKV {
		t.Fatalf("SWA KVVRAM = %d, want %d", res.KVVRAM, wantKV)
	}
	// Embeddings stay in RAM even on full offload.
	if res.WeightsRAM < meta.EmbedBytes {
		t.Fatalf("WeightsRAM = %d, expected >= EmbedBytes %d", res.WeightsRAM, meta.EmbedBytes)
	}

	// NoKVOffload moves all KV into KVRAM.
	s.NoKVOffload = true
	nkvo := Compute(meta, s, testHW())
	if nkvo.KVVRAM != 0 || nkvo.KVRAM != wantKV {
		t.Fatalf("NoKVOffload got KVVRAM=%d KVRAM=%d, want 0 and %d", nkvo.KVVRAM, nkvo.KVRAM, wantKV)
	}

	if !strings.Contains(nkvo.Breakdown(), "KV") || !strings.Contains(nkvo.Breakdown(), "RAM") {
		t.Fatalf("NoKVOffload Breakdown missing RAM KV info: %q", nkvo.Breakdown())
	}
	if !strings.Contains(res.Breakdown(), "GPU") {
		t.Fatalf("Hybrid Breakdown missing GPU split: %q", res.Breakdown())
	}

	// CPU-only host (VRAMTotal == 0) puts all weights, KV, and compute in RAM.
	s.NoKVOffload = false
	cpuHW := hw.Info{RAMTotal: 16 << 30, RAMAvail: 12 << 30}
	cpuRes := Compute(meta, s, cpuHW)
	if cpuRes.VRAMUsed() != 0 {
		t.Fatalf("CPU-only host must have 0 VRAMUsed, got %d", cpuRes.VRAMUsed())
	}
	if cpuRes.KVRAM != wantKV || cpuRes.WeightsRAM == 0 || cpuRes.ComputeRAM == 0 {
		t.Fatalf("CPU-only host must account everything in RAM: %+v", cpuRes)
	}
}

func TestSmartLoadModeGuardsAndDefaults(t *testing.T) {
	meta := testMeta(4, 0, 200<<20) // ~1.8G weights + ~1.5G KV/comp <= 50% of 12G RAMAvail
	s := flags.Default(0, 8080)
	s.Context = 4096
	resolved, res := ResolveSettings(s, meta, testHW())
	if resolved.LoadMode != "none" {
		t.Fatalf("dense full-GPU fit <=50%% RAM must resolve auto load_mode to none, got %q", resolved.LoadMode)
	}
	if NeedsAutoFit(res) {
		t.Fatalf("comfortably fitting model (%v) must not require --fit", res.VRAMRatio())
	}
	if res.LoadModeWarning != "" {
		t.Fatalf("unexpected load mode warning: %q", res.LoadModeWarning)
	}

	// Guard 1: MoE model (even if small enough to fit VRAM & <=50% RAM) must NEVER pick none/dio.
	moeFit := testMeta(4, 100<<20, 100<<20)
	if resMoE := Compute(moeFit, s, testHW()); SmartLoadMode(resMoE, testHW()) != "mmap" {
		t.Fatalf("MoE auto load mode must be mmap, got %q", SmartLoadMode(resMoE, testHW()))
	}

	// Guard 2: Partial offload (!AllOffloaded) must NEVER pick none/dio.
	sPartial := s
	sPartial.GPULayers = 2
	if resPartial := Compute(meta, sPartial, testHW()); SmartLoadMode(resPartial, testHW()) != "mmap" {
		t.Fatalf("partial offload auto load mode must be mmap, got %q", SmartLoadMode(resPartial, testHW()))
	}

	// Guard 3: Model + KV + overhead > 50% available RAM must NEVER pick none/dio, even on full GPU.
	lowRAMHW := testHW()
	lowRAMHW.RAMAvail = 4 << 30 // ~2.4G footprint > 2.0G (50%)
	if resTightRAM := Compute(meta, s, lowRAMHW); SmartLoadMode(resTightRAM, lowRAMHW) != "mmap" {
		t.Fatalf(">50%% RAM auto load mode must be mmap, got %q", SmartLoadMode(resTightRAM, lowRAMHW))
	}

	// Explicit user override of LoadMode is preserved, AND surfaces warning when unsafe.
	big := testMeta(8, 2<<30, 100<<20)
	resolvedBig, resBig := ResolveSettings(s, big, testHW())
	if resolvedBig.LoadMode != "mmap" {
		t.Fatalf("oversized MoE must resolve auto load_mode to mmap, got %q", resolvedBig.LoadMode)
	}
	if !NeedsAutoFit(resBig) {
		t.Fatal("oversized model must enable --fit")
	}

	s.LoadMode = "dio"
	explicit, explicitRes := ResolveSettings(s, big, testHW())
	if explicit.LoadMode != "dio" {
		t.Fatalf("explicit load_mode overwritten: %q", explicit.LoadMode)
	}
	if explicitRes.LoadModeWarning == "" {
		t.Fatal("forcing dio on MoE must populate LoadModeWarning")
	}
	// Even when dense + full-GPU fits cleanly, manual dio must warn AND SmartLoadMode must never pick dio.
	_, denseDioRes := ResolveSettings(s, meta, testHW())
	if denseDioRes.LoadModeWarning == "" {
		t.Fatal("manual dio must always populate LoadModeWarning even on safe dense GPU fits")
	}
	for name, tc := range map[string]Result{
		"dense_fit":   res,
		"moe_fit":     Compute(moeFit, s, testHW()),
		"partial":     Compute(meta, sPartial, testHW()),
		"tight_ram":   Compute(meta, s, lowRAMHW),
		"big_oversiz": resBig,
	} {
		hwCase := testHW()
		if name == "tight_ram" {
			hwCase = lowRAMHW
		}
		if mode := SmartLoadMode(tc, hwCase); mode != "none" && mode != "mmap" {
			t.Fatalf("%s: SmartLoadMode returned %q, only none or mmap allowed", name, mode)
		}
	}
	s.LoadMode = "none"
	_, lowRAMRes := ResolveSettings(s, meta, lowRAMHW)
	if lowRAMRes.LoadModeWarning == "" {
		t.Fatal("forcing none above 50% RAM must populate LoadModeWarning")
	}
}

func TestSmartKVOrderNeedleGateAndTradeoffNote(t *testing.T) {
	// Order: f16/f16 -> q8_0/q8_0 -> q8_0/q4_0 -> q4_0/q4_0; q4_0 disallowed > 16K.
	meta := testMeta(36, 0, 120<<20) // ~4.22G layers + 1G global = ~5.22G weights + 0.5G comp = 5.72G
	hw7G := hw.Info{VRAMTotal: 8 << 30, VRAMFree: 7 << 30, RAMTotal: 16 << 30, RAMAvail: 12 << 30}
	s := flags.Default(0, 8080)

	// Fit f16/f16 easily when VRAM free is large enough (4G weights + 1.1G KV + 0.5G comp <= 7G).
	small := testMeta(32, 0, 80<<20)
	sSmall := s
	sSmall.Context = 8192
	rSmall, resSmall := ResolveSettings(sSmall, small, hw.Info{VRAMTotal: 8 << 30, VRAMFree: 7900 << 20, RAMTotal: 16 << 30, RAMAvail: 12 << 30})
	if rSmall.KCache != "f16" || rSmall.VCache != "f16" || resSmall.KVQualityNote != "" {
		t.Fatalf("expected f16/f16 without quality note, got %s/%s (%q)", rSmall.KCache, rSmall.VCache, resSmall.KVQualityNote)
	}

	// Tight fit at 16K forcing q8_0/q4_0 or q4_0/q4_0 emits q4_0 quality-tradeoff note.
	sTight := s
	sTight.Context = 16384
	r16, res16 := ResolveSettings(sTight, meta, hw.Info{VRAMTotal: 8 << 30, VRAMFree: 8100 << 20, RAMTotal: 16 << 30, RAMAvail: 12 << 30})
	if r16.KCache != "q8_0" || r16.VCache != "q4_0" {
		t.Fatalf("expected q8_0/q4_0 step-down at 16K, got %s/%s (%s)", r16.KCache, r16.VCache, res16.KVReason)
	}
	if res16.KVQualityNote == "" {
		t.Fatal("expected quality tradeoff note when q4_0 is chosen")
	}

	// Above MaxQ4KVContext (32K), q4_0 must NEVER be chosen automatically, and must surface an -nkvo suggestion.
	s32 := s
	s32.Context = 32768
	r32, res32 := ResolveSettings(s32, meta, hw7G)
	if r32.KCache == "q4_0" || r32.VCache == "q4_0" {
		t.Fatalf("q4_0 must be gated above 16K needle limit, got %s/%s", r32.KCache, r32.VCache)
	}
	if res32.KVQualityNote != "" {
		t.Fatalf("unexpected q4_0 note at 32K: %q", res32.KVQualityNote)
	}
	if !strings.Contains(res32.KVReason, "-nkvo") || !strings.Contains(res32.KVReason, "blocked >16K") {
		t.Fatalf("expected >16K blocked/-nkvo guidance in KVReason, got %q", res32.KVReason)
	}
}

func TestMoECacheCapMiBAndModeThreadDefaults(t *testing.T) {
	// Formula: min(vramCap, ramCap) where
	//   vramCap = floor((freeVRAM - estGPU - 1024 MiB) / 256) * 256
	//   ramCap  = floor((availRAM - modelSize - 4 GiB) / 256) * 256
	info := hw.Info{
		VRAMTotal: 8188 << 20, VRAMFree: 7956 << 20,
		RAMTotal: 32 << 30, RAMAvail: 24 << 30,
		PhysCores: 10, PCores: 6, PLogicCores: 12, ECores: 4, LogicCores: 16,
	}
	smallMoE := testMeta(48, 350<<20, 40<<20) // ~1.9 GiB
	smallMoE.ExpertCount = 64
	res := Result{WeightsVRAM: 1900 << 20, KVVRAM: 900 << 20, ComputeVRAM: 512 << 20} // estGPU = 3312M -> vramCap 3584M
	capMiB, reason := MoECacheCapMiB(res, smallMoE, info)
	if capMiB != 3584 || reason == "" {
		t.Fatalf("MoECacheCapMiB = (%d, %q), want 3584", capMiB, reason)
	}
	if !MoECacheAllowed(smallMoE, res, info, 3584) {
		t.Fatal("expected MoECacheAllowed=true when RAMAvail >= model + cache + 4 GiB")
	}

	// RAM-constrained cap beats VRAM cap: 11 GiB avail - 5 GiB model - 4 GiB reserve = 2 GiB (2048 MiB).
	bigMoE := testMeta(48, 0, 0)
	bigMoE.ExpertCount = 64
	bigMoE.TotalBytes = 5 << 30
	infoTightRAM := info
	infoTightRAM.RAMTotal = 16 << 30
	infoTightRAM.RAMAvail = 11 << 30
	ramCap, ramReason := MoECacheCapMiB(res, bigMoE, infoTightRAM)
	if ramCap != 2048 || !strings.Contains(ramReason, "RAM") {
		t.Fatalf("expected RAM-limited cap 2048, got (%d, %q)", ramCap, ramReason)
	}
	if !MoECacheAllowed(bigMoE, res, infoTightRAM, 2048) {
		t.Fatal("expected MoECacheAllowed=true at 2048 MiB")
	}
	if MoECacheAllowed(bigMoE, res, infoTightRAM, 2304) {
		t.Fatal("expected MoECacheAllowed=false when model + cache + 4 GiB > RAMAvail")
	}

	// Model larger than avail RAM - 4 GiB clamps cap to 0 and disallows option.
	hugeMoE := testMeta(48, 0, 0)
	hugeMoE.ExpertCount = 64
	hugeMoE.TotalBytes = 10 << 30
	if z, _ := MoECacheCapMiB(res, hugeMoE, infoTightRAM); z != 0 {
		t.Fatalf("expected 0 cap when availRAM < model + 4G, got %d", z)
	}
	if MoECacheAllowed(hugeMoE, res, infoTightRAM, 0) {
		t.Fatal("expected MoECacheAllowed=false when availRAM < model + 4G")
	}

	// Overflow VRAM clamps at 0.
	resOver := Result{WeightsVRAM: 7000 << 20, KVVRAM: 500 << 20, ComputeVRAM: 512 << 20}
	if z, _ := MoECacheCapMiB(resOver, smallMoE, info); z != 0 {
		t.Fatalf("expected 0 min cap on tight VRAM, got %d", z)
	}

	// Mode thread & priority defaults + NO auto MoE cache:
	dense := testMeta(4, 0, 100<<20)
	// 1. Full GPU: leave -t/-tb/--prio default/omitted.
	rGPU, _ := ResolveSettings(flags.Default(0, 8080), dense, info)
	if rGPU.Threads != 0 || rGPU.Prio != "normal" {
		t.Fatalf("full GPU must leave threads/prio default, got t=%d prio=%q", rGPU.Threads, rGPU.Prio)
	}
	// 2. CPU-only dense: P-core physical threads (6) + --prio 2.
	sCPU := flags.Default(0, 8080)
	sCPU.GPULayers = 0
	rCPU, _ := ResolveSettings(sCPU, dense, info)
	if rCPU.Threads != 6 || rCPU.Prio != "high" {
		t.Fatalf("CPU dense want t=6 prio=high, got t=%d prio=%q", rCPU.Threads, rCPU.Prio)
	}
	// 3. Hybrid MoE: P-core logical threads (12), NEVER auto-enables --moe-cache-mib.
	moe := testMeta(48, 350<<20, 40<<20)
	sMoE := flags.Default(0, 8080)
	sMoE.NCpuMoe = 40
	rMoE, _ := ResolveSettings(sMoE, moe, info)
	if rMoE.Threads != 12 {
		t.Fatalf("hybrid MoE want t=12, got %d", rMoE.Threads)
	}
	if rMoE.MoeCacheMiB != 0 {
		t.Fatalf("Auto path must never set MoeCacheMiB, got %d", rMoE.MoeCacheMiB)
	}
	// Legacy negative auto value also normalizes cleanly to 0 (off).
	sLegacy := sMoE
	sLegacy.MoeCacheMiB = -1
	rLegacy, _ := ResolveSettings(sLegacy, moe, info)
	if rLegacy.MoeCacheMiB != 0 {
		t.Fatalf("legacy MoeCacheMiB=-1 must resolve to 0, got %d", rLegacy.MoeCacheMiB)
	}
}

func TestSmartThreadsTopologyFixtures(t *testing.T) {
	dense := testMeta(32, 0, 100<<20)
	moe := testMeta(48, 350<<20, 40<<20)

	cases := []struct {
		name         string
		info         hw.Info
		wantCPUDense int
		wantMoE      int
	}{
		{
			name:         "6P+4E with HT",
			info:         hw.Info{VRAMTotal: 8 << 30, RAMTotal: 16 << 30, PhysCores: 10, PCores: 6, PLogicCores: 12, ECores: 4, LogicCores: 16},
			wantCPUDense: 6,
			wantMoE:      12,
		},
		{
			name:         "8P+16E",
			info:         hw.Info{VRAMTotal: 8 << 30, RAMTotal: 32 << 30, PhysCores: 24, PCores: 8, PLogicCores: 16, ECores: 16, LogicCores: 32},
			wantCPUDense: 8,
			wantMoE:      16,
		},
		{
			name:         "8-core AMD without SMT",
			info:         hw.Info{VRAMTotal: 8 << 30, RAMTotal: 32 << 30, PhysCores: 8, PCores: 8, PLogicCores: 8, ECores: 0, LogicCores: 8},
			wantCPUDense: 8,
			wantMoE:      8,
		},
		{
			name:         "4-core",
			info:         hw.Info{VRAMTotal: 8 << 30, RAMTotal: 16 << 30, PhysCores: 4, PCores: 4, PLogicCores: 4, ECores: 0, LogicCores: 4},
			wantCPUDense: 4,
			wantMoE:      4,
		},
		{
			name:         "unknown topology falls back to llama.cpp default",
			info:         hw.Info{VRAMTotal: 8 << 30, RAMTotal: 16 << 30, LogicCores: 8},
			wantCPUDense: 0,
			wantMoE:      0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sCPU := flags.Default(0, 8080)
			sCPU.GPULayers = 0
			rCPU, _ := ResolveSettings(sCPU, dense, tc.info)
			if rCPU.Threads != tc.wantCPUDense || rCPU.Prio != "high" {
				t.Fatalf("CPU dense: got (t=%d, prio=%q), want (t=%d, prio=high)", rCPU.Threads, rCPU.Prio, tc.wantCPUDense)
			}

			sMoE := flags.Default(0, 8080)
			sMoE.NCpuMoe = 30
			rMoE, _ := ResolveSettings(sMoE, moe, tc.info)
			if rMoE.Threads != tc.wantMoE || rMoE.Prio != "normal" {
				t.Fatalf("hybrid MoE: got (t=%d, prio=%q), want (t=%d, prio=normal)", rMoE.Threads, rMoE.Prio, tc.wantMoE)
			}
		})
	}
}

func TestEstimatorMultiDeviceSharedIGPUCommitAndUnknownFallback(t *testing.T) {
	meta := testMeta(32, 0, 200<<20) // ~6.4 GiB weights + 1 GiB global

	t.Run("shared iGPU does not double count physical RAM", func(t *testing.T) {
		info := hw.Info{
			GPUName:   "Intel(R) UHD Graphics",
			VRAMTotal: 8 << 30,
			VRAMFree:  8 << 30,
			RAMTotal:  16 << 30,
			RAMAvail:  8 << 30,
			Devices: []hw.GPUDevice{{
				ID:         "Vulkan0",
				Backend:    "vulkan",
				Name:       "Intel(R) UHD Graphics",
				VRAMTotal:  8 << 30,
				VRAMFree:   8 << 30,
				Integrated: true,
			}},
		}
		s := flags.Default(8, 8080)
		s.Context = 8192
		s.GPULayers = flags.AllCount
		res := Compute(meta, s, info)
		if !res.UnifiedMemory {
			t.Fatal("expected UnifiedMemory=true for Intel UHD iGPU")
		}
		if res.Fits() {
			t.Fatalf("shared iGPU 6.4G weights + 1G global + KV in 8G avail RAM must not fit (RAMRatio=%.2f)", res.RAMRatio())
		}
	})

	t.Run("multi-GPU pools VRAM across devices", func(t *testing.T) {
		info := hw.Info{
			RAMTotal: 64 << 30,
			RAMAvail: 48 << 30,
			HasCUDA:  true,
			Devices: []hw.GPUDevice{
				{ID: "CUDA0", Backend: "cuda", Name: "RTX 4060", VRAMTotal: 8 << 30, VRAMFree: 7 << 30},
				{ID: "CUDA1", Backend: "cuda", Name: "RTX 4060", VRAMTotal: 8 << 30, VRAMFree: 7 << 30},
			},
		}
		s := flags.Default(8, 8080)
		s.GPULayers = flags.AllCount
		res := Compute(meta, s, info)
		if res.GPUCount != 2 || res.VRAMTotal != 16<<30 {
			t.Fatalf("pooled GPUs got count=%d total=%d", res.GPUCount, res.VRAMTotal)
		}
		if !res.Fits() {
			t.Fatalf("expected 6.4G model to fit across 2x8G GPUs: vram=%.2f ram=%.2f", res.VRAMRatio(), res.RAMRatio())
		}
	})

	t.Run("Windows commit limit overflow fails Fits", func(t *testing.T) {
		info := testHW()
		info.RAMCommitTotal = 16 << 30
		info.RAMCommitAvail = 2 << 30
		s := flags.Default(8, 8080)
		s.LoadMode = "none"
		s.GPULayers = flags.AllCount
		res := Compute(meta, s, info)
		if res.Fits() {
			t.Fatalf("expected commit overflow (used=%d > avail=%d) to fail Fits()", res.CommitUsed, res.CommitAvail)
		}
	})

	t.Run("unknown GPU falls back cleanly to CPU without zeroing TotalBytes", func(t *testing.T) {
		info := hw.Info{RAMTotal: 32 << 30, RAMAvail: 24 << 30}
		stripped := &gguf.Metadata{BlockCount: 32, GlobalBytes: 512 << 20, TotalBytes: 4 << 30}
		s := flags.Default(8, 8080)
		res := Compute(stripped, s, info)
		if res.WeightsRAM < 3<<30 {
			t.Fatalf("expected stripped fallback weights in RAM, got %d", res.WeightsRAM)
		}
		if res.VRAMRatio() != 0 {
			t.Fatalf("expected 0 VRAMRatio on CPU-only host, got %.2f", res.VRAMRatio())
		}
	})
}

func TestEstimatorNineDeviceProfilesAnd128KOverflowSafe(t *testing.T) {
	smallModel := testMeta(32, 0, 75<<20) // ~2.4G layers + 512M global (~2.9G total)
	bigModel := testMeta(48, 0, 400<<20)  // ~19.2G layers + 1G global

	profiles := []struct {
		name      string
		info      hw.Info
		meta      *gguf.Metadata
		gpuLayers int
		wantFits  bool
		wantUnif  bool
	}{
		{
			name:      "1_4GB_RAM_no_GPU",
			info:      hw.Info{RAMTotal: 4 << 30, RAMAvail: 3 << 30},
			meta:      smallModel,
			gpuLayers: 0,
			wantFits:  false,
		},
		{
			name:      "2_8GB_RAM_4GB_GPU",
			info:      hw.Info{RAMTotal: 8 << 30, RAMAvail: 6 << 30, HasCUDA: true, GPUName: "RTX 3050", VRAMTotal: 4 << 30, VRAMFree: 3500 << 20},
			meta:      smallModel,
			gpuLayers: flags.AllCount,
			wantFits:  false,
		},
		{
			name:      "3_16GB_RAM_8GB_GPU",
			info:      hw.Info{RAMTotal: 16 << 30, RAMAvail: 12 << 30, HasCUDA: true, GPUName: "RTX 4060", VRAMTotal: 8 << 30, VRAMFree: 7 << 30},
			meta:      smallModel,
			gpuLayers: flags.AllCount,
			wantFits:  true,
		},
		{
			name:      "4_32GB_RAM_24GB_GPU",
			info:      hw.Info{RAMTotal: 32 << 30, RAMAvail: 26 << 30, HasCUDA: true, GPUName: "RTX 4090", VRAMTotal: 24 << 30, VRAMFree: 23 << 30},
			meta:      bigModel,
			gpuLayers: flags.AllCount,
			wantFits:  true,
		},
		{
			name: "5_Intel_iGPU_only",
			info: hw.Info{
				RAMTotal: 16 << 30, RAMAvail: 10 << 30, HasVulkan: true, GPUName: "Intel(R) UHD Graphics", VRAMTotal: 8 << 30, VRAMFree: 8 << 30,
				Devices: []hw.GPUDevice{{ID: "Vulkan0", Backend: "vulkan", Name: "Intel(R) UHD Graphics", VRAMTotal: 8 << 30, VRAMFree: 8 << 30, Integrated: true}},
			},
			meta:      smallModel,
			gpuLayers: flags.AllCount,
			wantFits:  true,
			wantUnif:  true,
		},
		{
			name: "6_AMD_Vulkan_discrete",
			info: hw.Info{
				RAMTotal: 16 << 30, RAMAvail: 12 << 30, HasVulkan: true, GPUName: "AMD Radeon RX 7600", VRAMTotal: 8 << 30, VRAMFree: 7500 << 20,
				Devices: []hw.GPUDevice{{ID: "Vulkan0", Backend: "vulkan", Name: "AMD Radeon RX 7600", VRAMTotal: 8 << 30, VRAMFree: 7500 << 20}},
			},
			meta:      smallModel,
			gpuLayers: flags.AllCount,
			wantFits:  true,
		},
		{
			name: "7_Intel_plus_NVIDIA_laptop",
			info: hw.Info{
				RAMTotal: 16 << 30, RAMAvail: 12 << 30, HasCUDA: true, HasVulkan: true, GPUName: "RTX 4060", VRAMTotal: 8 << 30, VRAMFree: 7 << 30,
				Devices: []hw.GPUDevice{
					{ID: "CUDA0", Backend: "cuda", Name: "RTX 4060", VRAMTotal: 8 << 30, VRAMFree: 7 << 30},
					{ID: "Vulkan0", Backend: "vulkan", Name: "Intel(R) UHD Graphics", VRAMTotal: 8 << 30, VRAMFree: 8 << 30, Integrated: true},
				},
			},
			meta:      smallModel,
			gpuLayers: flags.AllCount,
			wantFits:  true,
			wantUnif:  false,
		},
		{
			name: "8_two_NVIDIA_GPUs",
			info: hw.Info{
				RAMTotal: 64 << 30, RAMAvail: 48 << 30, HasCUDA: true, GPUName: "RTX 3090", VRAMTotal: 24 << 30, VRAMFree: 23 << 30,
				Devices: []hw.GPUDevice{
					{ID: "CUDA0", Backend: "cuda", Name: "RTX 3090", VRAMTotal: 24 << 30, VRAMFree: 23 << 30},
					{ID: "CUDA1", Backend: "cuda", Name: "RTX 3090", VRAMTotal: 24 << 30, VRAMFree: 23 << 30},
				},
			},
			meta:      bigModel,
			gpuLayers: flags.AllCount,
			wantFits:  true,
		},
		{
			name:      "9_unknown_device_CPU_fallback",
			info:      hw.Info{RAMTotal: 16 << 30, RAMAvail: 12 << 30, GPUName: "Unknown Compute Accelerator"},
			meta:      smallModel,
			gpuLayers: 0,
			wantFits:  true,
		},
	}

	for _, tc := range profiles {
		t.Run(tc.name, func(t *testing.T) {
			s := flags.Default(8, 8080)
			s.Context = 4096
			s.GPULayers = tc.gpuLayers
			res := Compute(tc.meta, s, tc.info)
			if res.UnifiedMemory != tc.wantUnif {
				t.Fatalf("UnifiedMemory = %v, want %v", res.UnifiedMemory, tc.wantUnif)
			}
			if res.Fits() != tc.wantFits {
				t.Fatalf("Fits() = %v, want %v (VRAM %.2f, RAM %.2f, Commit %.2f)",
					res.Fits(), tc.wantFits, res.VRAMRatio(), res.RAMRatio(), res.CommitRatio())
			}
		})
	}

	// Verify 64-bit arithmetic at 128K context on a large model never overflows.
	s128k := flags.Default(8, 8080)
	s128k.Context = 131072
	s128k.GPULayers = 0
	s128k.NoKVOffload = true
	res128k := Compute(bigModel, s128k, profiles[7].info)
	if res128k.KVRAM == 0 || res128k.TotalFootprint() <= res128k.KVRAM {
		t.Fatalf("expected positive non-overflowing 128K estimate, got total=%d kv=%d", res128k.TotalFootprint(), res128k.KVRAM)
	}
}
