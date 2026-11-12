//go:build windows

package hw

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestParseSMILine(t *testing.T) {
	name, total, free, ok := parseSMI("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7000\n")
	if !ok {
		t.Fatal("line not parsed")
	}
	if name != "NVIDIA GeForce RTX 4060 Laptop GPU" {
		t.Errorf("name = %q", name)
	}
	if total != 8188*1024*1024 || free != 7000*1024*1024 {
		t.Errorf("total/free = %d/%d", total, free)
	}
}

func TestParseSMIRejectsJunk(t *testing.T) {
	if _, _, _, ok := parseSMI(""); ok {
		t.Fatal("empty output accepted")
	}
	if _, _, _, ok := parseSMI("GPU only"); ok {
		t.Fatal("incomplete line accepted")
	}
	if got := mibToBytes("[N/A]"); got != 0 {
		t.Errorf("N/A = %d", got)
	}
	if got := mibToBytes("8192 MiB"); got != 8192*1024*1024 {
		t.Errorf("suffixed value = %d", got)
	}
}

func TestMemoryStatusAndProcessRAM(t *testing.T) {
	total, avail, err := MemoryStatus()
	if err != nil || total == 0 || avail > total {
		t.Fatalf("MemoryStatus = (%d, %d, %v)", total, avail, err)
	}
	ws, priv, err := ProcessRAM(uint32(os.Getpid()))
	if err != nil || ws == 0 || priv == 0 {
		t.Fatalf("ProcessRAM = (%d, %d, %v)", ws, priv, err)
	}
}

func TestDetectFastTopologyAndDisk(t *testing.T) {
	t0 := time.Now()
	info := DetectFast()
	dur := time.Since(t0)
	t.Logf("DetectFast (%v): CPU=%q Phys=%d P=%d E=%d Logic=%d RAM=%s",
		dur, info.CPUName, info.PhysCores, info.PCores, info.ECores, info.LogicCores, GB(info.RAMTotal))
	if dur > 250*time.Millisecond {
		t.Fatalf("DetectFast too slow: %v (expected <250ms)", dur)
	}
	if info.PhysCores <= 0 || info.PCores <= 0 || info.PCores+info.ECores != info.PhysCores {
		t.Fatalf("invalid core topology: %+v", info)
	}
	if info.LogicCores < info.PhysCores || info.RAMTotal == 0 {
		t.Fatalf("invalid DetectFast info: %+v", info)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	kind, root := DriveKind(wd)
	if kind == "unknown" || root == "" {
		t.Fatalf("DriveKind(%q) = (%q, %q)", wd, kind, root)
	}
	if avail, total, err := DiskFreeBytes(filepath.VolumeName(wd) + `\`); err != nil || avail == 0 || total < avail {
		t.Fatalf("DiskFreeBytes = (%d, %d, %v)", avail, total, err)
	}
	if info.PLogicCores < info.PCores {
		t.Fatalf("invalid PLogicCores: %+v", info)
	}
}

func makeProcCoreRecord(eff byte, mask uint64) []byte {
	rec := make([]byte, 48)
	binary.LittleEndian.PutUint32(rec[0:4], relationProcessorCore)
	binary.LittleEndian.PutUint32(rec[4:8], 48)
	rec[9] = eff
	binary.LittleEndian.PutUint16(rec[30:32], 1)
	binary.LittleEndian.PutUint64(rec[32:40], mask)
	return rec
}

func TestParseLogicalProcessorInfoFixtures(t *testing.T) {
	build := func(pCount int, pMask uint64, pEff byte, eCount int, eMask uint64) []byte {
		var buf []byte
		for i := 0; i < pCount; i++ {
			buf = append(buf, makeProcCoreRecord(pEff, pMask)...)
		}
		for i := 0; i < eCount; i++ {
			buf = append(buf, makeProcCoreRecord(0, eMask)...)
		}
		return buf
	}

	cases := []struct {
		name                           string
		buf                            []byte
		wantPhys, wantP, wantPL, wantE int
		wantLogic                      int
		wantOK                         bool
	}{
		{
			name:      "6P+4E with HT",
			buf:       build(6, 0b11, 1, 4, 0b1),
			wantPhys:  10,
			wantP:     6,
			wantPL:    12,
			wantE:     4,
			wantLogic: 16,
			wantOK:    true,
		},
		{
			name:      "8P+16E",
			buf:       build(8, 0b11, 1, 16, 0b1),
			wantPhys:  24,
			wantP:     8,
			wantPL:    16,
			wantE:     16,
			wantLogic: 32,
			wantOK:    true,
		},
		{
			name:      "8-core AMD without SMT",
			buf:       build(8, 0b1, 0, 0, 0),
			wantPhys:  8,
			wantP:     8,
			wantPL:    8,
			wantE:     0,
			wantLogic: 8,
			wantOK:    true,
		},
		{
			name:      "4-core",
			buf:       build(4, 0b1, 0, 0, 0),
			wantPhys:  4,
			wantP:     4,
			wantPL:    4,
			wantE:     0,
			wantLogic: 4,
			wantOK:    true,
		},
		{
			name:   "unknown empty buffer",
			buf:    nil,
			wantOK: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			phys, p, pl, e, logic, ok := parseLogicalProcessorInfo(tc.buf)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if !ok {
				return
			}
			if phys != tc.wantPhys || p != tc.wantP || pl != tc.wantPL || e != tc.wantE || logic != tc.wantLogic {
				t.Fatalf("got (phys=%d, P=%d, PL=%d, E=%d, L=%d), want (%d, %d, %d, %d, %d)",
					phys, p, pl, e, logic, tc.wantPhys, tc.wantP, tc.wantPL, tc.wantE, tc.wantLogic)
			}
		})
	}
}

func TestParseListDevicesAndSMICapability(t *testing.T) {
	raw := `Available devices:
  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB, 7106 MiB free)
  Vulkan0: Intel(R) UHD Graphics (15920 MiB, 15920 MiB free)
`
	devs := ParseListDevices(raw)
	if len(devs) != 2 {
		t.Fatalf("got %d devices, want 2: %+v", len(devs), devs)
	}
	if devs[0].Backend != "cuda" || devs[0].VRAMTotal != 8187<<20 || devs[0].VRAMFree != 7106<<20 || devs[0].Integrated {
		t.Fatalf("unexpected CUDA device: %+v", devs[0])
	}
	if devs[1].Backend != "vulkan" || !devs[1].Integrated || devs[1].VRAMTotal != 15920<<20 {
		t.Fatalf("unexpected Vulkan iGPU device: %+v", devs[1])
	}

	// Pascal GTX 1080 (SM 6.1) must be flagged unsupported for CUDA builds.
	smiOld, reason := parseSMIDevices("NVIDIA GeForce GTX 1080, 8192, 7800, 6.1\n")
	if len(smiOld) != 0 || reason == "" {
		t.Fatalf("expected SM 6.1 rejection, got devs=%+v reason=%q", smiOld, reason)
	}

	// Dual RTX 4090 must aggregate cleanly.
	smiMulti, reasonMulti := parseSMIDevices("NVIDIA GeForce RTX 4090, 24564, 24000, 8.9\nNVIDIA GeForce RTX 4090, 24564, 24100, 8.9\n")
	if reasonMulti != "" || len(smiMulti) != 2 {
		t.Fatalf("expected 2 SM 8.9 devices, got devs=%+v reason=%q", smiMulti, reasonMulti)
	}
	if _, _, availCommit, err := func() (uint64, uint64, uint64, error) {
		ct, ca, e := CommitStatus()
		return 0, ct, ca, e
	}(); err != nil || availCommit == 0 {
		t.Logf("CommitStatus returned availCommit=%d err=%v", availCommit, err)
	}
}

func TestVolumeFilesystemAndDirWritable(t *testing.T) {
	dir := t.TempDir()
	fs := VolumeFilesystem(dir)
	if fs == "" {
		t.Fatalf("expected non-empty filesystem name for %s", dir)
	}
	if !DirWritable(dir) {
		t.Fatalf("expected %s to be writable", dir)
	}
	missing := dir + `\does-not-exist-subdir`
	if DirWritable(missing) {
		t.Fatalf("expected missing dir %s to report not writable", missing)
	}

	// Mixed CUDA + integrated Vulkan must NOT report unified memory overall.
	mixed := Info{
		HasCUDA: true,
		Devices: []GPUDevice{
			{ID: "CUDA0", Backend: "cuda", Name: "NVIDIA RTX 4060", Integrated: false},
			{ID: "Vulkan0", Backend: "vulkan", Name: "Intel(R) UHD Graphics", Integrated: true},
		},
	}
	if mixed.HasUnifiedMemory() {
		t.Fatal("mixed discrete CUDA + iGPU must not set HasUnifiedMemory")
	}
	igpuOnly := Info{
		Devices: []GPUDevice{
			{ID: "Vulkan0", Backend: "vulkan", Name: "Intel(R) UHD Graphics", Integrated: true},
		},
	}
	if !igpuOnly.HasUnifiedMemory() {
		t.Fatal("iGPU-only device must set HasUnifiedMemory")
	}
}
