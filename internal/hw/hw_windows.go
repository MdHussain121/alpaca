//go:build windows

// Package hw detects the GPU, RAM, CPU topology, and drive characteristics
// available to the launcher using native Win32 APIs.
package hw

import (
	"context"
	"math/bits"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/cpu"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

// Runner executes a command and returns its combined output. It is injectable
// so detection can be tested without touching the real machine.
type Runner func(ctx context.Context, name string, args ...string) (string, error)

// GPUDevice describes one compute device reported by nvidia-smi or llama-server --list-devices.
type GPUDevice struct {
	ID         string
	Backend    string // "cuda", "vulkan", "cpu"
	Name       string
	VRAMTotal  uint64 // bytes
	VRAMFree   uint64 // bytes
	ComputeCap string // CUDA SM capability e.g. "8.9"
	Integrated bool   // shared system RAM (iGPU/APU)
}

// Info is the detected hardware.
type Info struct {
	GPUName               string
	VRAMTotal             uint64 // bytes
	VRAMFree              uint64 // bytes
	HasCUDA               bool
	HasVulkan             bool
	HasAVX2               bool
	CUDAUnsupportedReason string
	Devices               []GPUDevice
	RAMTotal              uint64 // physical RAM bytes
	RAMAvail              uint64 // available physical RAM bytes
	RAMCommitTotal        uint64 // virtual commit limit bytes
	RAMCommitAvail        uint64 // available virtual commit bytes
	CPUName               string
	PhysCores             int
	PCores                int
	PLogicCores           int
	ECores                int
	LogicCores            int
	Err                   error
}

// HasUnifiedMemory reports whether the active GPU shares system RAM with the CPU.
func (i Info) HasUnifiedMemory() bool {
	if len(i.Devices) > 0 {
		allIntegrated := true
		for _, d := range i.Devices {
			if !d.Integrated {
				allIntegrated = false
				break
			}
		}
		if allIntegrated {
			return true
		}
	}
	return IsIntegratedGPUName(i.GPUName) && !i.HasCUDA
}

// IsIntegratedGPUName detects integrated graphics adapters that draw from system RAM.
func IsIntegratedGPUName(name string) bool {
	l := strings.ToLower(strings.TrimSpace(name))
	if l == "" {
		return false
	}
	for _, pat := range []string{
		"intel(r) uhd", "intel(r) iris", "intel(r) hd graphics", "intel(r) arc(tm) graphics",
		"radeon(tm) graphics", "amd radeon graphics", "vega", "microsoft basic render",
	} {
		if strings.Contains(l, pat) {
			return true
		}
	}
	return false
}

// HasAVX2 reports whether the host CPU supports AVX2 + FMA instructions.
func HasAVX2() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

// ExecRunner is the default Runner.
func ExecRunner(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	out, err := cmd.Output()
	return string(out), err
}

// DetectFast probes CPU topology, CPU name, ISA flags, Vulkan runtime DLL,
// and physical/commit RAM using pure Win32 syscalls (<5 ms, zero subprocesses).
func DetectFast() Info {
	info := Info{
		LogicCores: runtime.NumCPU(),
		CPUName:    cpuBrandName(),
		HasAVX2:    HasAVX2(),
		HasVulkan:  hasSystemVulkanDLL(),
	}

	if total, avail, cTotal, cAvail, err := memoryStatusFull(); err == nil {
		info.RAMTotal = total
		info.RAMAvail = avail
		info.RAMCommitTotal = cTotal
		info.RAMCommitAvail = cAvail
	} else {
		info.Err = err
	}

	if phys, pCores, pLogic, eCores, logic, ok := cpuTopology(); ok {
		info.PhysCores = phys
		info.PCores = pCores
		info.PLogicCores = pLogic
		info.ECores = eCores
		if logic > 0 {
			info.LogicCores = logic
		}
	} else if runtime.NumCPU() > 1 {
		// Topology API failed: keep PCores/PLogicCores at 0 so caller knows
		// P-core topology is unknown and falls back to llama.cpp's default.
		info.PhysCores = runtime.NumCPU() / 2
	} else {
		info.PhysCores = 1
	}
	return info
}

func hasSystemVulkanDLL() bool {
	sysDir, err := windows.GetSystemDirectory()
	if err != nil || sysDir == "" {
		return false
	}
	_, err = os.Stat(filepath.Join(sysDir, "vulkan-1.dll"))
	return err == nil
}

// HasNVCUDADriver checks whether nvcuda.dll is loadable on this machine.
func HasNVCUDADriver() bool {
	return windows.NewLazySystemDLL("nvcuda.dll").Load() == nil
}

// DetectGPU probes the NVIDIA GPU once via nvidia-smi without blocking Win32 queries.
func DetectGPU(run Runner) (name string, total, free uint64, ok bool) {
	devs, reason := DetectGPUDevices(run)
	if reason != "" || len(devs) == 0 {
		return "", 0, 0, false
	}
	var sumTotal, sumFree uint64
	for _, d := range devs {
		sumTotal += d.VRAMTotal
		sumFree += d.VRAMFree
	}
	name = devs[0].Name
	if len(devs) > 1 {
		name = strconv.Itoa(len(devs)) + "x " + name
	}
	return name, sumTotal, sumFree, sumTotal > 0
}

// DetectGPUDevices queries all NVIDIA CUDA GPUs via nvidia-smi and validates
// minimum SM 7.5+ compute capability required by packaged CUDA builds.
func DetectGPUDevices(run Runner) ([]GPUDevice, string) {
	if run == nil {
		run = ExecRunner
		if !HasNVCUDADriver() {
			return nil, "nvcuda.dll not found"
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	out, err := run(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,memory.free,compute_cap",
		"--format=csv,noheader,nounits")
	if err != nil {
		// Legacy driver fallback without compute_cap column.
		out, err = run(ctx, "nvidia-smi",
			"--query-gpu=name,memory.total,memory.free",
			"--format=csv,noheader,nounits")
		if err != nil {
			return nil, ""
		}
	}
	return parseSMIDevices(out)
}

// Detect probes CPU, RAM, and GPU hardware. It never panics; failures land in Info.Err.
func Detect(run Runner) Info {
	info := DetectFast()
	if devs, reason := DetectGPUDevices(run); len(devs) > 0 && reason == "" {
		info.Devices = devs
		var sumTotal, sumFree uint64
		for _, d := range devs {
			sumTotal += d.VRAMTotal
			sumFree += d.VRAMFree
		}
		info.GPUName = devs[0].Name
		if len(devs) > 1 {
			info.GPUName = strconv.Itoa(len(devs)) + "x " + info.GPUName
		}
		info.VRAMTotal = sumTotal
		info.VRAMFree = sumFree
		info.HasCUDA = true
	} else if reason != "" {
		info.CUDAUnsupportedReason = reason
	}
	return info
}

func parseSMI(out string) (name string, total, free uint64, ok bool) {
	devs, reason := parseSMIDevices(out)
	if reason != "" || len(devs) == 0 {
		return "", 0, 0, false
	}
	return devs[0].Name, devs[0].VRAMTotal, devs[0].VRAMFree, true
}

func parseSMIDevices(out string) ([]GPUDevice, string) {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	var devs []GPUDevice
	var unsupported string
	for idx, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		name := strings.TrimSpace(parts[0])
		total := mibToBytes(parts[1])
		free := mibToBytes(parts[2])
		if total == 0 {
			continue
		}
		cc := ""
		if len(parts) >= 4 {
			cc = strings.TrimSpace(parts[3])
			if major, minor, ok := parseComputeCap(cc); ok && (major < 7 || (major == 7 && minor < 5)) {
				unsupported = name + " (SM " + cc + " < 7.5 minimum)"
				continue
			}
		}
		devs = append(devs, GPUDevice{
			ID:         "CUDA" + strconv.Itoa(idx),
			Backend:    "cuda",
			Name:       name,
			VRAMTotal:  total,
			VRAMFree:   free,
			ComputeCap: cc,
			Integrated: IsIntegratedGPUName(name),
		})
	}
	if len(devs) > 0 {
		return devs, ""
	}
	return nil, unsupported
}

func parseComputeCap(s string) (major, minor int, ok bool) {
	parts := strings.Split(strings.TrimSpace(s), ".")
	if len(parts) != 2 {
		return 0, 0, false
	}
	mj, err1 := strconv.Atoi(parts[0])
	mn, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return mj, mn, true
}

// ParseListDevices parses `llama-server --list-devices` output lines such as:
//
//	CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB, 7106 MiB free)
func ParseListDevices(out string) []GPUDevice {
	var devs []GPUDevice
	for _, raw := range strings.Split(out, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "Available") || line == "(none)" {
			continue
		}
		colon := strings.IndexByte(line, ':')
		if colon <= 0 {
			continue
		}
		id := strings.TrimSpace(line[:colon])
		rest := strings.TrimSpace(line[colon+1:])
		openParen := strings.LastIndexByte(rest, '(')
		closeParen := strings.LastIndexByte(rest, ')')
		if openParen <= 0 || closeParen <= openParen {
			continue
		}
		name := strings.TrimSpace(rest[:openParen])
		memSpec := rest[openParen+1 : closeParen]
		var total, free uint64
		for _, seg := range strings.Split(memSpec, ",") {
			seg = strings.TrimSpace(seg)
			if strings.Contains(strings.ToLower(seg), "free") {
				free = mibToBytes(strings.TrimSpace(strings.TrimSuffix(strings.ToLower(seg), "free")))
			} else if total == 0 {
				total = mibToBytes(seg)
			}
		}
		if free == 0 {
			free = total
		}
		backend := "cpu"
		lowerID := strings.ToLower(id)
		switch {
		case strings.HasPrefix(lowerID, "cuda"):
			backend = "cuda"
		case strings.HasPrefix(lowerID, "vulkan"):
			backend = "vulkan"
		case strings.HasPrefix(lowerID, "rocm"):
			backend = "rocm"
		case strings.HasPrefix(lowerID, "metal"):
			backend = "metal"
		}
		if total > 0 {
			devs = append(devs, GPUDevice{
				ID:         id,
				Backend:    backend,
				Name:       name,
				VRAMTotal:  total,
				VRAMFree:   free,
				Integrated: backend == "vulkan" && IsIntegratedGPUName(name),
			})
		}
	}
	return devs
}

// ProbeBackendDevices invokes `exePath --list-devices` and returns discovered devices.
func ProbeBackendDevices(ctx context.Context, exePath string) ([]GPUDevice, error) {
	if strings.TrimSpace(exePath) == "" {
		return nil, os.ErrNotExist
	}
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
	}
	cmd := exec.CommandContext(ctx, exePath, "--list-devices")
	cmd.Dir = filepath.Dir(exePath)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}
	return ParseListDevices(string(out)), nil
}

func mibToBytes(s string) uint64 {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "[N/A]") {
		return 0
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		// Some drivers report "8192 MiB".
		f := strings.Fields(s)
		if len(f) > 0 {
			if v2, err2 := strconv.ParseUint(f[0], 10, 64); err2 == nil {
				return v2 * 1024 * 1024
			}
		}
		return 0
	}
	return v * 1024 * 1024
}

func cpuBrandName() string {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE,
		`HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE)
	if err != nil {
		return ""
	}
	defer k.Close()
	name, _, err := k.GetStringValue("ProcessorNameString")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(name)
}

var (
	kernel32                             = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalMemoryStatusEx             = kernel32.NewProc("GlobalMemoryStatusEx")
	procGetProcessMemoryInfo             = kernel32.NewProc("K32GetProcessMemoryInfo")
	procEmptyWorkingSet                  = kernel32.NewProc("K32EmptyWorkingSet")
	procGetLogicalProcessorInformationEx = kernel32.NewProc("GetLogicalProcessorInformationEx")
)

const relationProcessorCore = 0

// coreRecord captures the Windows EFFICIENCY_CLASS and logical-thread mask per core.
type coreRecord struct {
	effClass byte
	logical  int
}

func cpuTopology() (phys, pCores, pLogic, eCores, logic int, ok bool) {
	var bufLen uint32
	_, _, callErr := procGetLogicalProcessorInformationEx.Call(
		relationProcessorCore,
		0,
		uintptr(unsafeUint32Ptr(&bufLen)),
	)
	if bufLen == 0 {
		_ = callErr
		return 0, 0, 0, 0, 0, false
	}
	buf := make([]byte, bufLen)
	r, _, _ := procGetLogicalProcessorInformationEx.Call(
		relationProcessorCore,
		uintptr(unsafeByteSlicePtr(buf)),
		uintptr(unsafeUint32Ptr(&bufLen)),
	)
	if r == 0 || int(bufLen) > len(buf) {
		return 0, 0, 0, 0, 0, false
	}
	return parseLogicalProcessorInfo(buf[:bufLen])
}

// parseLogicalProcessorInfo decodes Win32 SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
// RelationProcessorCore records on 64-bit Windows.
func parseLogicalProcessorInfo(buf []byte) (phys, pCores, pLogic, eCores, logic int, ok bool) {
	var cores []coreRecord
	var maxEff byte
	off := 0
	for off+48 <= len(buf) {
		rel := readLE32(buf[off : off+4])
		size := int(readLE32(buf[off+4 : off+8]))
		if size < 48 || off+size > len(buf) {
			break
		}
		if rel == relationProcessorCore {
			eff := buf[off+9]
			if eff > maxEff {
				maxEff = eff
			}
			groupCount := int(readLE16(buf[off+30 : off+32]))
			logical := 0
			for g := 0; g < groupCount; g++ {
				gOff := off + 32 + g*16
				if gOff+8 > off+size {
					break
				}
				mask := readLE64(buf[gOff : gOff+8])
				logical += bits.OnesCount64(mask)
			}
			if logical == 0 {
				logical = 1
			}
			cores = append(cores, coreRecord{effClass: eff, logical: logical})
			logic += logical
		}
		off += size
	}
	if len(cores) == 0 {
		return 0, 0, 0, 0, 0, false
	}
	phys = len(cores)
	for _, c := range cores {
		if c.effClass == maxEff {
			pCores++
			pLogic += c.logical
		} else {
			eCores++
		}
	}
	return phys, pCores, pLogic, eCores, logic, true
}

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

type processMemoryCountersEx struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
	privateUsage               uintptr
}

func memoryStatusFull() (totalPhys, availPhys, totalCommit, availCommit uint64, err error) {
	var m memoryStatusEx
	m.Length = unsafeSizeofMemoryStatusEx
	r, _, callErr := procGlobalMemoryStatusEx.Call(uintptr(unsafePointer(&m)))
	if r == 0 {
		return 0, 0, 0, 0, callErr
	}
	return m.TotalPhys, m.AvailPhys, m.TotalPageFile, m.AvailPageFile, nil
}

func memoryStatus() (total, avail uint64, err error) {
	t, a, _, _, err := memoryStatusFull()
	return t, a, err
}

// MemoryStatus returns live system physical RAM (total, available) in bytes.
func MemoryStatus() (total, avail uint64, err error) {
	return memoryStatus()
}

// CommitStatus returns live Windows virtual commit limit (total, available) in bytes.
func CommitStatus() (total, avail uint64, err error) {
	_, _, ct, ca, err := memoryStatusFull()
	return ct, ca, err
}

// ProcessRAM returns the working-set and private-commit memory in bytes of pid.
func ProcessRAM(pid uint32) (workingSet, private uint64, err error) {
	if pid == 0 {
		return 0, 0, nil
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
	if err != nil {
		return 0, 0, err
	}
	defer windows.CloseHandle(h)

	var mc processMemoryCountersEx
	mc.cb = unsafeSizeofProcessMemCounters
	r, _, callErr := procGetProcessMemoryInfo.Call(uintptr(h), uintptr(unsafeProcMemPtr(&mc)), uintptr(mc.cb))
	if r == 0 {
		return 0, 0, callErr
	}
	return uint64(mc.workingSetSize), uint64(mc.privateUsage), nil
}

// TrimWorkingSet moves a live process's resident pages to the standby list.
// After a fully GPU-offloaded mmap load the GGUF's file pages are no longer
// needed on the host, so this keeps the reported RAM close to the real private
// host buffers instead of the whole model file.
func TrimWorkingSet(pid uint32) error {
	if pid == 0 {
		return nil
	}
	const access = windows.PROCESS_QUERY_INFORMATION | windows.PROCESS_SET_QUOTA
	h, err := windows.OpenProcess(access, false, pid)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	if r, _, callErr := procEmptyWorkingSet.Call(uintptr(h)); r == 0 {
		return callErr
	}
	return nil
}

// VRAMStatus queries current GPU VRAM total and free bytes via nvidia-smi.
func VRAMStatus(run Runner) (total, free uint64, ok bool) {
	_, total, free, ok = DetectGPU(run)
	return total, free, ok
}

// DriveKind returns "removable", "network", "fixed", or "unknown" plus the
// normalized volume root (e.g. `D:\`) for path.
func DriveKind(path string) (kind string, root string) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "unknown", ""
	}
	vol := filepath.VolumeName(abs)
	if vol == "" {
		return "unknown", ""
	}
	root = vol + `\`
	ptr, err := windows.UTF16PtrFromString(root)
	if err != nil {
		return "unknown", root
	}
	switch windows.GetDriveType(ptr) {
	case windows.DRIVE_REMOVABLE:
		return "removable", root
	case windows.DRIVE_REMOTE:
		return "network", root
	case windows.DRIVE_FIXED, windows.DRIVE_RAMDISK:
		return "fixed", root
	default:
		return "unknown", root
	}
}

// EstimatedReadMBps returns a realistic cold sequential read rate in MiB/s for
// the storage backing path so load ETAs reflect USB vs NVMe reality.
func EstimatedReadMBps(path string) float64 {
	kind, _ := DriveKind(path)
	switch kind {
	case "removable":
		return 24.0
	case "network":
		return 50.0
	case "fixed":
		return 1500.0
	default:
		return 200.0
	}
}

// DiskFreeBytes returns available free bytes for the caller and total volume bytes.
func DiskFreeBytes(path string) (avail, total uint64, err error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return 0, 0, err
	}
	ptr, err := windows.UTF16PtrFromString(abs)
	if err != nil {
		return 0, 0, err
	}
	var freeAvail, totalBytes, totalFree uint64
	if err := windows.GetDiskFreeSpaceEx(ptr, &freeAvail, &totalBytes, &totalFree); err != nil {
		return 0, 0, err
	}
	return freeAvail, totalBytes, nil
}

// FAT32MaxFileBytes is the largest single file size supported on a FAT32 volume.
const FAT32MaxFileBytes int64 = 4*1024*1024*1024 - 1

// VolumeFilesystem returns the filesystem name (e.g. "NTFS", "exFAT", "FAT32")
// for the volume hosting path, or "" if it cannot be determined.
func VolumeFilesystem(path string) string {
	_, root := DriveKind(path)
	if root == "" {
		return ""
	}
	rootPtr, err := windows.UTF16PtrFromString(root)
	if err != nil {
		return ""
	}
	fsBuf := make([]uint16, 64)
	var serial, maxCompLen, flags uint32
	if err := windows.GetVolumeInformation(rootPtr, nil, 0, &serial, &maxCompLen, &flags, &fsBuf[0], uint32(len(fsBuf))); err != nil {
		return ""
	}
	return windows.UTF16ToString(fsBuf)
}

// DirWritable reports whether new files can be created inside dir without leaving artifacts.
func DirWritable(dir string) bool {
	if strings.TrimSpace(dir) == "" {
		return false
	}
	f, err := os.CreateTemp(dir, ".alpaca-writecheck-*")
	if err != nil {
		return false
	}
	name := f.Name()
	_ = f.Close()
	_ = os.Remove(name)
	return true
}

// GB renders a byte count in GiB with one decimal.
func GB(b uint64) string {
	return strconv.FormatFloat(float64(b)/(1<<30), 'f', 1, 64)
}
