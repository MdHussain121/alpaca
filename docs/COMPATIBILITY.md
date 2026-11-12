# alpaca Compatibility Matrix

This document records verified hardware, backend, terminal, filesystem, and estimator behavior for **alpaca v1.0.0** (pinned `llama.cpp` `b12284`, Windows 10/11 x64).

## Clean-Machine Dependencies

All shipped executables are built without external C/C++ runtime DLLs:

| Binary | Build Runtime | Verified PE Dependencies (`dumpbin /dependents`) |
| --- | --- | --- |
| `alpaca.exe` | Go (`CGO_ENABLED=0`) | `KERNEL32.dll` |
| `bin/cpu/llama-server.exe` | MSVC `/MT` static CRT (`GGML_OPENMP=OFF`, AVX2+FMA) | `KERNEL32.dll`, `ADVAPI32.dll`, `WS2_32.dll` |
| `bin/cpu-compat/llama-server.exe` | MSVC `/MT` static CRT (`GGML_OPENMP=OFF`, SSE4.2) | `KERNEL32.dll`, `ADVAPI32.dll`, `WS2_32.dll` |
| `bin/cuda/llama-server.exe` | MSVC `/MT` static CRT, CUDA 13.2 (`75;80;86;89;120a-virtual`) | `KERNEL32.dll`, `ADVAPI32.dll`, `WS2_32.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` |

`bin/cuda` requires an NVIDIA driver providing `nvcuda.dll` and compute capability **>= 7.5** (Turing RTX 20-series / T4 or newer; Ada/Blackwell supported natively and via SM 12.0a PTX). Older GPUs (< 7.5) or missing drivers automatically fall back to Vulkan (if present) or CPU (`avx2` -> `cpu-compat` SSE4.2) with a one-line explanation.

---

## Model x Backend Test Matrix

Host hardware used for live measurements: Intel Hybrid 10-core CPU, 15.71 GiB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB VRAM, SM 8.9) + Intel UHD Graphics iGPU, exFAT USB flash drive + NTFS SSD, context `4096`, `q4_0` KV cache.

| Model | File Size | CUDA (`bin/cuda`) | CPU AVX2 (`bin/cpu`) | CPU SSE4.2 (`bin/cpu-compat`) | Vulkan (`bin/vulkan`) |
| --- | --- | --- | --- | --- | --- |
| `Qwen3.5-4B-3D-Coder-Q4_K_M.gguf` | 2.52 GiB | **RUNS** · ready 5.0s · **68.5 tok/s** | **RUNS-SLOW** · ready 5.0s · **9.2 tok/s** | **RUNS-SLOW** · ready 5.1s · **2.2 tok/s** | **BLOCKED-ENVIRONMENT** (Vulkan SDK not installed on build host; runtime `vulkan-1.dll` detection + fallback implemented) |
| `Spark-X2.6-4B-abliterated-Q4_K_M.gguf` | 2.42 GiB | **RUNS** · ready 4.0s · **73.0 tok/s** | **RUNS-SLOW** · ready 5.0s · **10.9 tok/s** | Compatible via SSE4.2 fallback | **BLOCKED-ENVIRONMENT** |
| `LFM2.5-8B-A1B-Q4_0.gguf` (MoE) | 4.51 GiB | **RUNS** · ready 7.0s · **158.4 tok/s** | **RUNS** · ready 8.0s · **24.6 tok/s** | Compatible via SSE4.2 fallback | **BLOCKED-ENVIRONMENT** |
| `Tiel-Coder-35B-A3B-UD-IQ4_XS.gguf` (MoE) | 16.51 GiB | **RUNS-SLOW** (hybrid spillover without `-ncmoe`: ready 20.0s · **8.6 tok/s**; warning raised when un-tuned `ngl=all` exceeds 8 GiB VRAM) | **BLOCKED-WITH-CLEAR-MESSAGE** (estimator flags 17.02 / 9.13 GiB = 186% RAM) | **BLOCKED-WITH-CLEAR-MESSAGE** (estimator flags over RAM budget) | **BLOCKED-ENVIRONMENT** |

---

## Environment Test Matrix

| Environment | Status | Details / Substitute |
| --- | --- | --- |
| Local Windows 11 Pro x64, CUDA | **PASS** | `nvcuda.dll` + SM >= 7.5 check + `llama-server --list-devices` verified live. |
| Local Windows 11 Pro x64, CPU-only (`CUDA_VISIBLE_DEVICES=-1`) | **PASS** | AVX2 + SSE4.2 `cpu-compat` verified live. |
| Clean Windows Sandbox / Hyper-V VM (4 / 8 / 16 GiB RAM, no VC++ CRT) | **BLOCKED** | `WindowsSandbox.exe` / Hyper-V not enabled on host. Verified instead via `dumpbin /dependents` proving zero `VCRUNTIME140`/`MSVCP140`/`VCOMP140` DLL imports and via 9-profile fixture estimator tests (`4 GiB` .. `32 GiB`, iGPU, dual-GPU, unknown GPU). |
| Alternate drive letter (`subst Z:`) + spaces & Unicode path | **PASS** | Launched `alpaca.exe --version` from `Z:\` and `cpu/llama-server.exe` from folder containing spaces and non-ASCII (`café`). |
| Write-protected / read-only folder | **DEGRADES-GRACEFULLY** | `DirWritable` precheck detects read-only destinations cleanly with a clear error without partial writes. |

---

## Estimator Accuracy (Predicted vs Measured)

All arithmetic uses unsigned 64-bit math (`uint64`) validated up to 128K context on multi-GPU configurations.

| Model (Backend) | Est. Weights | Actual CPU/GPU Split Reported | Est. KV | Actual KV (`llama-server` log) | Status |
| --- | --- | --- | --- | --- | --- |
| `Qwen3.5-4B` (CPU) | 2.5 GiB RAM | 2572.9 MiB CPU model | 0.1 GiB (128 MiB) | 128.0 MiB | **PASS (< 2% delta)** |
| `Spark-X2.6-4B` (CPU) | 2.4 GiB RAM | 2474.6 MiB CPU model | 0.2 GiB (252 MiB) | 252.0 MiB | **PASS (< 2% delta)** |
| `LFM2.5-8B-A1B` (CPU) | 4.5 GiB RAM | 4612.4 MiB CPU model | 48 MiB | 48.0 MiB | **PASS (< 2% delta)** |
| `Tiel-Coder-35B` (CUDA un-tuned) | 16.5 GiB vs 8 GiB VRAM | Estimator warned `does not fit (207% VRAM)`; runtime spilled 11.2 GiB to host RAM | 42 MiB | 42.5 MiB | **PASS (honest warning)** |

9 fixture hardware profiles covered in automated tests (`internal/estimate/estimate_test.go`):
1. 4 GiB RAM, no GPU (blocks 3 GiB model clearly)
2. 8 GiB RAM, 4 GiB discrete GPU (flags overflow)
3. 16 GiB RAM, 8 GiB discrete GPU
4. 32 GiB RAM, 24 GiB discrete GPU
5. Intel UHD integrated GPU only (unified RAM deduplicated once)
6. AMD discrete GPU via Vulkan
7. Dual-GPU laptop (Intel iGPU + NVIDIA dGPU selects discrete CUDA without double-counting unified RAM)
8. Dual NVIDIA GPUs (pools VRAM across CUDA devices)
9. Unknown accelerator fallback (degrades cleanly to CPU RAM-only estimate)

---

## Feature x Environment Matrix

| Feature | Windows Terminal | Legacy Console / `NO_COLOR` / `ascii=true` | exFAT | NTFS | FAT32 | Offline Air-Gapped | Non-Admin / Portable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Launcher & TUI Navigation | **PASS** | **DEGRADES-GRACEFULLY** (ASCII borders & gauges) | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| Clipboard Copy | **PASS** (Win32) | **DEGRADES-GRACEFULLY** (OSC 52 fallback) | — | — | — | **PASS** | **PASS** |
| Local Copy (`C` shortcut) | **PASS** | **PASS** | **PASS** (>4 GiB OK) | **PASS** | **DEGRADES-GRACEFULLY** (detects FAT32 and refuses >4 GiB single files clearly) | **PASS** | **PASS** |
| Shell Tool (`run_command`) | **PASS** (`cmd.exe`) | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| Web Search / Fetch & MCP | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **DEGRADES-GRACEFULLY** (no startup phone-home; clear error when offline/unconfigured) | **PASS** |
| Sessions, Skills & Checkpoints | **PASS** | **PASS** | **PASS** (no symlinks required) | **PASS** | **PASS** | **PASS** | **PASS** |
