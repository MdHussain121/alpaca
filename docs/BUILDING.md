# Building alpaca from Source

## Prerequisites

- **Go 1.27+** (with `CGO_ENABLED=0` for pure-Go builds of `alpaca.exe`)
- **Windows x64** (`win-x64` release target)
- Optional (for rebuilding `llama-server.exe` binaries under `bin/`):
  - Visual Studio 2022 Build Tools (`cl.exe`, `cmake`, `ninja` or MSBuild)
  - NVIDIA CUDA Toolkit 13.x (for the `cuda` backend build)

## Building `alpaca.exe`

```powershell
# Embed icon resource (requires Windows .syso already committed in cmd/alpaca/)
go generate ./...   # optional if regenerating PNG/ICO/SYSO assets via scripts/gen_assets.go

$env:CGO_ENABLED = "0"
go build -trimpath -ldflags "-s -w -X main.version=v1.0.0 -X main.commit=$(git rev-parse --short HEAD) -X main.llamaTag=b12284" -o alpaca.exe ./cmd/alpaca
```

## Running Tests and Cleanliness Checks

```powershell
go test ./...
go vet ./...
.\scripts\check-clean.ps1
```

## Building Static-CRT `llama.cpp` Binaries

Place a `llama.cpp` checkout at `build\llamacpp` (tag `b12284` recommended) and run [`build/build-llama.bat`](../build/build-llama.bat):

```cmd
build\build-llama.bat all
```

Available profiles inside [`build/build-llama.bat`](../build/build-llama.bat):

- `cpu` — AVX2 with `/MT` static MSVC CRT and `GGML_OPENMP=OFF`
- `cpu-compat` — SSE4.2 fallback with `/MT` static CRT (`GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_F16C=OFF`)
- `cuda` — Ampere/Ada/Hopper (`sm_75;sm_80;sm_86;sm_89;compute_120`) with `/MT` static CRT and dynamic CUDA runtime (`cublas64_13.dll`, `cublasLt64_13.dll`)
- `cuda-full` / `cuda-sm89` — Wider/targeted CUDA architecture builds

No `VCRUNTIME140.dll`, `MSVCP140.dll`, or `vcomp140.dll` redistributable installs are required by end users on clean Windows 10/11 machines.

## Packaging Release Archives

```powershell
# Inspect dry-run plan
.\scripts\release.ps1 -Version v1.0.0

# Build release zip archives + SHA256SUMS.txt
.\scripts\release.ps1 -Version v1.0.0 -Execute
```
