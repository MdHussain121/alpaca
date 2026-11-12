<p align="center">
  <img src="assets/logo.svg" alt="alpaca logo" width="128">
</p>

<h1 align="center">alpaca</h1>

<p align="center">
  Portable Windows terminal launcher, VRAM/RAM fit estimator, and local coding-agent TUI for <code>llama.cpp</code> GGUF models.
</p>

---

## Features

- **Portable Single Executable:** `alpaca.exe` runs directly from any folder or USB drive with zero installer, zero `.NET`/Python dependency, and static-CRT `llama-server.exe` backends (`CUDA`, `CPU AVX2`, `CPU SSE4.2`).
- **Live Memory & Context Estimator:** Inspects GGUF architecture tensors, RoPE, MoE buffers, KV-cache quantization, Windows commit limits, and unified/multi-GPU memory before launch.
- **Built-In Local Coding Agent:** Terminal chat with workspace-confined file tools, checkpoint `/undo`, permission modes (`ask`, `read-auto`, `workspace-auto`), Agent Skills (`use_skill` / `save_skill`), stdio MCP servers (`mcp.toml`), SSRF-safe web search/fetch, and a runtime-interpolated built-in system prompt (`/system`).
- **Branching Sessions & Context Meter:** JSONL sessions with `/edit` and `/regen` branches, Markdown export, live `/context` per-category token breakdown, and automatic compaction.

## Requirements

- **OS:** Windows 10 or Windows 11 (x64)
- **GPU (optional):** NVIDIA GPU with CUDA driver (for `bin\cuda`), or any x64 CPU (SSE4.2 / AVX2 supported via `bin\cpu-compat` and `bin\cpu`)
- **Models:** Local `.gguf` files (place next to `alpaca.exe`, point `models_dir` in `config.toml`, or press `o`)

## Quick Start & Verification

1. Download **`alpaca-<version>-win-x64-cuda.zip`** (NVIDIA CUDA + CPU backends) or **`alpaca-<version>-win-x64.zip`** (CPU-only backends) from Releases.
2. Verify the archive hash against `SHA256SUMS.txt`:

```powershell
Get-FileHash .\alpaca-v1.0.0-win-x64-cuda.zip -Algorithm SHA256
```

> **Windows SmartScreen note:** Release binaries are unsigned open-source builds compiled cleanly from this repository. Verify the SHA-256 checksum matches `SHA256SUMS.txt` before approving execution.

3. Extract into any folder (including a portable exFAT/NTFS USB drive), drop `.gguf` model files alongside `alpaca.exe`, and launch:

```powershell
.\alpaca.exe
```

Copy [`config.example.toml`](config.example.toml) to `config.toml` (and [`mcp.toml.example`](mcp.toml.example) to `mcp.toml` if using MCP servers) to customize defaults.

## Directory Layout

```text
<root>/
  alpaca.exe             Portable Go TUI & launcher
  config.toml            Optional configuration (migrates legacy settings automatically)
  mcp.toml               Optional stdio MCP server config
  SHA256SUMS.txt         Release SHA-256 verification manifest
  prompts/               Optional system.md / system-compact.md overrides
  bin/
    cuda/                CUDA llama-server.exe + cublas64_13.dll + cublasLt64_13.dll
    cpu/                 AVX2 static-CRT llama-server.exe
    cpu-compat/          SSE4.2 fallback static-CRT llama-server.exe
  skills/                Agent Skills (skills/<name>/SKILL.md)
  workspace/             Default sandboxed agent workspace
  presets/               Saved launcher parameter presets
  sessions/              Saved JSONL chats & Markdown exports
  logs/                  llama-server launch logs
```

## Documentation

- [Usage, Shortcuts & Slash Commands](docs/USAGE.md)
- [Hardware & OS Compatibility Matrix](docs/COMPATIBILITY.md)
- [Security Model & Permission Boundaries](docs/SECURITY.md)
- [Agent Skills Authoring & Progressive Loading](docs/SKILLS.md)
- [Building from Source & Packaging](docs/BUILDING.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)

## Security & Safety Limits

- **Workspace confinement:** File tools canonicalize symlinks/junctions and reject paths outside the active workspace, UNC paths, ADS streams, and reserved Windows device names.
- **Permission gates:** `run_command` (executes via `cmd.exe` with current user rights) and MCP tools (`server.tool`) always confirm unless allowlisted for the session; `/undo` reverts workspace file edits only, not external shell side effects.
- **Taint defense:** Web fetch/search results are wrapped in untrusted data tags, checked for SSRF (private/loopback/metadata IPs blocked), and elevate writes/exec calls to confirmation prompts.
- **Zero telemetry:** All state stays locally inside the launcher root folder.

## License & Notices

Licensed under the [MIT License](LICENSE). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for `llama.cpp` (`MIT`), Go dependencies (`MIT` / `BSD-3-Clause`), and NVIDIA CUDA `cuBLAS` redistribution notices. `alpaca` is an independent open-source project and is not affiliated with `llama.cpp`, `ggml-org`, or NVIDIA Corporation.
