# Build the portable folder for alpaca.exe
#
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1 [-Profile full|sm89] [-IncludeTools] [-SkipBuild]
#
# Produces dist\ with the layout:
#   alpaca.exe, config.toml, SHA256SUMS.txt, bin\cuda\llama-server.exe + cuBLAS DLLs,
#   bin\cpu\llama-server.exe, [tools\], models\, presets\, logs\

param(
    [string]$Out = "dist",
    [string]$LlamaRoot = "build\llamacpp",
    [ValidateSet("full", "sm89")]
    [string]$Profile = "full",
    [string]$CudaToolkit = "",   # auto-detected from CUDA_PATH, else the newest install
    [switch]$IncludeTools,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Info($m) { Write-Host "  $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "  $m" -ForegroundColor Red; exit 1 }

function Resolve-CudaBinDir([string]$root) {
    if (-not $root) { return "" }
    foreach ($candidate in @((Join-Path $root "bin\x64"), (Join-Path $root "bin"), $root)) {
        if ((Test-Path $candidate) -and (Get-ChildItem $candidate -Filter "cublas64_*.dll" -ErrorAction SilentlyContinue)) {
            return $candidate
        }
    }
    return ""
}

# No hardcoded CUDA location: use CUDA_PATH, else the newest installed toolkit.
if (-not $CudaToolkit) {
    if ($env:CUDA_PATH) {
        $CudaToolkit = Resolve-CudaBinDir $env:CUDA_PATH
    }
    if (-not $CudaToolkit) {
        $cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $cudaRoot) {
            $latest = Get-ChildItem $cudaRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
            if ($latest) { $CudaToolkit = Resolve-CudaBinDir $latest.FullName }
        }
    }
} else {
    $resolved = Resolve-CudaBinDir $CudaToolkit
    if ($resolved) { $CudaToolkit = $resolved }
}
if ($CudaToolkit -and (Test-Path $CudaToolkit)) {
    Info "CUDA runtime from $CudaToolkit (profile: $Profile)"
} else {
    $CudaToolkit = ""
    Write-Host "  warning: no CUDA toolkit found; bin\cuda keeps existing cuBLAS DLLs if present" -ForegroundColor Yellow
}

Write-Host "`n== packaging alpaca ($Profile) ==" -ForegroundColor Magenta

New-Item -ItemType Directory -Force -Path $Out | Out-Null

# 1. Go binary
if (-not $SkipBuild) {
    Info "building alpaca.exe"
    $env:CGO_ENABLED = "0"
    & go build -trimpath -ldflags "-s -w" -o "$Out\alpaca.exe" .\cmd\alpaca
    if ($LASTEXITCODE -ne 0) { Fail "go build failed" }
}
if (-not (Test-Path "$Out\alpaca.exe")) { Fail "alpaca.exe missing; run without -SkipBuild" }

$size = (Get-Item "$Out\alpaca.exe").Length
Info ("alpaca.exe {0:N2} MiB ($size B)" -f ($size / 1MB))

function Resolve-LlamaBin([string[]]$candidates) {
    foreach ($rel in $candidates) {
        foreach ($sub in @("bin\Release", "bin")) {
            $p = Join-Path $LlamaRoot (Join-Path $rel $sub)
            if (Test-Path (Join-Path $p "llama-server.exe")) { return $p }
        }
    }
    return ""
}

$cudaCandidates = if ($Profile -eq "sm89") { @("build-cuda-sm89") } else { @("build-cuda-full", "build-cuda") }
$cudaSrc = Resolve-LlamaBin $cudaCandidates
if (-not $cudaSrc) {
    Fail "no cuda build found for profile '$Profile' under $LlamaRoot (run build\build-llama.bat $Profile first)"
}
$cpuSrc = Resolve-LlamaBin @("build-cpu")

# 2. Clean runtime binary folders so old CLI/RPC tools or prebuilt/eval dirs never linger.
$binDir  = Join-Path $Out "bin"
$cudaDst = Join-Path $Out "bin\cuda"
$cpuDst  = Join-Path $Out "bin\cpu"
$toolsDst = Join-Path $Out "tools"
foreach ($dir in @($binDir, $toolsDst)) {
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
}

New-Item -ItemType Directory -Force -Path $cudaDst | Out-Null
Copy-Item (Join-Path $cudaSrc "llama-server.exe") $cudaDst -Force
if (Test-Path (Join-Path $cudaSrc "*.dll")) { Copy-Item (Join-Path $cudaSrc "*.dll") $cudaDst -Force }
Info "bin\cuda <- $cudaSrc (llama-server.exe)"

if ($cpuSrc) {
    New-Item -ItemType Directory -Force -Path $cpuDst | Out-Null
    Copy-Item (Join-Path $cpuSrc "llama-server.exe") $cpuDst -Force
    if (Test-Path (Join-Path $cpuSrc "*.dll")) { Copy-Item (Join-Path $cpuSrc "*.dll") $cpuDst -Force }
    Info "bin\cpu <- $cpuSrc (llama-server.exe)"
} else {
    Info "no cpu build found; deriving bin\cpu from bin\cuda (CUDA module removed)"
    New-Item -ItemType Directory -Force -Path $cpuDst | Out-Null
    Copy-Item (Join-Path $cudaDst "*") $cpuDst -Force -Recurse
    foreach ($dll in @("ggml-cuda.dll", "ggml-rpc.dll", "cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
        Remove-Item (Join-Path $cpuDst $dll) -Force -ErrorAction SilentlyContinue
    }
}

# 3. CUDA runtime DLLs (cublas + cublasLt always needed; cudart only if dynamically imported).
$serverBytes = [System.IO.File]::ReadAllText((Join-Path $cudaDst "llama-server.exe"), [System.Text.Encoding]::ASCII)
$needsCudart = $serverBytes.Contains("cudart64_")
$needed = @("cublas64_*.dll", "cublasLt64_*.dll")
if ($needsCudart) { $needed = @("cudart64_*.dll") + $needed } else { Info "cudart statically linked (GGML_STATIC=ON); omitting cudart64_*.dll" }

foreach ($pattern in $needed) {
    if (-not $CudaToolkit) { break }
    if (-not (Test-Path (Join-Path $cudaDst $pattern))) {
        $matches = Get-ChildItem $CudaToolkit -Filter $pattern -ErrorAction SilentlyContinue
        if ($matches) {
            Copy-Item $matches.FullName $cudaDst -Force
            Info "added $($matches.Name) from CUDA toolkit"
        } else {
            Write-Host "  warning: $pattern not found in $CudaToolkit" -ForegroundColor Yellow
        }
    }
}

# Optional diagnostic tools (llama-bench, llama-perplexity) separated into tools\
if ($IncludeTools) {
    New-Item -ItemType Directory -Force -Path $toolsDst | Out-Null
    foreach ($tool in @("llama-bench.exe", "llama-perplexity.exe")) {
        $tp = Join-Path $cudaSrc $tool
        if (-not (Test-Path $tp) -and $cpuSrc) { $tp = Join-Path $cpuSrc $tool }
        if (Test-Path $tp) {
            Copy-Item $tp $toolsDst -Force
            Info "tools\$tool <- $(Split-Path -Parent $tp)"
        }
    }
}

# 4. Folder skeleton & starter workspace / skills / config templates
foreach ($dir in @("models", "presets", "logs", "workspace", "sessions", "skills")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Out $dir) | Out-Null
}
$starterSkills = Join-Path $repo "skills"
if (Test-Path $starterSkills) {
    Get-ChildItem $starterSkills -Directory | ForEach-Object {
        $dstSkill = Join-Path (Join-Path $Out "skills") $_.Name
        if (-not (Test-Path $dstSkill)) {
            Copy-Item $_.FullName $dstSkill -Recurse -Force
        }
    }
    Info "staged starter skills in $Out\skills"
}

# 5. Default config templates
if (-not (Test-Path "$Out\config.toml")) {
    @'
theme = "mocha"
backend = "auto"
port = 8080
models_dir = "."
compaction_threshold_pct = 80
long_gen_bell = false
active_profile = "coding"

[web]
# Set search_provider = "brave" | "tavily" | "searxng" (or env TAVILY_API_KEY / BRAVE_SEARCH_API_KEY / SEARXNG_URL).
# Warning: API keys stored in config.toml travel with the drive.
search_provider = ""
search_api_key = ""
searxng_url = ""
max_fetch_bytes = 2097152

[last_used]
'@ | Set-Content -Path "$Out\config.toml" -Encoding ASCII
    Info "wrote config.toml"
}
if (-not (Test-Path "$Out\mcp.toml.example")) {
    @'
# Copy to mcp.toml to enable stdio MCP servers (tools exposed as server.tool).
# Environment secrets under [servers.<name>.env] are never shown to the model or logged.
#
# [servers.example]
# command = "my-mcp-server.exe"
# args = ["--stdio"]
# enabled = true
# timeout_s = 30
# class = "net"
# [servers.example.env]
# API_KEY = "replace-me"
'@ | Set-Content -Path "$Out\mcp.toml.example" -Encoding ASCII
    Info "wrote mcp.toml.example"
}

# 6. Sanity check required runtime binaries
foreach ($req in @(
    (Join-Path $cudaDst "llama-server.exe"),
    (Join-Path $cpuDst "llama-server.exe")
)) {
    if (-not (Test-Path $req)) { Fail "missing required binary: $req" }
}
if (-not (Get-ChildItem $cudaDst -Filter "cublas64_*.dll" -ErrorAction SilentlyContinue)) {
    Fail "bin\cuda missing cublas64_*.dll"
}

# 7. SHA-256 manifest of packaged runtime artifacts
$outAbs = (Resolve-Path $Out).Path
$manifestPath = Join-Path $Out "SHA256SUMS.txt"
$manifestFiles = @()
if (Test-Path (Join-Path $outAbs "alpaca.exe")) { $manifestFiles += Get-Item (Join-Path $outAbs "alpaca.exe") }
foreach ($sub in @("bin", "tools")) {
    $sp = Join-Path $outAbs $sub
    if (Test-Path $sp) {
        $manifestFiles += Get-ChildItem $sp -Recurse -File
    }
}
$lines = $manifestFiles | Sort-Object FullName | ForEach-Object {
    $hash = (Get-FileHash -Algorithm SHA256 -Path $_.FullName).Hash.ToLowerInvariant()
    $rel = $_.FullName.Substring($outAbs.Length).TrimStart('\', '/').Replace('\', '/')
    "$hash  $rel"
}
$lines | Set-Content -Path $manifestPath -Encoding ASCII
Info "wrote SHA256SUMS.txt ($($lines.Count) entries)"

$runtimeBytes = ($manifestFiles | Measure-Object -Property Length -Sum).Sum
Info ("runtime bundle ($Profile, without models): {0:N2} MiB ($runtimeBytes B)" -f ($runtimeBytes / 1MB))

Write-Host "`nDone. Copy $Out to the portable drive root, or run .\$Out\alpaca.exe`n" -ForegroundColor Green
