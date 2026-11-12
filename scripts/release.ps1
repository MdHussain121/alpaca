# Release packager for alpaca
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\release.ps1 -Version v1.0.0 [-Execute]
#
# Dry-run is the default; pass -Execute to build binaries, stage release folders,
# create zip archives (< 2 GiB), write SHA256SUMS.txt, and emit release-notes.md.

param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutDir = "dist\release",
    [string]$BinSource = "dist\bin",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$commit = (git rev-parse --short HEAD 2>$null).Trim()
if (-not $commit) { $commit = "unknown" }

$llamaTag = "b12284"
$llamaCommit = ""
if (Test-Path "build\llamacpp\.git") {
    $desc = (git -C build/llamacpp describe --tags --always 2>$null).Trim()
    if ($desc) { $llamaTag = $desc }
    $lc = (git -C build/llamacpp rev-parse HEAD 2>$null).Trim()
    if ($lc) { $llamaCommit = $lc }
}

function Ensure-Dir([string]$p) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Resolve-BackendFile([string]$profile, [string]$fileName) {
    $direct = Join-Path (Join-Path $BinSource $profile) $fileName
    if (Test-Path $direct) { return $direct }
    $fallbackMap = @{
        "cpu"        = "build\llamacpp\build-cpu\bin\Release\$fileName"
        "cpu-compat" = "build\llamacpp\build-cpu-compat\bin\Release\$fileName"
        "cuda"       = "build\llamacpp\build-cuda\bin\Release\$fileName"
    }
    if ($fallbackMap.ContainsKey($profile) -and (Test-Path $fallbackMap[$profile])) {
        return $fallbackMap[$profile]
    }
    return ""
}

$cpuExe = Resolve-BackendFile "cpu" "llama-server.exe"
$compatExe = Resolve-BackendFile "cpu-compat" "llama-server.exe"
$cudaExe = Resolve-BackendFile "cuda" "llama-server.exe"
$cublas = Resolve-BackendFile "cuda" "cublas64_13.dll"
$cublasLt = Resolve-BackendFile "cuda" "cublasLt64_13.dll"

foreach ($req in @(
    @{ Name = "cpu llama-server.exe"; Path = $cpuExe },
    @{ Name = "cpu-compat llama-server.exe"; Path = $compatExe },
    @{ Name = "cuda llama-server.exe"; Path = $cudaExe },
    @{ Name = "cublas64_13.dll"; Path = $cublas },
    @{ Name = "cublasLt64_13.dll"; Path = $cublasLt }
)) {
    if (-not $req.Path -or -not (Test-Path $req.Path)) {
        throw "Missing required backend artifact: $($req.Name)"
    }
}

$cpuZipName = "alpaca-$Version-win-x64.zip"
$cudaZipName = "alpaca-$Version-win-x64-cuda.zip"

$docFiles = @(
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "config.example.toml",
    "mcp.toml.example"
)
$docSubFiles = @(
    "docs\USAGE.md",
    "docs\COMPATIBILITY.md",
    "docs\BUILDING.md",
    "docs\SECURITY.md",
    "docs\SKILLS.md"
)
$promptFiles = @(
    "prompts\system.md",
    "prompts\system-compact.md"
)

Write-Host "== alpaca release plan ($Version) ==" -ForegroundColor Cyan
Write-Host "  Commit       : $commit (llama.cpp $llamaTag)"
Write-Host "  Output dir   : $OutDir"
Write-Host "  Backends src : cpu=$($cpuExe), cpu-compat=$($compatExe), cuda=$($cudaExe)"
Write-Host "  Artifacts    : $cpuZipName, $cudaZipName, alpaca.exe, BUILD_INFO.txt, SHA256SUMS.txt, release-notes.md"

if (-not $Execute) {
    Write-Host "`n[DRY-RUN] Re-run with -Execute to build and package artifacts." -ForegroundColor Yellow
    exit 0
}

Ensure-Dir $OutDir
$stageCpu = Join-Path $OutDir "stage-cpu"
$stageCuda = Join-Path $OutDir "stage-cuda"
foreach ($d in @($stageCpu, $stageCuda)) {
    if (Test-Path $d) { Remove-Item $d -Recurse -Force }
    Ensure-Dir $d
}

$alpacaExe = Join-Path $OutDir "alpaca.exe"
$ldflags = "-s -w -X main.version=$Version -X main.commit=$commit -X main.llamaTag=$llamaTag"
Write-Host "  Building $alpacaExe..." -ForegroundColor Cyan
$env:CGO_ENABLED = "0"
& go build -trimpath -ldflags $ldflags -o $alpacaExe .\cmd\alpaca
if ($LASTEXITCODE -ne 0) { throw "go build failed" }

$goVer = (go version).Trim()
$buildInfo = @"
alpaca release build info
=========================
Version       : $Version
Git commit    : $commit
llama.cpp tag : $llamaTag $(if ($llamaCommit) { "($llamaCommit)" })
Go toolchain  : $goVer
Build flags   : CGO_ENABLED=0 -trimpath -ldflags "$ldflags"
MSVC CRT      : static (/MT), GGML_OPENMP=OFF
CUDA arches   : sm_75;sm_80;sm_86;sm_89;compute_120 (cuBLAS dynamic)
Built UTC     : $((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))
"@
$buildInfoPath = Join-Path $OutDir "BUILD_INFO.txt"
Set-Content -Path $buildInfoPath -Value $buildInfo -Encoding UTF8

function Populate-Stage([string]$stageDir, [bool]$includeCuda) {
    Copy-Item $alpacaExe (Join-Path $stageDir "alpaca.exe") -Force
    Copy-Item $buildInfoPath (Join-Path $stageDir "BUILD_INFO.txt") -Force

    foreach ($f in $docFiles) {
        if (Test-Path $f) { Copy-Item $f (Join-Path $stageDir $f) -Force }
    }
    foreach ($sub in @("docs", "prompts")) {
        Ensure-Dir (Join-Path $stageDir $sub)
    }
    foreach ($f in ($docSubFiles + $promptFiles)) {
        if (Test-Path $f) { Copy-Item $f (Join-Path $stageDir $f) -Force }
    }

    $cpuDst = Join-Path $stageDir "bin\cpu"
    $compatDst = Join-Path $stageDir "bin\cpu-compat"
    Ensure-Dir $cpuDst
    Ensure-Dir $compatDst
    Copy-Item $cpuExe (Join-Path $cpuDst "llama-server.exe") -Force
    Copy-Item $compatExe (Join-Path $compatDst "llama-server.exe") -Force

    if ($includeCuda) {
        $cudaDst = Join-Path $stageDir "bin\cuda"
        Ensure-Dir $cudaDst
        Copy-Item $cudaExe (Join-Path $cudaDst "llama-server.exe") -Force
        Copy-Item $cublas (Join-Path $cudaDst "cublas64_13.dll") -Force
        Copy-Item $cublasLt (Join-Path $cudaDst "cublasLt64_13.dll") -Force
    }

    # Internal package manifest inside each zip
    $stageAbs = (Resolve-Path $stageDir).Path.TrimEnd('\', '/')
    $manifestLines = Get-ChildItem $stageAbs -Recurse -File | Sort-Object FullName | ForEach-Object {
        $rel = $_.FullName.Substring($stageAbs.Length + 1).Replace('\', '/')
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash *$rel"
    }
    Set-Content -Path (Join-Path $stageAbs "SHA256SUMS.txt") -Value $manifestLines -Encoding ASCII
}

Populate-Stage $stageCpu $false
Populate-Stage $stageCuda $true

$cpuZipPath = Join-Path $OutDir $cpuZipName
$cudaZipPath = Join-Path $OutDir $cudaZipName
foreach ($z in @($cpuZipPath, $cudaZipPath)) {
    if (Test-Path $z) { Remove-Item $z -Force }
}

Write-Host "  Compressing $cpuZipName..." -ForegroundColor Cyan
Compress-Archive -Path (Join-Path $stageCpu "*") -DestinationPath $cpuZipPath -CompressionLevel Optimal
Write-Host "  Compressing $cudaZipName..." -ForegroundColor Cyan
Compress-Archive -Path (Join-Path $stageCuda "*") -DestinationPath $cudaZipPath -CompressionLevel Optimal

foreach ($z in @($cpuZipPath, $cudaZipPath, $alpacaExe)) {
    $sz = (Get-Item $z).Length
    if ($sz -ge 2040109465) { # ~1.90 GiB safety ceiling below GitHub 2 GiB release asset limit
        throw "Asset $z exceeds 1.9 GiB ($sz bytes)"
    }
}

$sums = foreach ($item in @($cpuZipPath, $cudaZipPath, $alpacaExe, $buildInfoPath)) {
    $h = (Get-FileHash $item -Algorithm SHA256).Hash.ToLowerInvariant()
    $name = Split-Path -Leaf $item
    "$h *$name"
}
$sumsPath = Join-Path $OutDir "SHA256SUMS.txt"
Set-Content -Path $sumsPath -Value $sums -Encoding ASCII

$notes = @"
# alpaca $Version

Portable Windows x64 terminal launcher, VRAM/RAM fit estimator, and local coding-agent TUI powered by `llama.cpp` ($llamaTag).

## Release Assets

- **$cudaZipName** — Full Windows x64 bundle with NVIDIA CUDA (`sm_75`..`compute_120`, `cuBLAS 13.2`), AVX2 CPU (`bin/cpu`), and SSE4.2 fallback CPU (`bin/cpu-compat`).
- **$cpuZipName** — Compact CPU-only Windows x64 bundle (`bin/cpu` AVX2 + `bin/cpu-compat` SSE4.2 static-CRT builds; zero CUDA runtime needed).
- **alpaca.exe** — Standalone Go executable.
- **SHA256SUMS.txt** — Checksums for verifying downloaded assets.

## Verification & SmartScreen Notice

Before running downloaded executables, verify SHA-256 checksums:

``````powershell
Get-FileHash .\$cudaZipName -Algorithm SHA256
Get-FileHash .\$cpuZipName -Algorithm SHA256
``````

All binaries are unsigned open-source builds (`CGO_ENABLED=0`, static MSVC `/MT` CRT). Windows SmartScreen may warn on first launch; confirm the SHA-256 hash matches `SHA256SUMS.txt` before proceeding.

## SHA-256 Checksums

``````text
$($sums -join "`n")
``````
"@
Set-Content -Path (Join-Path $OutDir "release-notes.md") -Value $notes -Encoding UTF8

Remove-Item $stageCpu, $stageCuda -Recurse -Force

Write-Host "`nRelease packaging complete:" -ForegroundColor Green
Get-ChildItem $OutDir -File | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0,-32} {1,10:N2} MiB" -f $_.Name, ($_.Length / 1MB))
}
