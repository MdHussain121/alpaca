# Enforces repository cleanliness standards across source, scripts, and docs.
# Exits 0 on clean pass, 1 on any violation.
param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$violations = [System.Collections.Generic.List[string]]::new()

function Add-Violation([string]$kind, [string]$file, [int]$line, [string]$text) {
    $rel = $file
    if ($file.StartsWith($repo)) {
        $rel = $file.Substring($repo.Length).TrimStart('\', '/')
    }
    $violations.Add(("[$kind] {0}:{1}: {2}" -f $rel, $line, $text.Trim()))
}

$excludedDirs = '\\(\.git|dist|build|evidence|perf|eval|release-evidence|vendor|node_modules)\\'
$excludedDocs = @(
    'docs\DECISIONS.md',
    'docs\PERF.md',
    'docs\AGENT_NOTES.md',
    'docs\AGENT_EVAL.md'
)
$legacyExemptFiles = @(
    'internal\config\migrate.go',
    'internal\config\migrate_test.go',
    'scripts\check-clean.ps1',
    'scripts\cleanup-c-drive.ps1'
)

$sourceFiles = Get-ChildItem -Recurse -File | Where-Object {
    $_.FullName -notmatch $excludedDirs -and
    $_.Extension -in @('.go', '.ps1', '.bat', '.md', '.toml', '.txt', '.json', '.svg')
}

foreach ($f in $sourceFiles) {
    $rel = $f.FullName.Substring($repo.Length).TrimStart('\', '/').Replace('/', '\')
    if ($rel -in $excludedDocs) { continue }
    $isSelf = ($rel -eq 'scripts\check-clean.ps1')
    $isLegacyExempt = ($rel -in $legacyExemptFiles)
    if ($isSelf) { continue }

    $lines = Get-Content -Path $f.FullName -ErrorAction SilentlyContinue
    if (-not $lines) { continue }

    $codeCommentRun = 0
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $ln = [string]$lines[$i]
        $lineNo = $i + 1

        # 1. Phase / gate / milestone / old codename wording
        if ($ln -match '(?i)\bphase[\s_-]*[0-9]+\b|\bphase\s+(one|two|three|four|five|six|seven|eight|nine)\b') {
            Add-Violation 'PHASE' $rel $lineNo $ln
        }
        if ($ln -match '(?i)\bgate[\s_-]*[0-9]+\b' -and $ln -notmatch 'approval gate') {
            Add-Violation 'GATE' $rel $lineNo $ln
        }
        if ($ln -match '(?i)\bP[0-8]\b' -and $ln -notmatch 'P0\s+to\s+P8' -and $rel -notmatch 'internal\\hw\\') {
            # Allow hardware core/port labels if any, catch P0-P8 milestones
            if ($ln -match '\bP[0-8]\b.*(task|phase|milestone|priority|gate|step|sprint)') {
                Add-Violation 'MILESTONE' $rel $lineNo $ln
            }
        }
        if ($ln -match '(?i)(TurboQuant|PortableAI|\bpllama\b)' -and -not $isLegacyExempt) {
            if ($ln -notmatch 'legacy.*pllama') {
                Add-Violation 'CODENAME' $rel $lineNo $ln
            }
        }
        if ($ln -match '(?i)\bv1\.[12](?!\.\d)\b') {
            Add-Violation 'OLD_VER' $rel $lineNo $ln
        }
        if ($ln -match '(?i)\bpendrive\b') {
            Add-Violation 'OLD_TERM' $rel $lineNo $ln
        }
        if ($ln -match '\bllm\.exe\b|\bllmbench\b' -and -not $isLegacyExempt) {
            if ($ln -notmatch 'legacy') {
                Add-Violation 'OLD_BIN' $rel $lineNo $ln
            }
        }

        # 2. Hardcoded local paths / usernames / PII
        if ($ln -match '(?i)(C:\\Users\\|C:/Users/|\\moham\b|/moham\b|C:\\tmp|C:/tmp)') {
            Add-Violation 'HARD_PATH' $rel $lineNo $ln
        }
        $symbolicPathExempt = ($rel -in @(
            'internal\agent\sandbox.go',
            'internal\agent\sandbox_test.go',
            'internal\agent\agent_test.go',
            'internal\flags\flags_test.go',
            'internal\hw\hw_windows.go',
            'internal\ui\model.go'
        ))
        if (-not $symbolicPathExempt -and $ln -match '(?i)(?<![A-Za-z0-9_])[CD]:\\(?![rnts\\*])') {
            Add-Violation 'HARD_PATH' $rel $lineNo $ln
        }

        # 3. TODO / FIXME without issue tracker number (#123)
        if ($ln -match '\b(TODO|FIXME)\b' -and $ln -notmatch '(TODO|FIXME)\s*\(#[0-9]+\)|#\d+') {
            Add-Violation 'TODO' $rel $lineNo $ln
        }

        # 4. Commented-out code blocks (> 3 consecutive lines of Go/script code comments)
        if ($f.Extension -eq '.go') {
            if ($ln -match '^\s*//\s*(.+)$') {
                $body = $Matches[1]
                if ($body -match '(:=|^\s*(func|if|for|switch|return|var|type|case)\b|[{};]\s*$|\.[A-Za-z0-9_]+\()') {
                    $codeCommentRun++
                    if ($codeCommentRun -eq 4) {
                        Add-Violation 'COMMENTED_CODE' $rel ($lineNo - 3) "commented-out code block (>3 lines)"
                    }
                } else {
                    $codeCommentRun = 0
                }
            } else {
                $codeCommentRun = 0
            }
        }
    }
}

# 5. Stray temporary/binary files outside dist/ and build/
$strayBins = Get-ChildItem -Recurse -File | Where-Object {
    $_.FullName -notmatch $excludedDirs -and
    ($_.Extension -in @('.log', '.tmp', '.bak', '.exe') -or $_.Name -match '\.test\.exe$')
}
foreach ($b in $strayBins) {
    $rel = $b.FullName.Substring($repo.Length).TrimStart('\', '/')
    Add-Violation 'STRAY_FILE' $rel 0 $b.Name
}

if ($violations.Count -gt 0) {
    Write-Host ("`nCleanliness check FAILED ({0} violations):" -f $violations.Count) -ForegroundColor Red
    foreach ($v in $violations) {
        Write-Host "  $v" -ForegroundColor Yellow
    }
    exit 1
}

Write-Host "Cleanliness check PASSED (0 violations across $($sourceFiles.Count) files)." -ForegroundColor Green
exit 0
