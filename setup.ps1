<#
.SYNOPSIS
  Sets up the Mars repo on Windows: checks Go and the .NET SDK, downloads
  Go modules, builds the simulator tools and the observer, and creates the
  training folders. Safe to run again; finished steps are skipped.

.PARAMETER Install
  Install a missing Go or .NET SDK with winget (asks winget to accept the
  package agreements for you).

.PARAMETER Test
  Run the Go tests after building.

.PARAMETER Mocap
  Download the CMU motion-capture recordings used to extract the get-up
  references (about 12 MB, into sim/mocap/cmu-*). Only needed to
  re-extract references; the extracted ones are committed.

.PARAMETER Save
  Snapshot the training data (sim/checkpoints, sim/runs) into training/,
  which is tracked by git: commit it to carry training to other systems.

.PARAMETER Restore
  Copy the training/ snapshot over sim/checkpoints and sim/runs even if
  they already exist. (Setup restores it automatically when there is no
  local checkpoint yet.)

.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -Install -Test
  .\setup.ps1 -Save; git add training; git commit -m "training snapshot"
#>
[CmdletBinding()]
param([switch]$Install, [switch]$Test, [switch]$Mocap, [switch]$Save, [switch]$Restore)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$minGo = [version]'1.22'
$dotnetMajor = 10

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# Tools installed earlier in this session may not be on this shell's PATH yet.
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Find-Go {
    $cmd = Get-Command go -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in "$env:ProgramFiles\Go\bin\go.exe", "$env:LOCALAPPDATA\Programs\Go\bin\go.exe") {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Go-Version($go) {
    $v = & $go version   # e.g. "go version go1.27.3 windows/amd64"
    if ($v -match 'go(\d+)\.(\d+)') { return [version]"$($Matches[1]).$($Matches[2])" }
    return [version]'0.0'
}

function Has-Dotnet {
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { return $false }
    return [bool](& dotnet --list-sdks 2>$null | Where-Object { $_ -match "^$dotnetMajor\." })
}

function Winget-Install($id, $what) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is not available; install $what by hand."
    }
    Step "Installing $what with winget ($id)"
    & winget install --exact --id $id --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $what (exit $LASTEXITCODE)." }
    Refresh-Path
}

Refresh-Path

# --- Training data snapshot -------------------------------------------------
# The trainer rewrites sim/checkpoints and sim/runs every iteration, so
# those stay out of git; training/ holds a snapshot that git carries.
$snap = Join-Path $root 'training'
$live = @{ 'checkpoints' = (Join-Path $root 'sim\checkpoints'); 'runs' = (Join-Path $root 'sim\runs') }

function Copy-Tree($from, $to) {
    if (-not (Test-Path $from)) { return 0 }
    New-Item -ItemType Directory -Force $to | Out-Null
    $files = Get-ChildItem $from -File
    foreach ($f in $files) { Copy-Item $f.FullName (Join-Path $to $f.Name) -Force }
    return $files.Count
}

if ($Save) {
    Step 'Saving the training data snapshot (training/)'
    $n = 0
    foreach ($k in $live.Keys) { $n += Copy-Tree $live[$k] (Join-Path $snap $k) }
    if ($n -eq 0) { Warn 'no training data in sim/checkpoints or sim/runs yet' } else { Ok "$n files; commit training/ to carry them: git add training; git commit -m `"training snapshot`"" }
    return
}

# --- Go -------------------------------------------------------------------
Step "Go $minGo or newer"
$go = Find-Go
if ($go -and (Go-Version $go) -ge $minGo) {
    Ok "$(& $go version)"
} elseif ($Install) {
    Winget-Install 'GoLang.Go' 'Go'
    $go = Find-Go
    if (-not $go) { throw 'Go was installed but cannot be found; open a new terminal and run setup again.' }
    Ok "$(& $go version)"
} else {
    if ($go) { Warn "found $(& $go version), need $minGo or newer" } else { Warn 'Go not found' }
    throw "Install Go $minGo+ from https://go.dev/dl/ or run: .\setup.ps1 -Install"
}

# --- .NET SDK ---------------------------------------------------------------
Step ".NET SDK $dotnetMajor"
if (Has-Dotnet) {
    Ok ((& dotnet --list-sdks | Where-Object { $_ -match "^$dotnetMajor\." } | Select-Object -First 1))
} elseif ($Install) {
    Winget-Install "Microsoft.DotNet.SDK.$dotnetMajor" ".NET SDK $dotnetMajor"
    if (-not (Has-Dotnet)) { throw '.NET was installed but cannot be found; open a new terminal and run setup again.' }
    Ok 'installed'
} else {
    Warn ".NET SDK $dotnetMajor not found"
    throw "Install the .NET $dotnetMajor SDK from https://dotnet.microsoft.com/download or run: .\setup.ps1 -Install"
}

# --- Simulator ---------------------------------------------------------------
$sim = Join-Path $root 'sim'
Push-Location $sim
try {
    Step 'Go modules'
    & $go mod download
    if ($LASTEXITCODE -ne 0) { throw 'go mod download failed' }
    Ok 'downloaded'

    Step 'Building simulator tools (sim/bin)'
    New-Item -ItemType Directory -Force (Join-Path $sim 'bin') | Out-Null
    foreach ($tool in 'simserver', 'train', 'mocapref') {
        # Build beside the old exe and swap it in: Windows cannot overwrite
        # an exe that is running (training or the sim server may be).
        & $go build -o "bin\$tool.new.exe" "./cmd/$tool"
        if ($LASTEXITCODE -ne 0) { throw "building $tool failed" }
        try {
            Move-Item -Force "bin\$tool.new.exe" "bin\$tool.exe" -ErrorAction Stop
            Ok "bin\$tool.exe"
        } catch {
            Warn "bin\$tool.exe is running, so it was kept; the new build is bin\$tool.new.exe"
        }
    }

    Step 'Training folders'
    foreach ($d in 'checkpoints', 'runs') { New-Item -ItemType Directory -Force (Join-Path $sim $d) | Out-Null }
    Ok 'sim\checkpoints, sim\runs'
    $haveLocal = Test-Path (Join-Path $live['checkpoints'] 'policy.json')
    if (Test-Path $snap) {
        if ($Restore -or -not $haveLocal) {
            Step 'Restoring training data from training/'
            $n = 0
            foreach ($k in $live.Keys) { $n += Copy-Tree (Join-Path $snap $k) $live[$k] }
            Ok "$n files; resume with: cd sim; .\bin\train.exe -scenario getup -resume"
        } else {
            Ok 'local training data kept (use -Restore to replace it with the training/ snapshot)'
        }
    }

    if ($Test) {
        Step 'Go tests'
        & $go test ./...
        if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
    }
} finally {
    Pop-Location
}

# --- Observer ----------------------------------------------------------------
Step 'Building the observer (Blazor)'
$obsDir = Join-Path $root 'observer'
if (Get-Process Mars.Observer -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($obsDir, [StringComparison]::OrdinalIgnoreCase) }) {
    Warn 'the observer is running, so it was not rebuilt; stop it and run setup again to rebuild'
} else {
    & dotnet build (Join-Path $root 'observer') --nologo -v q
    if ($LASTEXITCODE -ne 0) { throw 'observer build failed' }
    Ok 'built'
}

# --- Motion capture (optional) ------------------------------------------------
if ($Mocap) {
    Step 'CMU motion capture (http://mocap.cs.cmu.edu; free for research and commercial use per CMU)'
    $clips = @{
        '114' = @('114.asf', '114_02.amc', '114_11.amc')
        '139' = @('139.asf', '139_16.amc', '139_17.amc', '139_18.amc')
        '140' = @('140.asf', '140_01.amc', '140_02.amc', '140_03.amc', '140_04.amc', '140_08.amc', '140_09.amc')
    }
    foreach ($s in $clips.Keys) {
        $dir = Join-Path $sim "mocap\cmu-$s"
        New-Item -ItemType Directory -Force $dir | Out-Null
        foreach ($f in $clips[$s]) {
            $out = Join-Path $dir $f
            if (Test-Path $out) { continue }
            Invoke-WebRequest -UseBasicParsing "http://mocap.cs.cmu.edu/subjects/$s/$f" -OutFile $out -TimeoutSec 120
            Ok "mocap\cmu-$s\$f"
        }
    }
    Ok 'all recordings present'
}

Write-Host ''
Write-Host 'Ready. Next:' -ForegroundColor Cyan
Write-Host '  cd sim; .\bin\train.exe -scenario getup              # train (writes checkpoints\policy.json)'
Write-Host '  cd sim; .\bin\simserver.exe -policy checkpoints\policy.json -follow'
Write-Host '  dotnet run --project observer                         # then open http://localhost:5217'
