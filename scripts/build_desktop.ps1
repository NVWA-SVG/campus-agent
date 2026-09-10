[CmdletBinding()]
param(
    [switch]$IncludeModelCache,
    [switch]$SkipTests,
    [ValidateSet("official", "rsproxy")]
    [string]$CargoRegistry = "official"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$cargoManifest = Join-Path $projectRoot "desktop-rust\Cargo.toml"
$workerSpec = Join-Path $projectRoot "packaging\campus-agent-worker.spec"
$cargoMirrorConfig = Join-Path $projectRoot "packaging\cargo-rsproxy.toml"
$buildRoot = Join-Path $projectRoot "build\desktop"
$pyinstallerDist = Join-Path $buildRoot "pyinstaller-dist"
$pyinstallerWork = Join-Path $buildRoot "pyinstaller-work"
$distribution = Join-Path $projectRoot "dist\CampusAgent"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment was not found: $python"
}
$cargoCommand = Get-Command cargo.exe -ErrorAction SilentlyContinue
$cargo = if ($cargoCommand) {
    $cargoCommand.Source
} else {
    Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
}
if (-not (Test-Path -LiteralPath $cargo -PathType Leaf)) {
    throw "Rust Cargo was not found. Install Rustup and reopen PowerShell first."
}

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is missing. Run: .\.venv\Scripts\python.exe -m pip install -e ".[semantic,dev,desktop-build]"'
}

if (-not $SkipTests) {
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "Python tests failed; desktop packaging was stopped."
    }
}

Push-Location $projectRoot
try {
    $cargoRegistryArgs = @()
    if ($CargoRegistry -eq "rsproxy") {
        $cargoRegistryArgs = @("--config", $cargoMirrorConfig)
    }

    & $cargo test --manifest-path $cargoManifest @cargoRegistryArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Rust tests failed."
    }

    & $cargo build --manifest-path $cargoManifest --release @cargoRegistryArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Rust release build failed."
    }

    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $pyinstallerDist `
        --workpath $pyinstallerWork `
        $workerSpec
    if ($LASTEXITCODE -ne 0) {
        throw "Python Agent Worker packaging failed."
    }

    if (Test-Path -LiteralPath $distribution) {
        $resolvedDistRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "dist")).Path
        $resolvedTarget = (Resolve-Path -LiteralPath $distribution).Path
        if (-not $resolvedTarget.StartsWith($resolvedDistRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to replace a distribution directory outside dist."
        }
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $distribution | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $distribution "runtime") | Out-Null

    Copy-Item -LiteralPath `
        (Join-Path $projectRoot "desktop-rust\target\release\campus-agent-desktop.exe") `
        -Destination (Join-Path $distribution "CampusAgent.exe")
    Copy-Item -LiteralPath `
        (Join-Path $pyinstallerDist "campus-agent-worker") `
        -Destination (Join-Path $distribution "runtime\campus-agent-worker") `
        -Recurse

    if ($IncludeModelCache) {
        $modelCache = Join-Path $projectRoot ".campus_agent_data\models"
        if (-not (Test-Path -LiteralPath $modelCache -PathType Container)) {
            throw "The local BGE model cache does not exist: $modelCache"
        }
        Copy-Item -LiteralPath $modelCache -Destination (Join-Path $distribution "models") -Recurse
    }

    Write-Host ""
    Write-Host "Campus Agent desktop distribution created:"
    Write-Host "  $distribution"
    Write-Host ""
    Write-Host "Run:"
    Write-Host "  $distribution\CampusAgent.exe"
} finally {
    Pop-Location
}
