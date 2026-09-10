[CmdletBinding()]
param(
    [ValidateSet("lan", "localhost", "tunnel")]
    [string]$ExpoConnection = "lan"
)

$ErrorActionPreference = "Stop"
$appDirectory = Join-Path $PSScriptRoot "mobile-app"
if (-not (Test-Path -LiteralPath (Join-Path $appDirectory "package.json"))) {
    throw "找不到 mobile-app/package.json"
}

Push-Location $appDirectory
try {
    & npm.cmd start -- "--$ExpoConnection"
}
finally {
    Pop-Location
}
