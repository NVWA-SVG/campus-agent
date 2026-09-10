[CmdletBinding()]
param(
    [string]$ListenAddress = $(
        if ($env:CAMPUS_SERVER_HOST) { $env:CAMPUS_SERVER_HOST } else { "127.0.0.1" }
    ),
    [int]$ServerPort = $(
        if ($env:CAMPUS_SERVER_PORT) { [int]$env:CAMPUS_SERVER_PORT } else { 8000 }
    )
)

$ErrorActionPreference = "Stop"

if (-not $env:CAMPUS_MOBILE_API_TOKEN -or $env:CAMPUS_MOBILE_API_TOKEN.Length -lt 32) {
    throw "请先设置至少32位的 CAMPUS_MOBILE_API_TOKEN"
}

python -m campus_agent.mobile_api --host $ListenAddress --port $ServerPort
