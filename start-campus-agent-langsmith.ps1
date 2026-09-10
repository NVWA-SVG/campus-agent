[CmdletBinding()]
param(
    [ValidateSet("us", "eu", "apac", "aws-us")]
    [string]$Region = "us",

    [ValidatePattern("^[A-Za-z0-9._-]{1,80}$")]
    [string]$Project = "campus-agent-dev",

    [string]$WorkspaceId = "",

    [switch]$Lan,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$baseLauncher = Join-Path $projectRoot "start-campus-agent.cmd"

if (-not (Test-Path -LiteralPath $baseLauncher -PathType Leaf)) {
    throw "Base launcher was not found: $baseLauncher"
}

$endpoints = @{
    "us" = "https://api.smith.langchain.com"
    "eu" = "https://eu.api.smith.langchain.com"
    "apac" = "https://apac.api.smith.langchain.com"
    "aws-us" = "https://aws.api.smith.langchain.com"
}

$apiKey = [Environment]::GetEnvironmentVariable(
    "LANGSMITH_API_KEY",
    [EnvironmentVariableTarget]::Process
)
$secureKey = $null
$credential = $null

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host "Paste a NEW LangSmith API key. Input is hidden and is not saved to disk."
    $secureKey = Read-Host "LangSmith API key" -AsSecureString
    $credential = [PSCredential]::new("langsmith", $secureKey)
    $apiKey = $credential.GetNetworkCredential().Password
}

if ([string]::IsNullOrWhiteSpace($apiKey) -or $apiKey.Length -lt 20) {
    throw "A non-empty LangSmith API key is required."
}

$managedVariables = @(
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
    "CAMPUS_SKIP_BROWSER"
)
$previousValues = @{}
foreach ($name in $managedVariables) {
    $previousValues[$name] = [Environment]::GetEnvironmentVariable(
        $name,
        [EnvironmentVariableTarget]::Process
    )
}

$exitCode = 1
try {
    $env:LANGSMITH_TRACING = "true"
    $env:LANGSMITH_API_KEY = $apiKey
    $env:LANGSMITH_PROJECT = $Project
    $env:LANGSMITH_ENDPOINT = $endpoints[$Region]

    if ([string]::IsNullOrWhiteSpace($WorkspaceId)) {
        Remove-Item Env:LANGSMITH_WORKSPACE_ID -ErrorAction SilentlyContinue
    } else {
        $env:LANGSMITH_WORKSPACE_ID = $WorkspaceId
    }

    if ($NoBrowser) {
        $env:CAMPUS_SKIP_BROWSER = "1"
    }

    Write-Host ""
    Write-Host "LangSmith tracing enabled for this process only."
    Write-Host "Project : $Project"
    Write-Host "Region  : $Region"
    Write-Host "Endpoint: $($endpoints[$Region])"
    Write-Host "Dashboard: https://smith.langchain.com/"
    Write-Host "The API key will not be printed or written to the project."
    Write-Host ""

    if (-not $NoBrowser) {
        Start-Process "https://smith.langchain.com/"
    }

    $launcherArguments = @()
    if ($Lan) {
        $launcherArguments += "lan"
    }
    & $baseLauncher @launcherArguments
    $exitCode = $LASTEXITCODE
} finally {
    foreach ($name in $managedVariables) {
        $previousValue = $previousValues[$name]
        if ($null -eq $previousValue) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable(
                $name,
                $previousValue,
                [EnvironmentVariableTarget]::Process
            )
        }
    }
    $apiKey = $null
    $credential = $null
    $secureKey = $null
}

exit $exitCode
