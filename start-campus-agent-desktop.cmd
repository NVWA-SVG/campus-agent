@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CAMPUS_AGENT_PROJECT_ROOT=%CD%"
if not defined CAMPUS_BUSINESS_API_MODE set "CAMPUS_BUSINESS_API_MODE=mock"

if exist ".campus_agent_data\models" (
    if not defined CAMPUS_EMBEDDING_PROVIDER set "CAMPUS_EMBEDDING_PROVIDER=sentence-transformers"
    if not defined CAMPUS_EMBEDDING_MODEL_CACHE_DIR set "CAMPUS_EMBEDDING_MODEL_CACHE_DIR=%CD%\.campus_agent_data\models"
    if not defined CAMPUS_EMBEDDING_LOCAL_ONLY set "CAMPUS_EMBEDDING_LOCAL_ONLY=true"
) else (
    if not defined CAMPUS_EMBEDDING_PROVIDER set "CAMPUS_EMBEDDING_PROVIDER=hashing"
)

if exist "dist\CampusAgent\CampusAgent.exe" (
    set "CAMPUS_AGENT_PROJECT_ROOT=%CD%\dist\CampusAgent"
    pushd "dist\CampusAgent"
    CampusAgent.exe
    set "CAMPUS_EXIT_CODE=%ERRORLEVEL%"
    popd
    exit /b %CAMPUS_EXIT_CODE%
)

if exist "desktop-rust\target\release\campus-agent-desktop.exe" (
    "desktop-rust\target\release\campus-agent-desktop.exe"
    exit /b %ERRORLEVEL%
)

where cargo.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Rust/Cargo was not found and no compiled desktop EXE exists.
    echo Run .\scripts\build_desktop.ps1 after installing Rustup.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found: .venv
    pause
    exit /b 1
)

cargo run --manifest-path "desktop-rust\Cargo.toml" --release
exit /b %ERRORLEVEL%

