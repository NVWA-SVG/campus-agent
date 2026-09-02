@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "CAMPUS_PYTHON=%CD%\.venv\Scripts\python.exe"
set "CAMPUS_PORT=8000"
set "CAMPUS_HOST=127.0.0.1"
set "CAMPUS_BROWSER_URL=http://127.0.0.1:%CAMPUS_PORT%"
if not defined CAMPUS_WEB_ALLOWED_HOSTS set "CAMPUS_WEB_ALLOWED_HOSTS=127.0.0.1,localhost"

if /I not "%~1"=="lan" goto after_lan_config
set "CAMPUS_HOST=0.0.0.0"
for /f "tokens=2 delims=:" %%I in ('ipconfig ^| findstr /C:"IPv4"') do (
    set "CAMPUS_LAN_IP=%%I"
    set "CAMPUS_LAN_IP=!CAMPUS_LAN_IP: =!"
    if defined CAMPUS_LAN_IP set "CAMPUS_WEB_ALLOWED_HOSTS=!CAMPUS_WEB_ALLOWED_HOSTS!,!CAMPUS_LAN_IP!"
)
:after_lan_config

if not exist "%CAMPUS_PYTHON%" (
    echo [ERROR] Python virtual environment was not found: .venv
    echo.
    echo Run these commands first:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -e ".[semantic,dev]"
    echo.
    pause
    exit /b 1
)

if not exist ".campus_agent_data\models" (
    echo [ERROR] Local BGE model directory was not found: .campus_agent_data\models
    echo.
    echo Prepare the model first:
    echo   .\.venv\Scripts\python.exe -m scripts.prepare_embedding_model --allow-download --model-cache-dir .campus_agent_data\models
    echo.
    pause
    exit /b 1
)

set "CAMPUS_EMBEDDING_PROVIDER=sentence-transformers"
set "CAMPUS_EMBEDDING_MODEL_CACHE_DIR=%CD%\.campus_agent_data\models"
set "CAMPUS_EMBEDDING_LOCAL_ONLY=true"
set "CAMPUS_EMBEDDING_MINIMUM_SIMILARITY=0.48"
set "CAMPUS_BUSINESS_API_MODE=mock"

echo ============================================================
echo Starting Campus Agent
echo Local URL: %CAMPUS_BROWSER_URL%
if /I "%~1"=="lan" (
    echo LAN mode: listening on 0.0.0.0:%CAMPUS_PORT%
    echo Other devices can use: http://YOUR_IPV4_ADDRESS:%CAMPUS_PORT%
    echo Run ipconfig in another terminal to find your IPv4 address.
    echo Trusted hosts: !CAMPUS_WEB_ALLOWED_HOSTS!
    echo WARNING: Only use LAN mode on a trusted private network.
)
echo Press Ctrl+C to stop the server.
echo ============================================================

if not defined CAMPUS_SKIP_BROWSER start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "$url='%CAMPUS_BROWSER_URL%'; for ($i=0; $i -lt 60; $i++) { try { $response=Invoke-WebRequest -UseBasicParsing -Uri ($url + '/api/health') -TimeoutSec 1; if ($response.StatusCode -eq 200) { Start-Process $url; break } } catch {}; Start-Sleep -Milliseconds 500 }"

"%CAMPUS_PYTHON%" -m campus_agent.web --host %CAMPUS_HOST% --port %CAMPUS_PORT%
set "CAMPUS_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%CAMPUS_EXIT_CODE%"=="0" echo Campus Agent exited with code %CAMPUS_EXIT_CODE%.
if "%CAMPUS_EXIT_CODE%"=="0" echo Campus Agent stopped.
pause
exit /b %CAMPUS_EXIT_CODE%
