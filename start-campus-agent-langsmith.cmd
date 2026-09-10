@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-campus-agent-langsmith.ps1" %*
exit /b %ERRORLEVEL%
