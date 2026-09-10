@echo off
setlocal
powershell.exe -NoProfile -File "%~dp0start-campus-agent-server.ps1" %*
exit /b %ERRORLEVEL%
