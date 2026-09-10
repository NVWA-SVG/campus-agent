@echo off
setlocal
powershell.exe -NoProfile -File "%~dp0start-campus-agent-mobile.ps1" %*
exit /b %ERRORLEVEL%
