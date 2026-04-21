@echo off
setlocal


endlocal

echo Starting platform GUI...
%FWSW_APPS%/uv/bin/uv run e201

if errorlevel 1 (
    echo [ERROR] GUI failed to start
    echo [INFO] Preparing environment...
    call "%~dp0prepare_venv.bat"

    %FWSW_APPS%\uv\bin\uv run e201
)

pause
