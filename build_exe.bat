@echo off
setlocal EnableDelayedExpansion

echo.
echo   E201GUI - CREATING EXE FILE
echo.

REM =================================================
REM Clean previous artifacts
REM =================================================

echo Cleaning build folders...

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

REM =================================================
REM Sync environment
REM =================================================

echo.
echo Syncing environment from lockfile...

uv sync --dev --frozen

if %errorlevel% neq 0 (
    echo.
    echo ERROR: uv sync failed.
    exit /b 1
)


REM =================================================
REM Build EXE
REM =================================================

echo.
echo Building executable...
echo.

uv run pyinstaller e201_gui.spec --clean --noconfirm

if %errorlevel% neq 0 (
    echo.
    echo Build FAILED.
    exit /b 1
)

echo.
echo Build SUCCESSFUL.
echo.

pause