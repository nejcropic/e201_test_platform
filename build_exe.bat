@echo off
setlocal EnableDelayedExpansion

echo.
echo   E201GUI - CREATING EXE FILE
echo.

REM 1. Clean previous artifacts
echo Cleaning build folders...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

REM 2. Clean & sync environment
echo Syncing environment from lockfile...
%UV% sync --all-extras --dev --frozen

if %errorlevel% neq 0 (
    echo.
    echo ERROR: uv sync failed.
    exit /b 1
)

REM 3. Ensure PyInstaller

echo Checking PyInstaller...
%UV% run pyinstaller --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    %UV% add --dev pyinstaller
)

REM 4. Build EXE

echo.
echo Building executable...
echo.

%UV% run pyinstaller e201_gui.spec --clean --noconfirm


if %errorlevel% neq 0 (
    echo.
    echo Build FAILED.
    exit /b 1
)

pause