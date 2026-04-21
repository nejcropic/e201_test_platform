@echo off
setlocal

REM Absolute path to this .bat file's directory
set SCRIPT_DIR=%~dp0

"%SCRIPT_DIR%../../../../.venv/Scripts/pyuic5.exe" ^
    "%SCRIPT_DIR%e201_ui_template.ui" ^
    -o "%SCRIPT_DIR%e201_ui_template.py"


