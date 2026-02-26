@echo off
call ../../../.venv/Scripts/activate.bat

call pyuic5 -o ui_template.py ui_template.ui

call ../../../.venv/Scripts/deactivate
echo UI converted!
pause