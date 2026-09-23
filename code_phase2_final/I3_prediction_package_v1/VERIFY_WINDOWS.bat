@echo off
cd /d "%~dp0"
python -B verify_package.py
if errorlevel 1 exit /b 1
