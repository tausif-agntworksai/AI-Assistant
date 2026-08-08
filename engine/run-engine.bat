@echo off
REM Start the JARVIS engine. Any arguments are passed through to the CLI,
REM e.g.  run-engine.bat --text "chrome kholo"
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup first:
    echo     powershell -ExecutionPolicy Bypass -File setup.ps1
    exit /b 1
)

".venv\Scripts\python.exe" -m jarvis %*
