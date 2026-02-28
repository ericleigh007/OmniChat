@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================
echo     OMNICHAT — Multimodal Voice Assistant
echo ============================================
echo.

:: Use the project-local virtual environment
set PYTHON=%~dp0.venv\Scripts\python.exe

:: Verify Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo.
    echo Create a virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Run setup script (idempotent — safe to run every time)
echo Running setup check...
"%PYTHON%" setup.py
if errorlevel 1 (
    echo.
    echo Setup encountered errors. See messages above.
    pause
    exit /b 1
)

echo.
echo Launching OmniChat...
echo.
"%PYTHON%" main.py %*

if errorlevel 1 (
    echo.
    echo App exited with an error. Check messages above.
    pause
)
endlocal
