@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================
echo     OMNICHAT LIVE DEMO
echo     MiniCPM-o 4.5 Capabilities Showcase
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

:: Pass any arguments through (e.g. --headless, --acts 1,3, --strict)
echo Starting demo...
echo.
"%PYTHON%" -m demos.run_demo %*

if errorlevel 1 (
    echo.
    echo Demo finished with failures. Check output above.
    pause
)
endlocal
