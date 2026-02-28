@echo off
setlocal
cd /d "%~dp0.."

echo.
echo ============================================
echo     OMNICHAT QUANTIZATION BENCHMARK
echo     Compare bf16 / int8 / int4 audio quality
echo ============================================
echo.

:: Use the project-local virtual environment
set PYTHON=%~dp0..\.venv\Scripts\python.exe

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

:: Pass any arguments through (e.g. --quants none,int8, --voice-ref path, --temperature 0.3)
echo Starting benchmark...
echo.
"%PYTHON%" -m benchmarks.run_benchmark %*

if errorlevel 1 (
    echo.
    echo Benchmark finished with errors. Check output above.
    pause
)
endlocal
