@echo off
rem ==============================================================================
rem MODTRAN-at-home Offline / Air-Gapped Installer for Windows (cmd)
rem ==============================================================================

setlocal enabledelayedexpansion

echo ======================================================================
echo     MODTRAN-at-home (mwir_atmos) Windows Offline Installation
echo ======================================================================

rem 1. Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [-] ERROR: Python is not found in PATH. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

rem 2. Check offline wheels and data
if not exist "wheels\*.whl" (
    echo [-] ERROR: Offline wheels\ directory is missing or empty.
    pause
    exit /b 1
)

if not exist "data\hitran\*.npz" (
    echo [-] ERROR: Offline data\hitran\ directory is missing or empty.
    pause
    exit /b 1
)

echo [+] Verified offline wheel cache and HITRAN spectroscopic data.

rem 3. Setup Virtual Environment
if not exist "venv" (
    echo [+] Creating virtual environment in .\venv...
    python -m venv venv
) else (
    echo [+] Using existing virtual environment in .\venv
)

call venv\Scripts\activate.bat

rem 4. Install dependencies strictly offline
echo [+] Installing dependencies offline from local .\wheels...
python -m pip install --no-index --find-links .\wheels --upgrade pip setuptools wheel
python -m pip install --no-index --find-links .\wheels -e ".[gui,dev]"

if %ERRORLEVEL% NEQ 0 (
    echo [-] Installation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo [+] Successfully installed mwir_atmos.

rem 5. Run verification tests
echo.
echo ======================================================================
echo     Running Offline Verification Test Suite
echo ======================================================================
pytest tests\ -v

echo.
echo ======================================================================
echo     Installation & Verification Complete!
echo ======================================================================
echo To use mwir_atmos:
echo   1. Activate environment:   call venv\Scripts\activate.bat
echo   2. Run CLI commands:       mwir --help
echo   3. Launch Web GUI:         streamlit run mwir_atmos\gui\app.py
echo ======================================================================
