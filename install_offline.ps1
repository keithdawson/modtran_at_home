# ==============================================================================
# MODTRAN-at-home Offline / Air-Gapped Installer for Windows (PowerShell)
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "    MODTRAN-at-home (mwir_atmos) Windows Offline Installation" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python
try {
    $pyVersion = python --version
    Write-Host "[+] Found $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[-] ERROR: Python is not found in PATH." -ForegroundColor Red
    Exit 1
}

# 2. Check offline wheels and data
if (-not (Test-Path "wheels\*.whl")) {
    Write-Host "[-] ERROR: Offline wheels\ directory is missing or empty." -ForegroundColor Red
    Exit 1
}

if (-not (Test-Path "data\hitran\*.npz")) {
    Write-Host "[-] ERROR: Offline data\hitran\ directory is missing or empty." -ForegroundColor Red
    Exit 1
}

Write-Host "[+] Verified offline wheel cache and HITRAN spectroscopic data." -ForegroundColor Green

# 3. Setup Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "[+] Creating virtual environment in .\venv..." -ForegroundColor Yellow
    Invoke-Python -m venv venv
} else {
    Write-Host "[+] Using existing virtual environment in .\venv" -ForegroundColor Green
}

& ".\venv\Scripts\Activate.ps1"

# 4. Install dependencies strictly offline
Write-Host "[+] Installing dependencies offline from local .\wheels..." -ForegroundColor Yellow
Invoke-Python -m pip install --no-index --find-links .\wheels --upgrade pip setuptools wheel
Invoke-Python -m pip install --no-index --find-links .\wheels -e ".[gui,dev]"

Write-Host "[+] Successfully installed mwir_atmos in offline mode." -ForegroundColor Green

# 5. Run verification tests
Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "    Running Offline Verification Test Suite" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Invoke-Python -m pytest tests\ -v

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "    Installation & Verification Complete!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To use mwir_atmos:"
Write-Host "  1. Activate environment:   .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run CLI commands:       mwir --help"
Write-Host "  3. Launch Web GUI:         streamlit run mwir_atmos\webgui\app.py"
Write-Host "======================================================================"
