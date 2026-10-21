#!/usr/bin/env bash
# ==============================================================================
# MODTRAN-at-home Offline / Air-Gapped Installer for Linux
# Supported OS: Rocky Linux 8/9, RHEL 8/9, Ubuntu 20.04/22.04/24.04, Debian 11/12
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "    MODTRAN-at-home (mwir_atmos) Offline Installation Suite"
echo "======================================================================"

# 1. Detect Python version
PYTHON_BIN=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &> /dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ] && [ "$MINOR" -le 12 ]; then
            PYTHON_BIN="$cmd"
            echo "[+] Found compatible Python: $cmd (v$VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[-] ERROR: Python 3.11 or 3.12 is required (the bundled wheels are built for them)."
    echo "    On Rocky Linux / RHEL: sudo dnf install python3.11 python3.11-pip"
    echo "    On Ubuntu / Debian:   sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# 2. Check offline wheels and data
if [ ! -d "wheels" ] || [ -z "$(ls -A wheels/*.whl 2>/dev/null)" ]; then
    echo "[-] ERROR: Local wheels/ directory is missing or empty."
    exit 1
fi

if [ ! -d "data/hitran" ] || [ -z "$(ls -A data/hitran/*.npz 2>/dev/null)" ]; then
    echo "[-] ERROR: HITRAN offline data cache in data/hitran/ is missing or empty."
    exit 1
fi

echo "[+] Verified offline wheel cache and HITRAN spectroscopic data."

# 3. Create or reuse virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creating virtual environment in ./$VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[+] Using existing virtual environment in ./$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# 4. Install dependencies strictly offline
echo "[+] Installing dependencies strictly offline from local ./wheels..."
pip install --no-index --find-links ./wheels --upgrade pip setuptools wheel
EXTRAS="gui,dev"
# GPU backend: when an NVIDIA driver is present and the bundle carries the
# CUDA 13 wheels (CuPy plus its CUDA runtime, NVRTC and cuBLAS).
if command -v nvidia-smi >/dev/null 2>&1 && ls wheels/cupy_cuda13x-*.whl >/dev/null 2>&1; then
    echo "[+] NVIDIA driver found: installing the CUDA 13 GPU backend (cupy-cuda13x)"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
    EXTRAS="$EXTRAS,cuda13"
else
    echo "[*] No NVIDIA driver (nvidia-smi) found: CPU backend only"
fi
pip install --no-index --find-links ./wheels -e ".[$EXTRAS]"
if [[ "$EXTRAS" == *cuda13* ]]; then
    python -c "from mwir_atmos.optics import lbl_cuda; import sys; sys.exit(0 if lbl_cuda.available() else 1)" \
        && echo "[+] GPU backend available: mwir --backend cuda" \
        || echo "[-] WARNING: cupy-cuda13x installed but no usable CUDA device; CPU backend only"
fi

echo "[+] Successfully installed mwir_atmos in offline mode."

# 5. Run verification tests
echo ""
echo "======================================================================"
echo "    Running Offline Physics & Verification Test Suite"
echo "======================================================================"
pytest tests/ -v

echo ""
echo "======================================================================"
echo "    Installation & Verification Complete!"
echo "======================================================================"
echo "To use mwir_atmos:"
echo "  1. Activate the environment: source ./$VENV_DIR/bin/activate"
echo "  2. Run CLI commands:         mwir --help"
echo "  3. Launch Streamlit Web GUI: streamlit run mwir_atmos/webgui/app.py"
echo "======================================================================"
