"""Download pip wheels for air-gapped deployment on Linux and Windows.

Downloads all wheels required for core, gui, and test environments without installing them,
placing them into the local `wheels/` directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Requirements list
REQUIREMENTS = [
    "numpy>=2.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0",
    "streamlit>=1.49",
    "plotly>=5.18.0",
    "matplotlib>=3.7.0",
    "pytest>=7.4.0",
    "pytest-xdist>=3.5",
    "setuptools>=68.0.0",
    "wheel",
    "pip",
]

#: GPU backend for the work server (Linux, CUDA 13 driver): the
#: pyproject "cuda13" extra. Only the CUDA libraries mwir_atmos uses (runtime,
#: NVRTC, cuBLAS) -- cupy's own [ctk] extra adds ~600 MB of cuFFT, cuRAND,
#: cuSOLVER and cuSPARSE that nothing here calls.
GPU_REQUIREMENTS = [
    "cupy-cuda13x>=14.2",
    "cuda-toolkit[cublas,cudart,nvrtc]==13.*",
]
#: Rocky Linux 9 has glibc 2.34; the NVIDIA wheels are tagged up to manylinux_2_27.
GPU_PLATFORMS = ["manylinux2014_x86_64", "manylinux_2_27_x86_64", "manylinux_2_28_x86_64",
                 "manylinux_2_34_x86_64"]


def download_gpu_wheels(target_dir: Path, py_version: str) -> None:
    print(f"\n[+] Fetching CUDA 13 GPU wheels for Linux, Python {py_version}")
    cmd = [sys.executable, "-m", "pip", "download", "--dest", str(target_dir),
           "--python-version", py_version, "--only-binary=:all:",
           "--implementation", "cp", "--abi", f"cp{py_version}"]
    for plat in GPU_PLATFORMS:
        cmd += ["--platform", plat]
    print("Executing:", " ".join(cmd + GPU_REQUIREMENTS))
    subprocess.run(cmd + GPU_REQUIREMENTS, check=True)


def download_wheels(target_dir: Path, platforms: list[str], py_version: str = "311"):
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading wheels to {target_dir.resolve()} for Python {py_version}...")

    for plat in platforms:
        print(f"\n[+] Fetching wheels for platform: {plat}")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(target_dir),
            "--python-version",
            py_version,
            "--only-binary=:all:",
        ]

        if plat == "current":
            cmd.extend(REQUIREMENTS)
        else:
            cmd.extend([
                "--platform",
                plat,
                "--implementation",
                "cp",
                "--abi",
                f"cp{py_version}",
            ] + REQUIREMENTS)

        print("Executing:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download air-gapped dependency wheels.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("wheels"),
        help="Destination directory for wheels (default: wheels/)",
    )
    parser.add_argument(
        "--py-version",
        type=str,
        default="311",
        help="Target Python version tag, e.g. 311 for 3.11 (default: 311)",
    )
    parser.add_argument(
        "--all-platforms",
        action="store_true",
        help="Download for both current OS and manylinux2014_x86_64",
    )
    args = parser.parse_args()

    platforms = ["current"]
    if args.all_platforms:
        platforms.append("manylinux2014_x86_64")
        platforms.append("win_amd64")

    download_wheels(args.dest, platforms, args.py_version)
    if args.all_platforms:
        download_gpu_wheels(args.dest, args.py_version)
    print(f"\n[OK] Finished downloading wheels. Total files in {args.dest}: {len(list(args.dest.glob('*.whl')))}")


if __name__ == "__main__":
    main()
