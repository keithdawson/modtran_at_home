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
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0",
    "streamlit>=1.30.0",
    "plotly>=5.18.0",
    "matplotlib>=3.7.0",
    "pytest>=7.4.0",
    "setuptools>=68.0.0",
    "wheel",
]


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
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[-] Warning: Some wheels for {plat} could not be downloaded as binary. Falling back...")


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
    print(f"\n[OK] Finished downloading wheels. Total files in {args.dest}: {len(list(args.dest.glob('*.whl')))}")


if __name__ == "__main__":
    main()
