"""Create a self-contained, air-gapped distribution archive (.tar.gz and .zip).

Packages:
- mwir_atmos/ source code
- data/hitran/ offline line lists and MT_CKD continuum tables
- wheels/ pre-downloaded binary wheels for Linux (manylinux x86_64) and Windows (win_amd64)
- scenarios/ example input scenarios
- tests/ full verification test suite
- install_offline.sh, install_offline.bat, install_offline.ps1
- Dockerfile.rocky9, Dockerfile.ubuntu, docker-compose.yml
- Documentation (README.md, CONTRACT.md, OFFLINE_DEPLOYMENT.md, pyproject.toml)
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT_DIR / "dist"
BUNDLE_NAME = "modtran_at_home_airgap_v0.1.0"

INCLUDE_PATTERNS = [
    "mwir_atmos",
    "data",
    "wheels",
    "scenarios",
    "tests",
    "Dockerfile.rocky9",
    "Dockerfile.ubuntu",
    "docker-compose.yml",
    "install_offline.sh",
    "install_offline.bat",
    "install_offline.ps1",
    "pyproject.toml",
    "README.md",
    "CONTRACT.md",
    "OFFLINE_DEPLOYMENT.md",
    "SERVER CONFIGURATION.txt",
]


def build_bundle():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DIST_DIR / f"{BUNDLE_NAME}.tar.gz"
    zip_path = DIST_DIR / f"{BUNDLE_NAME}.zip"

    print(f"[*] Packaging air-gap bundle from {ROOT_DIR}...")

    # Filter out cache files and bytecode
    def tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = tarinfo.name
        if "__pycache__" in name or name.endswith((".pyc", ".pyo", ".pytest_cache")):
            return None
        return tarinfo

    # 1. Build .tar.gz
    print(f"[+] Creating {tar_path.name}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in INCLUDE_PATTERNS:
            item_path = ROOT_DIR / item
            if item_path.exists():
                tar.add(item_path, arcname=f"{BUNDLE_NAME}/{item}", filter=tar_filter)

    # 2. Build .zip
    print(f"[+] Creating {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDE_PATTERNS:
            item_path = ROOT_DIR / item
            if item_path.is_file():
                zipf.write(item_path, arcname=f"{BUNDLE_NAME}/{item}")
            elif item_path.is_dir():
                for p in item_path.rglob("*"):
                    if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                        continue
                    zipf.write(p, arcname=f"{BUNDLE_NAME}/{item}/{p.relative_to(item_path)}")

    tar_mb = tar_path.stat().st_size / (1024 * 1024)
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Air-gap distribution packages built successfully:")
    print(f"  - {tar_path.name} ({tar_mb:.1f} MB)")
    print(f"  - {zip_path.name} ({zip_mb:.1f} MB)")


if __name__ == "__main__":
    build_bundle()
