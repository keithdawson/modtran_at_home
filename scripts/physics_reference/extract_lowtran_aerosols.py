"""Extract offline aerosol data from the pinned, unmodified LOWTRAN 7 archive.

Usage: python scripts/physics_reference/extract_lowtran_aerosols.py PATH_TO_CLONE
The reference clone is https://github.com/space-physics/lowtran at COMMIT below.
No network access is performed. Also extracts original routines for validation.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

import numpy as np

COMMIT = "5b8d38714b204eaa12595f5f15e2babec1dbdfb5"
ROOT = Path(__file__).resolve().parents[2]


def extract(clone: Path) -> None:
    revision = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Expected {COMMIT}, got {revision}")
    source = clone / "src/lowtran/fortran/lowtran7.f"
    raw = source.read_bytes()
    # Refuse local edits as well as a different revision.
    original = subprocess.check_output(["git", "-C", str(clone), "show", "HEAD:src/lowtran/fortran/lowtran7.f"])
    if raw.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
        raise ValueError("Reference source has local modifications")
    lines = raw.decode("ascii").splitlines()
    statements = "\n".join(line[6:72] for line in lines if line and line[0] not in "Cc*!")
    arrays = {}
    names = ["VX2", "ZHT", "FAWI50", "FAWI23", "SPSU50", "SPSU23", "BASTFW", "BASTSS", "UPNATM"]
    names += [f"{p}{q}{r}" for p in ("RUR", "URB", "OCN", "TRO") for q in "EAG" for r in range(1, 5)]
    names += [p + q for p in ("BST", "DME") for q in ("EXT", "ABS", "SYM")]
    for name in names:
        match = re.search(r"\bDATA\s+" + name + r"\s*/([^/]+)/", statements, re.I)
        if match is None:
            raise ValueError(f"Missing {name}")
        values = []
        for token in match[1].replace("\n", "").split(","):
            parts = token.strip().split("*")
            values.extend([float(parts[-1])] * (int(parts[0]) if len(parts) == 2 else 1))
        arrays[name] = np.array(values, dtype=np.float64)
        assert len(values) == (34 if name in names[:9] and name != "VX2" else 47), name
    rows = re.findall(r"\bDATA\s+HZ2K\([^/]+/([^/]+)/", statements)
    arrays["HZ2K"] = np.array([[float(v) for v in row.split(",")] for row in rows])
    assert arrays["HZ2K"].shape == (3, 5)
    dest = ROOT / "data/aerosol"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "lowtran7.npz", **arrays)
    shutil.copyfile(clone / "LICENSE.txt", dest / "LOWTRAN_LICENSE.txt")
    metadata = dict(source="https://github.com/space-physics/lowtran", commit=COMMIT,
                    source_file="src/lowtran/fortran/lowtran7.f",
                    source_sha256_lf=hashlib.sha256(original.replace(b"\r\n", b"\n")).hexdigest(),
                    release="LOWTRAN 7, revision 4.2, February 1992",
                    units="VX2: micrometers; ZHT: km MSL; profiles: km^-1 at 550 nm; E/A: normalized to extinction at 550 nm; G: dimensionless",
                    humidity_percent=[0, 70, 80, 99])
    (dest / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    chunks = []
    for kind, name in [("SUBROUTINE", "EXABIN"), ("SUBROUTINE", "AEREXT"), ("SUBROUTINE", "AERPRF"), ("BLOCK DATA", "EXTDTA"), ("BLOCK DATA", "PRFDTA")]:
        start = next(i for i, line in enumerate(lines) if line.startswith(f"      {kind} {name}"))
        end = next(i for i in range(start + 1, len(lines)) if re.match(r"^      END(?:\s|$)", lines[i]))
        chunks.extend(lines[start:end + 1])
    ref = ROOT / "artifacts/lowtran-reference-kernel"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "original.f").write_text("\n".join(chunks) + "\n", encoding="ascii")
    print(f"Extracted {len(arrays)} arrays and original reference routines to {dest}")


if __name__ == "__main__":
    extract(Path(sys.argv[1]))
