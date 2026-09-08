"""Locate scientific tables in a source checkout or a standard wheel install."""
from pathlib import Path
import sys


def hitran_data_dir() -> Path:
    source = Path(__file__).resolve().parents[1] / "data" / "hitran"
    if source.is_dir():
        return source
    return Path(sys.prefix) / "share" / "mwir-atmos" / "hitran"
