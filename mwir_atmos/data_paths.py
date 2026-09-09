"""Locate scientific tables in a source checkout or a standard wheel install."""
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError
import sys


def aerosol_data_dir() -> Path:
    return _data_dir("aerosol")


def hitran_data_dir() -> Path:
    return _data_dir("hitran")


def _data_dir(name: str) -> Path:
    source = Path(__file__).resolve().parents[1] / "data" / name
    if source.is_dir():
        return source
    # Wheel RECORD paths resolve the actual install scheme, including user
    # installs and Linux distributions whose data prefix differs from sys.prefix.
    try:
        package = distribution("mwir-atmos")
        for item in package.files or ():
            if f"share/mwir-atmos/{name}/" in str(item).replace("\\", "/"):
                directory = Path(package.locate_file(item)).parent
                if directory.is_dir():
                    return directory
    except PackageNotFoundError:
        pass
    return Path(sys.prefix) / "share" / "mwir-atmos" / name
