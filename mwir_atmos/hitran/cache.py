"""LineList container and ``.npz`` persistence for cached HITRAN line data.

A :class:`LineList` holds one molecule's spectral lines, pooled across the
isotopologues selected in :mod:`mwir_atmos.hitran.fetch`. On disk it is one
compressed ``.npz`` per species under ``data/hitran/`` -- the format the
finished, offline package ships and loads at runtime (CONTRACT.md invariant
5: the final package must run with no network).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from ..types import Species

__all__ = ["LineList", "save_linelist", "load_linelist"]

PathLike = Union[str, "os.PathLike[str]"]

#: Array-valued fields, in HITRAN units -- see the dataclass docstring below.
_ARRAY_FIELDS = (
    "nu_cm1",
    "s296",
    "gamma_air",
    "gamma_self",
    "e_lower",
    "n_air",
    "delta_air",
    "local_iso_id",
    "iso_abundance",
)

#: Fields that are stored/loaded as plain integer arrays rather than float64.
_INT_ARRAY_FIELDS = ("local_iso_id",)


@dataclass
class LineList:
    """One molecule's HITRAN spectral lines, across selected isotopologues.

    Every array has shape ``(n_lines,)``, row-aligned across fields but in
    no particular sort order (lines are pooled across isotopologues as
    HITRAN returns them; sort by ``nu_cm1`` explicitly if you need that).

    ``s296`` is HITRAN's ``sw`` -- line intensity at the 296 K HITRAN
    reference temperature, cm^-1/(molecule cm^-2). Per HITRAN convention
    this is **already weighted by the terrestrial abundance** of
    ``local_iso_id`` (confirmed against
    https://hitran.org/docs/jpl-cdms-conversion/, which states that the
    JPL/CDMS catalogs assume 100% abundance per isotopologue "whereas the
    HITRAN database incorporates a terrestrial abundance scaling", i.e. you
    must *multiply* a JPL/CDMS intensity by the HITRAN abundance to match
    HITRAN, meaning HITRAN's own ``sw`` already has that factor baked in).
    Consequently, optical depth is ``sum_lines(cross_section) *
    column_cm2[species]`` where ``column_cm2[species]`` is the *total*
    column of the species (all natural isotopologues combined) -- do
    **not** multiply by ``iso_abundance`` again, that would double-count
    the abundance factor. ``iso_abundance`` is carried for diagnostics only
    (e.g. isolating a single isotopologue's contribution).
    """

    species: Species
    molecule_id: int                 # HITRAN molecule number
    nu_cm1: np.ndarray               # line centre, cm^-1 (vacuum)
    s296: np.ndarray                 # intensity @296 K, cm^-1/(molecule cm^-2)
    gamma_air: np.ndarray            # air-broadened HWHM @296 K, 1 atm, cm^-1/atm
    gamma_self: np.ndarray           # self-broadened HWHM, cm^-1/atm
    e_lower: np.ndarray              # lower-state energy, cm^-1
    n_air: np.ndarray                # T exponent of gamma_air
    delta_air: np.ndarray            # air pressure shift, cm^-1/atm
    local_iso_id: np.ndarray         # HITRAN local isotopologue number
    iso_abundance: np.ndarray        # terrestrial abundance of each line's isotopologue

    def __len__(self) -> int:
        return int(self.nu_cm1.size)


def save_linelist(linelist: LineList, path: PathLike) -> None:
    """Persist a :class:`LineList` as a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in _ARRAY_FIELDS:
        dtype = np.int64 if name in _INT_ARRAY_FIELDS else np.float64
        arrays[name] = np.asarray(getattr(linelist, name), dtype=dtype)
    np.savez_compressed(
        path,
        species=np.asarray(linelist.species.value),
        molecule_id=np.asarray(linelist.molecule_id, dtype=np.int64),
        **arrays,
    )


def load_linelist(path: PathLike) -> LineList:
    """Load a :class:`LineList` previously written by :func:`save_linelist`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        species = Species(str(data["species"]))
        molecule_id = int(data["molecule_id"])
        kwargs = {name: np.array(data[name]) for name in _ARRAY_FIELDS}
    return LineList(species=species, molecule_id=molecule_id, **kwargs)
