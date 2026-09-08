"""Total internal partition sums Q(T), from HITRAN's TIPS reference tables.

Q(T) rescales HITRAN's 296 K line intensities to the layer temperature (the
``Q(296)/Q(T)`` factor in the standard HITRAN intensity relation -- see
:func:`mwir_atmos.optics.lbl.absorption_cross_section`).

Source: HITRAN TIPS-2021 (Gamache et al., "Total internal partition sums for
167 isotopologues of 53 molecules of astrophysical interest", JQSRT 271
(2021) 107492), served by HITRANonline as one file per isotopologue at
``https://hitran.org/data/Q/q<global_id>.txt`` -- two whitespace-separated
columns, integer T in Kelvin starting at 1 K and Q(T). These were fetched
once during development (2026-08-02) for the 25 isotopologues listed in
:data:`mwir_atmos.hitran.fetch.GLOBAL_ISOTOPOLOGUES` and consolidated onto a
common 70-3000 K grid in ``data/hitran/tips_qtable.npz`` (arrays
``global_ids`` (25,), ``temperature_k`` (2931,), ``q_table`` (25, 2931)).
Spot check: Q(296 K) for H2O-161 (global id 1) came back 174.581 and for
CO2-626 (global id 7) 286.094, matching the commonly-cited reference values
for those isotopologues, which is good evidence the fetched table is sound.

This module **never touches the network** -- it only reads the shipped
``.npz``, satisfying CONTRACT.md's offline-runtime invariant. To regenerate
the cache, re-download each ``q<global_id>.txt`` above for the IDs in
:data:`mwir_atmos.hitran.fetch.GLOBAL_ISOTOPOLOGUES` and rebuild the arrays
described above (there is deliberately no fetch code path in this module).

Native HITRAN tables do not all reach 3000 K (O3's five isotopologues top
out at 1000 K, CH4 at 2500 K, CO2-628/627 at 3500 K; H2O, N2O and CO reach
5000-9000 K). Above an isotopologue's native ceiling, the cached table holds
the last tabulated value constant (flat extrapolation) rather than
omitting it. This is a simplification that only matters far outside the
~150-330 K range relevant to atmospheric transmittance -- flagged here
rather than hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fetch import GLOBAL_ISOTOPOLOGUES, MOLECULE_IDS

__all__ = ["partition_function"]

from ..data_paths import hitran_data_dir

_QTABLE_PATH = hitran_data_dir() / "tips_qtable.npz"

#: (molecule_id, local_iso_id) -> HITRAN global isotopologue ID, derived from
#: the same table :mod:`mwir_atmos.hitran.fetch` uses to fetch line data, so
#: the two stay in lockstep by construction.
_GLOBAL_ID_BY_MOLECULE_LOCAL: dict[tuple[int, int], int] = {
    (MOLECULE_IDS[species], local_iso_id): global_id
    for species, rows in GLOBAL_ISOTOPOLOGUES.items()
    for global_id, local_iso_id, _abundance, _mass in rows
}

_table_cache: dict[str, np.ndarray] | None = None


def _load_table() -> dict[str, np.ndarray]:
    global _table_cache
    if _table_cache is None:
        if not _QTABLE_PATH.exists():
            raise FileNotFoundError(
                f"TIPS partition-sum cache not found at {_QTABLE_PATH}. This "
                "ships with the repo; see the mwir_atmos.hitran.tips module "
                "docstring for how it was built."
            )
        with np.load(_QTABLE_PATH) as data:
            _table_cache = {
                "global_ids": np.array(data["global_ids"]),
                "temperature_k": np.array(data["temperature_k"]),
                "q_table": np.array(data["q_table"]),
            }
    return _table_cache


def partition_function(
    molecule_id: int,
    local_iso_id: int,
    temperature_k: float | np.ndarray,
) -> float | np.ndarray:
    """Total internal partition sum Q(T), 70-3000 K.

    Linear interpolation on the cached 1 K-native 70-3000 K grid (see
    module docstring). Requests outside that range are clamped to the
    nearest edge rather than extrapolated or raising, since Q(T) is smooth
    and this package never evaluates far outside atmospheric temperatures.

    Raises ``KeyError`` if ``(molecule_id, local_iso_id)`` is not one of the
    isotopologues in :data:`mwir_atmos.hitran.fetch.GLOBAL_ISOTOPOLOGUES`.
    """
    key = (int(molecule_id), int(local_iso_id))
    try:
        global_id = _GLOBAL_ID_BY_MOLECULE_LOCAL[key]
    except KeyError as exc:
        raise KeyError(
            f"no cached TIPS partition function for molecule_id={molecule_id}, "
            f"local_iso_id={local_iso_id}"
        ) from exc

    table = _load_table()
    row_matches = np.nonzero(table["global_ids"] == global_id)[0]
    row = int(row_matches[0])
    q_of_t = table["q_table"][row]
    t_grid = table["temperature_k"]

    t = np.asarray(temperature_k, dtype=np.float64)
    t_clamped = np.clip(t, t_grid[0], t_grid[-1])
    result = np.interp(t_clamped, t_grid, q_of_t)
    return float(result) if np.ndim(temperature_k) == 0 else result
