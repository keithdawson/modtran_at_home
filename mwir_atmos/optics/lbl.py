"""Voigt line-by-line molecular absorption engine.

The headline function is :func:`molecular_optical_depth`: given a resolved
path (:class:`~mwir_atmos.types.PathLayer` sequence, e.g. from
:func:`mwir_atmos.testing.make_synthetic_path` today, or
:func:`mwir_atmos.geometry.path.build_path` once dev-beta's geometry lands)
and a :class:`~mwir_atmos.types.SpectralGrid`, it returns per-layer,
per-species optical depth from HITRAN line data plus the H2O continuum.

Physics, all per CONTRACT.md's "dev-alpha" physics requirements:

* **Line shape**: Voigt, via ``scipy.special.wofz`` (the Faddeeva function).
  Not Lorentz-only -- above ~20 km the air pressure collapses and the
  Doppler width dominates the line core, where a pure Lorentz profile is
  visibly wrong (too narrow a core, wrong far-wing falloff).
* **Intensity temperature scaling** (:func:`_line_intensity_at_t`): the full
  HITRAN relation, all three factors --
  ``Q(296)/Q(T)`` (:func:`mwir_atmos.hitran.tips.partition_function`),
  the Boltzmann lower-state population ratio, and the stimulated-emission
  correction. See the HITRAN definitions page
  (https://hitran.org/faq/, https://hitran.org/docs/definitions/) for the
  reference form of this relation.
* **Pressure broadening**: ``gamma_L = (gamma_air*p_air + gamma_self*p_self)
  * (296/T)**n_air``, pressures in atm. ``p_air`` is the *total* layer
  pressure minus the absorber's own partial pressure (collisions with
  "everything else"), not the total pressure -- see
  :func:`absorption_cross_section`.
* **Pressure shift**: line centres are shifted by ``delta_air * p_total``
  before evaluating the profile.
* **Isotopologue abundance**: HITRAN's ``sw`` (``LineList.s296``) is already
  scaled by the terrestrial abundance of its isotopologue -- see the long
  note in :class:`mwir_atmos.hitran.cache.LineList`. This code sums
  per-line cross-sections directly and multiplies by the *total* species
  column (:data:`PathLayer.column_cm2`); it does **not** multiply by
  ``iso_abundance`` again.
* **Doppler width**: uses each line's isotopologue-specific molar mass
  (HDO and H2-18O have measurably different Doppler widths than the main
  isotopologue), not a single species-average mass.

Performance: everything above is vectorised across all lines of a
(layer, species) pair at once using a fixed-width wing window (default
+/-25 cm^-1) and a scatter-add (``numpy.bincount``) onto the output grid --
no per-line Python loop. The only Python-level loop in this module is over
``(layer, species)`` pairs (tens, not tens of thousands, of iterations).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import wofz

from ..constants import C2_RAD, GAS_CONSTANT_R, LIGHT_C, STD_PRESSURE_MBAR
from ..hitran.cache import LineList
from ..hitran.fetch import GLOBAL_ISOTOPOLOGUES, ensure_linelists
from ..hitran.tips import partition_function
from ..types import PathLayer, REQUIRED_SPECIES, Species, SpectralGrid
from .continuum import co2_continuum_optical_depth, h2o_continuum_optical_depth

__all__ = ["absorption_cross_section", "molecular_optical_depth"]

#: HITRAN reference temperature for tabulated line intensities and widths.
_T_REF_K = 296.0


def _isotopologue_mass_g_mol(linelist: LineList) -> np.ndarray:
    """Per-line molar mass (g/mol) of ``linelist.local_iso_id``, for the
    Doppler width -- vectorised fancy-index lookup, no per-line loop."""
    table = GLOBAL_ISOTOPOLOGUES[linelist.species]
    max_local = int(linelist.local_iso_id.max()) if len(linelist) else 0
    lookup = np.zeros(max_local + 1, dtype=np.float64)
    for _global_id, local_iso_id, _abundance, molar_mass in table:
        if local_iso_id <= max_local:
            lookup[local_iso_id] = molar_mass
    return lookup[linelist.local_iso_id]


def _line_intensity_at_t(linelist: LineList, temperature_k: float) -> np.ndarray:
    """Full HITRAN intensity temperature scaling S(T) from S(296 K).

    ``S(T) = S(296) * [Q(296)/Q(T)] * exp(-c2*E"*(1/T - 1/296))
             * (1 - exp(-c2*nu0/T)) / (1 - exp(-c2*nu0/296))``

    Reduces exactly to ``S(296) == s296`` when ``temperature_k == 296.0``
    (every factor above is identically 1 at T=296 K), which is asserted in
    ``tests/test_lbl.py``.
    """
    # Q(T) only depends on isotopologue, not per-line -- resolve once per
    # unique isotopologue (a handful) and broadcast back, rather than
    # calling partition_function per line.
    unique_isos, inverse = np.unique(linelist.local_iso_id, return_inverse=True)
    q_ref = np.array(
        [partition_function(linelist.molecule_id, int(iso), _T_REF_K) for iso in unique_isos]
    )
    q_t = np.array(
        [partition_function(linelist.molecule_id, int(iso), temperature_k) for iso in unique_isos]
    )
    q_ratio = (q_ref / q_t)[inverse]

    boltzmann = np.exp(-C2_RAD * linelist.e_lower * (1.0 / temperature_k - 1.0 / _T_REF_K))
    # (1 - exp(-x)) == -expm1(-x); ratio of two such terms == ratio of the
    # expm1(-x) values directly (negatives cancel), and expm1 keeps
    # precision better than 1-exp for small x.
    stim = np.expm1(-C2_RAD * linelist.nu_cm1 / temperature_k) / np.expm1(
        -C2_RAD * linelist.nu_cm1 / _T_REF_K
    )
    return linelist.s296 * q_ratio * boltzmann * stim


def absorption_cross_section(
    linelist: LineList,
    grid: SpectralGrid,
    pressure_mbar: float,
    temperature_k: float,
    partial_pressure_mbar: float,
    *,
    wing_cutoff_cm1: float = 25.0,
    min_strength_fraction: float = 1.0e-4,
) -> np.ndarray:
    """Voigt cross-section, cm^2/molecule, shape ``(grid.n,)``.

    ``pressure_mbar`` is the total (broadening) pressure of the layer at the
    species' Curtis-Godson effective point (``PathLayer.p_eff_mbar[species]``);
    ``partial_pressure_mbar`` is the absorber's own partial (self) pressure
    at that same point. Air (foreign) broadening uses
    ``pressure_mbar - partial_pressure_mbar``, clamped at zero.

    Vectorised over all lines at once: each line contributes only within
    ``+/-wing_cutoff_cm1`` of its (pressure-shifted) centre, via a dense
    ``(n_lines, n_window)`` Voigt evaluation scattered onto the grid with
    ``numpy.bincount`` -- not a per-line Python loop over the full grid.

    ``min_strength_fraction`` drops lines whose temperature-scaled intensity
    is below this fraction of the strongest line *in this call* (same
    species, same layer conditions) before the dense Voigt evaluation. This
    matters because the full 1775-4325 cm^-1 HITRAN fetch for some species
    is dominated by extremely weak hot/combination-band lines: e.g. O3 and
    N2O each return 200k+ lines, but 99.9%+ of total line strength is
    concentrated in a few percent of them (verified empirically during
    development -- at the default 1e-4, every species in this package's
    fetch loses < 1% of summed line strength, usually << 0.1%, spread thinly
    across the spectrum rather than concentrated at one wavenumber). Without
    this, a 50-layer x 2500-point run is roughly a minute instead of a few
    seconds. Set to 0 to disable and use every line.
    """
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if len(linelist) == 0:
        return np.zeros(grid.n, dtype=np.float64)

    p_total_atm = pressure_mbar / STD_PRESSURE_MBAR
    p_self_atm = max(partial_pressure_mbar, 0.0) / STD_PRESSURE_MBAR
    p_air_atm = max(p_total_atm - p_self_atm, 0.0)

    s_t = _line_intensity_at_t(linelist, temperature_k)

    # Doppler HWHM, cm^-1: nu0 * sqrt(2 ln2 R T / (M c^2)), isotopologue mass.
    molar_mass = _isotopologue_mass_g_mol(linelist)
    m_kg_mol = molar_mass * 1.0e-3
    alpha_d = linelist.nu_cm1 * np.sqrt(
        2.0 * np.log(2.0) * GAS_CONSTANT_R * temperature_k / (m_kg_mol * LIGHT_C**2)
    )

    # Lorentz (pressure-broadened) HWHM and pressure-shifted line centre.
    gamma_l = (linelist.gamma_air * p_air_atm + linelist.gamma_self * p_self_atm) * (
        _T_REF_K / temperature_k
    ) ** linelist.n_air
    nu_shifted = linelist.nu_cm1 + linelist.delta_air * p_total_atm

    resolution = grid.resolution_cm1
    nu_grid = grid.wavenumber_cm1
    n_grid = grid.n
    grid_min = nu_grid[0]

    half_width_pts = int(np.ceil(wing_cutoff_cm1 / resolution))
    offsets = np.arange(-half_width_pts, half_width_pts + 1)
    center_idx = np.round((nu_shifted - grid_min) / resolution).astype(np.int64)

    # Drop lines whose window doesn't touch the grid at all, and (unless
    # disabled) negligibly weak lines -- both keep the dense matrix below
    # limited to lines that can actually matter, without a per-line loop.
    keep = (center_idx + half_width_pts >= 0) & (center_idx - half_width_pts < n_grid)
    if min_strength_fraction > 0.0:
        s_max = s_t.max() if s_t.size else 0.0
        if s_max > 0.0:
            keep &= s_t >= (min_strength_fraction * s_max)
    if not np.any(keep):
        return np.zeros(n_grid, dtype=np.float64)
    center_idx = center_idx[keep]
    s_t = s_t[keep]
    alpha_d = alpha_d[keep]
    gamma_l = gamma_l[keep]
    nu_shifted = nu_shifted[keep]

    idx_matrix = center_idx[:, None] + offsets[None, :]        # (n_lines, n_window)
    valid = (idx_matrix >= 0) & (idx_matrix < n_grid)
    safe_idx = np.clip(idx_matrix, 0, n_grid - 1)

    dnu = nu_grid[safe_idx] - nu_shifted[:, None]               # exact offset from centre, cm^-1

    ln2 = np.log(2.0)
    x = np.sqrt(ln2) * dnu / alpha_d[:, None]
    y = np.sqrt(ln2) * gamma_l[:, None] / alpha_d[:, None]
    voigt_shape = np.sqrt(ln2 / np.pi) / alpha_d[:, None] * wofz(x + 1j * y).real

    contribution = s_t[:, None] * voigt_shape
    contribution = np.where(valid, contribution, 0.0)

    cross_section = np.bincount(safe_idx.ravel(), weights=contribution.ravel(), minlength=n_grid)
    return cross_section[:n_grid].astype(np.float64)


def molecular_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
    *,
    species: Sequence[Species] | None = None,
    linelists: dict[Species, LineList] | None = None,
    include_continuum: bool = True,
    wing_cutoff_cm1: float = 25.0,
    min_strength_fraction: float = 1.0e-4,
) -> tuple[np.ndarray, dict[Species, np.ndarray]]:
    """THE headline function. Returns ``(tau_total, tau_by_species)``, each
    shape ``(len(layers), grid.n)``.

    ``linelists`` defaults to :func:`mwir_atmos.hitran.fetch.ensure_linelists`
    (offline-safe: loads the shipped cache, only fetches if it is missing).
    ``species`` defaults to :data:`mwir_atmos.types.REQUIRED_SPECIES` (the
    six species every profile must supply -- excludes O2, which has no
    HITRAN line data fetched for this package and negligible MWIR line
    absorption anyway).

    Per layer and species, broadening/shift use the Curtis-Godson effective
    pressure and temperature (``PathLayer.p_eff_mbar`` / ``t_eff_k``); the
    species' self partial pressure is derived from its column fraction of
    the total air column (``column_cm2[species] / air_column_cm2``), i.e.
    the path-mean volume mixing ratio, times that effective pressure.
    """
    if species is None:
        species = REQUIRED_SPECIES
    if linelists is None:
        linelists = ensure_linelists()

    n_layers = len(layers)
    tau_by_species: dict[Species, np.ndarray] = {
        sp: np.zeros((n_layers, grid.n), dtype=np.float64) for sp in species
    }

    for row, layer in enumerate(layers):
        for sp in species:
            column = layer.column_cm2.get(sp, 0.0)
            if column <= 0.0:
                continue
            linelist = linelists.get(sp)
            if linelist is None:
                continue

            p_eff = layer.p_eff_mbar.get(sp, layer.p_mbar)
            t_eff = layer.t_eff_k.get(sp, layer.t_k)
            vmr = column / layer.air_column_cm2 if layer.air_column_cm2 > 0 else 0.0
            p_self = p_eff * vmr

            xsec = absorption_cross_section(
                linelist,
                grid,
                p_eff,
                t_eff,
                p_self,
                wing_cutoff_cm1=wing_cutoff_cm1,
                min_strength_fraction=min_strength_fraction,
            )
            tau_by_species[sp][row] = xsec * column

    if include_continuum:
        if Species.H2O in tau_by_species:
            tau_by_species[Species.H2O] = tau_by_species[Species.H2O] + h2o_continuum_optical_depth(
                layers, grid
            )
        if Species.CO2 in tau_by_species:
            tau_by_species[Species.CO2] = tau_by_species[Species.CO2] + co2_continuum_optical_depth(
                layers, grid
            )

    if tau_by_species:
        tau_total = np.sum(np.stack(list(tau_by_species.values()), axis=0), axis=0)
    else:
        tau_total = np.zeros((n_layers, grid.n), dtype=np.float64)

    return tau_total, tau_by_species
