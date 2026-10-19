"""Assembly of per-layer optical depths into path transmittance.

This is the seam between the two halves of the physics: the
line-by-line molecular engine (``optics.lbl`` + ``optics.continuum``) and the
particulate/molecular scattering models (``optics.aerosol``, ``optics.rayleigh``).

Transmittance uses *extinction* -- absorption plus scattering -- because photons
scattered out of the line of sight are lost to the sensor just as absorbed ones
are. The decomposition is kept intact in :class:`LayerOptics` because the
radiance layer needs absorption and scattering separately: only absorption
drives thermal emission, only scattering drives the solar/lunar source terms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .._array import array_namespace
from ..types import (
    AerosolModel,
    LayerOptics,
    PathGeometry,
    SpectralGrid,
    Species,
    TransmittanceResult,
)
from .aerosol import aerosol_optical_depth
from .continuum import co2_continuum_optical_depth
from .lbl import DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF, molecular_optical_depth
from .rayleigh import rayleigh_optical_depth

__all__ = ["compute_layer_optics", "path_transmittance", "transmittance_for_path"]


def compute_layer_optics(
    path: PathGeometry,
    grid: SpectralGrid,
    *,
    aerosol: AerosolModel = AerosolModel.RURAL,
    visibility_km: float = 23.0,
    wind_speed_m_s: float = 5.0,
    aerosol_winter: bool = False,
    species: Sequence[Species] | None = None,
    linelists: dict | None = None,
    include_continuum: bool = True,
    include_rayleigh: bool = True,
    include_co2_continuum: bool = False,
    wing_cutoff_cm1: float = 25.0,
    min_strength_fraction: float = 0.0,
    min_peak_optical_depth: float = DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF,
    n_threads: int | None = None,
    strength_references: dict | None = None,
    backend: str | None = None,
) -> LayerOptics:
    """Compute every optical-depth contribution for each layer of ``path``.

    ``include_co2_continuum`` is off by default: the CO2 continuum is currently
    a documented zero stub, so enabling it costs a call and changes nothing.

    ``min_strength_fraction``, ``min_peak_optical_depth`` and
    ``wing_cutoff_cm1`` control line-by-line accuracy versus cost; see
    :func:`mwir_atmos.optics.lbl.absorption_cross_section`. ``n_threads``,
    ``strength_references`` and ``backend`` are passed to
    :func:`~mwir_atmos.optics.lbl.molecular_optical_depth`.
    """
    layers = path.layers
    n_layers = len(layers)
    shape = (n_layers, grid.n)

    tau_mol, tau_by_species = molecular_optical_depth(
        layers,
        grid,
        species=species,
        linelists=linelists,
        include_continuum=include_continuum,
        wing_cutoff_cm1=wing_cutoff_cm1,
        min_strength_fraction=min_strength_fraction,
        min_peak_optical_depth=min_peak_optical_depth,
        n_threads=n_threads,
        strength_references=strength_references,
        backend=backend,
    )

    if include_co2_continuum:
        tau_mol = tau_mol + co2_continuum_optical_depth(layers, grid)

    tau_aer_abs, tau_aer_sca, g_aer = aerosol_optical_depth(
        layers, grid, aerosol, visibility_km=visibility_km, wind_speed_m_s=wind_speed_m_s,
        winter=aerosol_winter,
    )

    tau_ray = (
        rayleigh_optical_depth(layers, grid)
        if include_rayleigh
        else array_namespace(grid.wavenumber_cm1).zeros(shape, dtype=np.float64)
    )

    return LayerOptics(
        tau_molecular=tau_mol,
        tau_aerosol_abs=tau_aer_abs,
        tau_aerosol_sca=tau_aer_sca,
        tau_rayleigh=tau_ray,
        grid=grid,
        tau_by_species=tau_by_species,
        aerosol_asymmetry=g_aer,
    )


def path_transmittance(optics: LayerOptics) -> TransmittanceResult:
    """Collapse per-layer optical depths into sensor-to-source transmittance.

    ``cumulative[k]`` is the transmittance from the sensor to the *entry* face
    of layer ``k``; ``cumulative[0]`` is unity and ``cumulative[-1]`` equals
    ``total``. The radiance integration walks this array, so it is always
    populated.
    """
    grid = optics.grid
    tau_total = optics.tau_total
    xp = array_namespace(tau_total)

    total_od = tau_total.sum(axis=0)
    # exp(-[0, cumsum]) built in place: no concatenated copy.
    cumulative = xp.empty((tau_total.shape[0] + 1, grid.n), dtype=np.float64)
    cumulative[0] = 0.0
    xp.cumsum(tau_total, axis=0, out=cumulative[1:])
    xp.negative(cumulative, out=cumulative)
    xp.exp(cumulative, out=cumulative)

    tau_aerosol = optics.tau_aerosol_abs + optics.tau_aerosol_sca

    return TransmittanceResult(
        grid=grid,
        total=xp.exp(-total_od),
        molecular=xp.exp(-optics.tau_molecular.sum(axis=0)),
        aerosol=xp.exp(-tau_aerosol.sum(axis=0)),
        rayleigh=xp.exp(-optics.tau_rayleigh.sum(axis=0)),
        by_species={
            sp: xp.exp(-t.sum(axis=0)) for sp, t in optics.tau_by_species.items()
        },
        cumulative=cumulative,
    )


def transmittance_for_path(
    path: PathGeometry, grid: SpectralGrid, **kwargs
) -> tuple[LayerOptics, TransmittanceResult]:
    """Convenience wrapper: layer optics and path transmittance in one call."""
    optics = compute_layer_optics(path, grid, **kwargs)
    return optics, path_transmittance(optics)
