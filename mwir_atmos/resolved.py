"""Line-resolved spectral sampling: a fine monochromatic calculation seen
through an instrument function.

Why
---
Atmospheric absorption lines are ~0.07 cm^-1 wide at the surface and a few
thousandths of a cm^-1 in the stratosphere. With ``SpectralSampling.POINT`` the
physics runs only at the output grid points, so on a 1 cm^-1 grid each sample
lands at a random place relative to the nearby lines: sometimes in a core,
usually in a gap. The sample is an exact *monochromatic* value, but not what a
1 cm^-1 instrument measures, and transmittance is nonlinear in optical depth, so
the errors do not average out. On a 2 km horizontal path they reached 3.8
percentage points in single bins (0.6 pp on average) over 2040-2060 cm^-1.

Method
------
1. The output grid is exactly the POINT grid (``resolution_cm1`` spacing).
2. A fine lattice with step ``h = resolution_cm1 / r`` (integer ``r``) contains
   every output point and extends past both band edges by the instrument
   function's support, because an instrument centred near a band edge sees
   radiation from beyond it.
3. Consecutive output points are processed in chunks (bounded memory). For each
   chunk the complete physics -- layer optics, transmittance and radiance --
   runs on the chunk's fine grid, and every spectral array is reduced to the
   output points: ``out_j = sum_k w_k f(nu_j + k h)``, with normalised
   instrument weights ``w``. Chunks share the lattice, so line evaluation does
   not depend on where the spectrum was split
   (:attr:`~mwir_atmos.types.SpectralGrid.lattice_origin_cm1`).
4. Monochromatic radiance from the single-scattering solver is computed on the
   fine grid directly. DISORT and spherical Monte Carlo solve each spectral
   point separately, which is too costly on a fine grid; for them the fine
   points under each instrument function are ranked by column optical depth
   and split into cumulative-weight intervals (``_G_EDGES``); layer optics are
   averaged within each interval and the solver runs once per interval. This
   is the correlated-k / spectral-mapping approximation (e.g. Lacis & Oinas
   1991, doi:10.1029/90JD01945); :func:`binned_radiance` documents its
   assumptions. Transmittance is always the exact fine-grid result.

Resolution of the monochromatic calculation
-------------------------------------------
:func:`automatic_fine_step` sizes ``h`` from typical line half-widths along
the path. Coarse fine steps under-resolve narrow lines; the ``fine_step_cm1``
scenario field overrides the automatic choice for convergence checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Callable, Optional, Sequence

import numpy as np

from .constants import GAS_CONSTANT_R, LIGHT_C, STD_PRESSURE_MBAR
from .types import (
    InstrumentFunction,
    LayerOptics,
    PathLayer,
    RadianceResult,
    SpectralGrid,
    TransmittanceResult,
)

__all__ = [
    "FineGridPlan",
    "automatic_fine_step",
    "binned_radiance",
    "degrade",
    "instrument_weights",
    "plan_fine_grid",
]

#: Gaussian instrument functions are truncated at +/- this many FWHM
#: (4.7 standard deviations; the discarded tails hold 3e-6 of the area).
_GAUSSIAN_SUPPORT_FWHM = 2.0

#: Fine grid points per Voigt half-width of the reference line (see
#: :func:`automatic_fine_step`).
_SAMPLES_PER_HWHM = 2.0
#: Layers are ignored when choosing the fine step if, together with every
#: layer of narrower lines, they hold less than this fraction of the path's
#: absorber column (their lines are too few to matter; see automatic_fine_step).
_NEGLIGIBLE_COLUMN_FRACTION = 0.01
#: Automatic fine steps are clamped to this range, cm^-1.
_FINE_STEP_BOUNDS = (5.0e-4, 0.05)
#: Reference line for automatic sizing: a CO2-like line at 296 K with an
#: air-broadened half-width of 0.07 cm^-1/atm and temperature exponent 0.75.
_REF_GAMMA_AIR = 0.07
_REF_N_AIR = 0.75
_REF_MASS_KG_MOL = 44.0e-3

#: (fine points per chunk) x (layers) held in memory at once.
_CHUNK_ELEMENTS = 1 << 21
#: Chunks never shrink below this many fine points (per-chunk overhead).
_MIN_CHUNK_FINE_POINTS = 4096

#: Cumulative-weight edges of the correlated-k intervals used by
#: :func:`binned_radiance`: finer toward the strongly absorbing tail, where
#: optical depth changes fastest with g.
_G_EDGES = np.array([0.0, 0.30, 0.55, 0.75, 0.875, 0.94, 0.975, 0.99, 1.0])


def instrument_weights(function: InstrumentFunction, fwhm_cm1: float,
                       step_cm1: float) -> np.ndarray:
    """Normalised weights at offsets ``k*step`` for ``k = -S..S``, shape ``(2S+1,)``.

    ``BOX`` integrates the uniform response over each fine cell (so a box
    whose width is an even number of steps reduces to the trapezoid rule and a
    box spanning an odd number of steps to a plain average). ``TRIANGLE``
    (half-base = FWHM) and ``GAUSSIAN`` are sampled at the fine points. Weights
    sum to one, so a constant spectrum is reproduced exactly.
    """
    if not (math.isfinite(fwhm_cm1) and fwhm_cm1 > 0 and math.isfinite(step_cm1) and step_cm1 > 0):
        raise ValueError("instrument FWHM and fine step must be finite and positive")
    function = InstrumentFunction(function)
    if function == InstrumentFunction.BOX:
        half = 0.5 * fwhm_cm1
        support = int(math.ceil(half / step_cm1 + 0.5))
        x = np.arange(-support, support + 1) * step_cm1
        w = np.minimum(x + 0.5 * step_cm1, half) - np.maximum(x - 0.5 * step_cm1, -half)
        w = np.clip(w, 0.0, None)
    elif function == InstrumentFunction.TRIANGLE:
        support = int(math.ceil(fwhm_cm1 / step_cm1))
        x = np.arange(-support, support + 1) * step_cm1
        w = np.clip(1.0 - np.abs(x) / fwhm_cm1, 0.0, None)
    else:
        support = int(math.ceil(_GAUSSIAN_SUPPORT_FWHM * fwhm_cm1 / step_cm1))
        x = np.arange(-support, support + 1) * step_cm1
        w = np.exp(-4.0 * math.log(2.0) * (x / fwhm_cm1) ** 2)
    # Trim zero-weight ends symmetrically so the support is as short as possible.
    nonzero = np.flatnonzero(w > 0.0)
    trim = min(int(nonzero[0]), int(w.size - 1 - nonzero[-1]))
    if trim:
        w = w[trim:-trim]
    return w / w.sum()


def _layer_hwhm(layer: PathLayer, wavenumber_cm1: float) -> float:
    """Voigt half-width (cm^-1) of the reference line in ``layer``."""
    p = max(layer.p_mbar, 0.0) / STD_PRESSURE_MBAR
    t = max(layer.t_k, 1.0)
    gamma = _REF_GAMMA_AIR * p * (296.0 / t) ** _REF_N_AIR
    alpha = wavenumber_cm1 * math.sqrt(2.0 * math.log(2.0) * GAS_CONSTANT_R * t
                                       / (_REF_MASS_KG_MOL * LIGHT_C**2))
    return 0.5346 * gamma + math.sqrt(0.2166 * gamma * gamma + alpha * alpha)


def automatic_fine_step(layers: Sequence[PathLayer], wavenumber_min_cm1: float) -> float:
    """Fine step resolving the narrowest lines that carry a non-negligible column.

    Each layer gets the Voigt half-width of a representative line: CO2-like
    broadening and mass, evaluated at the band's *lowest* wavenumber, where
    Doppler widths are narrowest. Layers are ranked from narrowest to broadest
    lines; the narrowest are skipped while their cumulative share of the
    path's air column stays below ``_NEGLIGIBLE_COLUMN_FRACTION`` (e.g. the
    mesosphere on a ground-to-space path). The step is the remaining minimum
    half-width divided by ``_SAMPLES_PER_HWHM``, clamped to
    ``_FINE_STEP_BOUNDS``. This is a sizing rule, not an error bound; check
    convergence by setting ``fine_step_cm1`` explicitly.
    """
    if not layers:
        return _FINE_STEP_BOUNDS[1]
    widths = np.array([_layer_hwhm(l, wavenumber_min_cm1) for l in layers])
    columns = np.array([max(l.air_column_cm2, 0.0) for l in layers])
    total = columns.sum()
    order = np.argsort(widths)
    if total > 0:
        share = np.cumsum(columns[order]) / total
        first = int(np.searchsorted(share, _NEGLIGIBLE_COLUMN_FRACTION, side="right"))
        first = min(first, len(order) - 1)
    else:
        first = 0
    step = widths[order[first]] / _SAMPLES_PER_HWHM
    return float(np.clip(step, *_FINE_STEP_BOUNDS))


@dataclass(frozen=True)
class FineGridPlan:
    """How an output grid maps onto its fine lattice."""

    #: Fine step, cm^-1 (= output resolution / ``ratio``).
    step_cm1: float
    #: Fine steps per output step (integer).
    ratio: int
    #: Instrument weights at fine offsets ``-support..support``.
    weights: np.ndarray
    #: Lattice origin: the first output point.
    origin_cm1: float

    @property
    def support(self) -> int:
        return (self.weights.size - 1) // 2


def plan_fine_grid(output: SpectralGrid, *, fine_step_cm1: float,
                   function: InstrumentFunction, fwhm_cm1: Optional[float]) -> FineGridPlan:
    """Fit the requested fine step to an integer subdivision of the output step.

    The fine step never exceeds the request, and never exceeds a quarter of
    the instrument FWHM (so the instrument function itself is resolved).
    """
    fwhm = output.resolution_cm1 if fwhm_cm1 is None else float(fwhm_cm1)
    target = min(float(fine_step_cm1), 0.25 * fwhm)
    ratio = max(1, int(math.ceil(output.resolution_cm1 / target - 1e-9)))
    step = output.resolution_cm1 / ratio
    return FineGridPlan(step_cm1=step, ratio=ratio,
                        weights=instrument_weights(function, fwhm, step),
                        origin_cm1=float(output.wavenumber_cm1[0]))


def chunks(plan: FineGridPlan, n_output: int, n_layers: int):
    """Yield ``(first, stop, fine_grid)`` for consecutive output-point chunks."""
    target = max(_MIN_CHUNK_FINE_POINTS, _CHUNK_ELEMENTS // max(n_layers, 1))
    per_chunk = max(1, (target - 2 * plan.support) // plan.ratio)
    for first in range(0, n_output, per_chunk):
        stop = min(first + per_chunk, n_output)
        index0 = first * plan.ratio - plan.support
        count = (stop - 1 - first) * plan.ratio + 2 * plan.support + 1
        index = index0 + np.arange(count, dtype=np.float64)
        grid = SpectralGrid(wavenumber_cm1=plan.origin_cm1 + plan.step_cm1 * index,
                            resolution_cm1=plan.step_cm1,
                            lattice_origin_cm1=plan.origin_cm1, lattice_index0=index0)
        yield first, stop, grid


def degrade(values: np.ndarray, plan: FineGridPlan, n_output: int) -> np.ndarray:
    """Apply the instrument function along the last axis of a chunk array.

    ``values[..., i]`` is on the chunk's fine grid; output point ``j`` of the
    chunk is centred on fine index ``support + j*ratio``.
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(values, dtype=np.float64), plan.weights.size, axis=-1)
    return windows[..., ::plan.ratio, :][..., :n_output, :] @ plan.weights


def degrade_transmittance(t: TransmittanceResult, plan: FineGridPlan, n_output: int,
                          grid: SpectralGrid) -> TransmittanceResult:
    return TransmittanceResult(
        grid=grid,
        total=degrade(t.total, plan, n_output),
        molecular=degrade(t.molecular, plan, n_output),
        aerosol=degrade(t.aerosol, plan, n_output),
        rayleigh=degrade(t.rayleigh, plan, n_output),
        by_species={sp: degrade(v, plan, n_output) for sp, v in t.by_species.items()},
        cumulative=None if t.cumulative is None else degrade(t.cumulative, plan, n_output),
    )


_RADIANCE_ARRAYS = tuple(f.name for f in fields(RadianceResult)
                         if f.name not in ("grid", "sampling_standard_error", "stokes",
                                           "stokes_standard_error", "sensor_signal"))


def degrade_radiance(r: RadianceResult, plan: FineGridPlan, n_output: int,
                     grid: SpectralGrid) -> RadianceResult:
    arrays = {name: degrade(getattr(r, name), plan, n_output) for name in _RADIANCE_ARRAYS}
    return RadianceResult(grid=grid, **arrays)


def concatenate_transmittance(parts: Sequence[TransmittanceResult],
                              grid: SpectralGrid) -> TransmittanceResult:
    join = lambda get: np.concatenate([get(p) for p in parts], axis=-1)
    return TransmittanceResult(
        grid=grid,
        total=join(lambda p: p.total),
        molecular=join(lambda p: p.molecular),
        aerosol=join(lambda p: p.aerosol),
        rayleigh=join(lambda p: p.rayleigh),
        by_species={sp: join(lambda p, sp=sp: p.by_species[sp]) for sp in parts[0].by_species},
        cumulative=(None if parts[0].cumulative is None else join(lambda p: p.cumulative)),
    )


def concatenate_radiance(parts: Sequence[RadianceResult], grid: SpectralGrid) -> RadianceResult:
    arrays = {name: np.concatenate([getattr(p, name) for p in parts], axis=-1)
              for name in _RADIANCE_ARRAYS}
    out = RadianceResult(grid=grid, **arrays)
    for name in ("sampling_standard_error", "stokes", "stokes_standard_error"):
        keys = getattr(parts[0], name).keys()
        setattr(out, name, {k: np.concatenate([getattr(p, name)[k] for p in parts], axis=0)
                            for k in keys})
    return out


def binned_radiance(
    column_optics: LayerOptics,
    plan: FineGridPlan,
    n_output: int,
    output_wavenumbers: np.ndarray,
    solve: Callable[[SpectralGrid, LayerOptics], RadianceResult],
    grid: SpectralGrid,
) -> RadianceResult:
    """Radiance for per-spectral-point solvers via correlated-k binning.

    For output point ``j``, the fine points under its instrument function are
    ranked by column extinction optical depth. Their cumulative instrument
    weight ``g`` is split at ``_G_EDGES``; within each interval every layer's
    optical depths are weight-averaged (the aerosol asymmetry parameter is
    averaged with aerosol-scattering weights), giving at most
    ``len(_G_EDGES) - 1`` pseudo-monochromatic columns per output point. The
    solver sees them on a pseudo grid whose wavenumbers repeat the output
    point's, so Planck, solar and lunar sources are evaluated at the output
    wavenumber. Output radiance is the interval-weighted sum; Monte Carlo
    standard errors are combined in quadrature with the same weights.

    Assumptions (the usual correlated-k ones): spectral ordering by column
    optical depth is the same in every layer, and sources and surface
    properties are constant across one instrument function.
    """
    from scipy import sparse

    width = plan.weights.size
    tau_column = column_optics.tau_total.sum(axis=0)
    index = (np.arange(n_output)[:, None] * plan.ratio + np.arange(width)[None, :])
    order = np.argsort(tau_column[index], axis=1, kind="stable")
    fine = np.take_along_axis(index, order, axis=1)
    weight = plan.weights[order]
    middle = np.cumsum(weight, axis=1) - 0.5 * weight
    interval = np.searchsorted(_G_EDGES[1:-1], middle, side="right")
    n_intervals = _G_EDGES.size - 1
    group = (np.arange(n_output)[:, None] * n_intervals + interval).ravel()
    group_weight = np.bincount(group, weights=weight.ravel(), minlength=n_output * n_intervals)
    used = np.flatnonzero(group_weight > 0.0)
    pseudo = np.full(group_weight.size, -1, dtype=np.int64)
    pseudo[used] = np.arange(used.size)

    # Averaging operator: fine point -> pseudo point.
    average = sparse.csr_matrix(
        (weight.ravel() / group_weight[group], (fine.ravel(), pseudo[group])),
        shape=(tau_column.size, used.size))

    def binned(a: np.ndarray) -> np.ndarray:
        return np.asarray((average.T @ a.T).T)

    scattering = column_optics.tau_aerosol_sca
    asymmetry = column_optics.aerosol_asymmetry
    if asymmetry is not None:
        num = binned(asymmetry * scattering)
        den = binned(scattering)
        asymmetry = np.divide(num, den, out=binned(asymmetry), where=den > 0)
    pseudo_grid = SpectralGrid(
        wavenumber_cm1=np.asarray(output_wavenumbers, dtype=np.float64)[used // n_intervals],
        resolution_cm1=grid.resolution_cm1)
    pseudo_optics = LayerOptics(
        tau_molecular=binned(column_optics.tau_molecular),
        tau_aerosol_abs=binned(column_optics.tau_aerosol_abs),
        tau_aerosol_sca=binned(scattering),
        tau_rayleigh=binned(column_optics.tau_rayleigh),
        grid=pseudo_grid,
        tau_by_species={},
        aerosol_asymmetry=asymmetry,
    )
    result = solve(pseudo_grid, pseudo_optics)

    # Combination operator: pseudo point -> output point, weight = interval width.
    combine = sparse.csr_matrix((group_weight[used], (np.arange(used.size), used // n_intervals)),
                                shape=(used.size, n_output))
    arrays = {name: np.asarray(combine.T @ getattr(result, name)) for name in _RADIANCE_ARRAYS}
    out = RadianceResult(grid=grid, **arrays)
    squared = combine.multiply(combine)
    out.sampling_standard_error = {
        k: np.sqrt(np.asarray(squared.T @ (v * v))) for k, v in result.sampling_standard_error.items()}
    out.stokes = {k: np.asarray(combine.T @ v) for k, v in result.stokes.items()}
    out.stokes_standard_error = {
        k: np.sqrt(np.asarray(squared.T @ (v * v))) for k, v in result.stokes_standard_error.items()}
    return out
