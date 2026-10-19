"""Voigt line-by-line molecular absorption engine.

The headline function is :func:`molecular_optical_depth`: given a resolved
path (:class:`~mwir_atmos.types.PathLayer` sequence, e.g. from
:func:`mwir_atmos.geometry.path.build_path`) and a
:class:`~mwir_atmos.types.SpectralGrid`, it returns per-layer, per-species
optical depth from HITRAN line data plus the H2O continuum.

Physics (requirements in ``docs/build_contract.md``):

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

Performance
-----------
Per-line constants (isotopologue masses and partition-function indices, the
stimulated-emission reference, a wavenumber sort) are derived once per
:class:`LineList` and cached. Each evaluation then selects only the lines
whose wings can reach the requested grid (a binary search, not a scan of
200k+ lines), and evaluates the profile over a *ragged* set of
``(line, grid point)`` pairs, so windows clipped by the grid edge cost nothing.
Contributions are scatter-added with ``numpy.bincount``.

Two evaluation methods produce the same physical profile:

``"direct"``
    Evaluate the Voigt profile at every grid point within the cutoff of every
    line. Exact, and cheapest when a line's window is only tens of points.
``"multigrid"``
    For fine grids, where a +/-25 cm^-1 wing spans thousands of points but is
    smooth beyond a few line widths. Every line is evaluated on a coarse mesh
    (every ``m``-th point of the grid's lattice) and the summed mesh is
    interpolated with a four-point Lagrange cubic. Each line is then corrected
    *exactly* -- profile minus its own interpolant -- over its core and over
    the two stretches of grid where the interpolation stencil straddles the
    wing cutoff. The remaining approximation is cubic interpolation of smooth
    far wings; mesh spacing is tied to the core width so that this error is a
    small fraction of each wing (see :func:`_multigrid_plan`). The coarse mesh
    is anchored to the grid's lattice (``SpectralGrid.lattice_origin_cm1``), so
    spectral chunks of one calculation reproduce an undivided evaluation.

``"auto"`` (used by :func:`molecular_optical_depth`) picks multigrid only when
it is expected to be several times cheaper than direct evaluation.

:func:`molecular_optical_depth` evaluates its independent ``(layer, species)``
pairs on a thread pool; NumPy and ``scipy.special.wofz`` release the GIL in
their inner loops. Each pair writes only its own output row, so results do not
depend on the thread count. ``n_threads`` (or the ``MWIR_THREADS`` environment
variable) controls the pool; the default uses every available CPU.
"""

from __future__ import annotations

import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.special import erfcx, wofz

from .._array import array_namespace
from ..constants import C2_RAD, GAS_CONSTANT_R, LIGHT_C, STD_PRESSURE_MBAR
from ..hitran.cache import LineList
from ..hitran.fetch import GLOBAL_ISOTOPOLOGUES, ensure_linelists
from ..hitran.tips import partition_function
from ..types import PathLayer, REQUIRED_SPECIES, Species, SpectralGrid
from .continuum import co2_continuum_optical_depth, h2o_continuum_optical_depth

__all__ = ["absorption_cross_section", "evaluations_per_line", "line_strength_reference",
           "molecular_optical_depth", "prepare_linelists", "resolve_backend",
           "resolve_thread_count"]

#: HITRAN reference temperature for tabulated line intensities and widths.
_T_REF_K = 296.0

#: The MT_CKD local-line convention: water lines are truncated at, and have
#: their Lorentz value at, +/-25 cm^-1 subtracted.
_MT_CKD_CUTOFF_CM1 = 25.0

#: Largest number of (line, grid point) pairs evaluated in one vectorised
#: batch. Temporaries then stay at 128-256 KB, inside per-core L2 cache and
#: below the size where allocators return memory to the OS on every free.
#: Measured on the full default scenario: 2**17 was 4x slower than 2**14
#: single-threaded (page-fault and allocator cost), and scaled worse.
_BATCH_ELEMENTS = 1 << 14

#: Default per-layer peak optical depth below which a line is not evaluated
#: (``Scenario.line_optical_depth_cutoff`` has the same default).
DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF = 3.0e-6

_LN2 = np.log(2.0)
_SQRT_LN2 = np.sqrt(_LN2)
_SQRT_LN2_OVER_PI = np.sqrt(_LN2 / np.pi)
_METHODS = ("auto", "direct", "multigrid")


_BACKENDS = ("cpu", "cuda")


def resolve_backend(backend: Optional[str] = None) -> str:
    """Line-by-line backend: explicit value, else ``MWIR_BACKEND``, else ``"cpu"``.

    ``"cuda"`` raises if CuPy or a CUDA device is missing, rather than
    silently running on the CPU.
    """
    if backend is None:
        backend = os.environ.get("MWIR_BACKEND", "").strip().lower() or "cpu"
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend == "cuda":
        from . import lbl_cuda
        if not lbl_cuda.available():
            raise RuntimeError("backend 'cuda' requested, but CuPy or a CUDA device is not "
                               "available (pip install cupy-cuda12x)")
    return backend


def resolve_thread_count(n_threads: Optional[int] = None) -> int:
    """Worker threads to use: explicit value, else ``MWIR_THREADS``, else all CPUs."""
    if n_threads is None:
        env = os.environ.get("MWIR_THREADS", "").strip()
        if env:
            try:
                n_threads = int(env)
            except ValueError as exc:
                raise ValueError(f"MWIR_THREADS must be a positive integer, got {env!r}") from exc
        else:
            n_threads = os.cpu_count() or 1
    if type(n_threads) is not int or n_threads < 1:
        raise ValueError(f"n_threads must be a positive integer, got {n_threads!r}")
    return n_threads


# =========================================================================
# Per-line constants, derived once per LineList
# =========================================================================

_LINE_FIELDS = ("nu_cm1", "s296", "gamma_air", "gamma_self", "e_lower",
                "n_air", "delta_air", "local_iso_id")


@dataclass(frozen=True, eq=False)
class _PreparedLines:
    """A LineList sorted by wavenumber, with per-line constants precomputed."""

    #: The source arrays at preparation time; a later reassignment of any
    #: LineList field invalidates the cache entry (identity comparison).
    source_arrays: tuple
    molecule_id: int
    nu: np.ndarray
    s296: np.ndarray
    gamma_air: np.ndarray
    gamma_self: np.ndarray
    e_lower: np.ndarray
    n_air: np.ndarray
    delta_air: np.ndarray
    #: Per-line index into ``unique_isos``.
    iso_index: np.ndarray
    unique_isos: tuple
    q_ref: np.ndarray
    m_kg_mol: np.ndarray
    #: ``expm1(-c2 * nu / 296)``, the stimulated-emission reference factor.
    stim_ref: np.ndarray
    max_abs_shift: float
    #: Width statistics of the whole list for sizing the multigrid mesh:
    #: 99.9th-percentile gamma_air and gamma_self (cm^-1/atm), the 0.1/99.9th
    #: percentile of n_air, and the largest Doppler coefficient
    #: ``alpha_D / sqrt(T)``. Derived from the list, not from the lines near a
    #: particular grid, so every spectral chunk chooses the same mesh.
    width_reference: tuple
    #: Per-line upper bounds on the Voigt peak cross-section for the
    #: peak-optical-depth pre-filter (see :func:`_peak_candidates`):
    #: ``S296 * sqrt(ln2/pi) / (alpha_D / sqrt(T))`` (Doppler limit, divide by
    #: sqrt(T)) and ``S296 / (pi * gamma_air)`` (Lorentz limit, divide by
    #: ``p_air * (296/T)**n``; infinite where gamma_air <= 0).
    peak_doppler: np.ndarray
    peak_lorentz: np.ndarray
    #: ``(min E'', min n_air, max n_air)`` over the list, and whether every
    #: gamma_self is >= 0 (the Lorentz bound drops the self-broadened term).
    bound_reference: tuple


_PREPARED: dict[int, _PreparedLines] = {}
_PREPARED_LOCK = threading.Lock()


def prepare_linelists(linelists: Mapping) -> None:
    """Build the cached per-list constants now (e.g. before forking workers)."""
    for linelist in linelists.values():
        _prepare(linelist)


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


def _width_reference(linelist: LineList) -> tuple:
    if len(linelist) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    gamma_air, gamma_self = np.percentile(linelist.gamma_air, 99.9), np.percentile(linelist.gamma_self, 99.9)
    n_lo, n_hi = np.percentile(linelist.n_air, [0.1, 99.9])
    doppler = linelist.nu_cm1 * np.sqrt(2.0 * _LN2 * GAS_CONSTANT_R
                                        / (_isotopologue_mass_g_mol(linelist) * 1.0e-3 * LIGHT_C**2))
    return (float(gamma_air), float(gamma_self), float(n_lo), float(n_hi), float(np.max(doppler)))


def _prepare(linelist: LineList) -> _PreparedLines:
    """Cached :class:`_PreparedLines` for ``linelist`` (thread-safe)."""
    arrays = tuple(getattr(linelist, name) for name in _LINE_FIELDS)
    key = id(linelist)
    cached = _PREPARED.get(key)
    if (cached is not None and cached.molecule_id == linelist.molecule_id
            and all(a is b for a, b in zip(cached.source_arrays, arrays))):
        return cached

    order = np.argsort(linelist.nu_cm1, kind="stable")
    unique_isos, iso_index = np.unique(linelist.local_iso_id, return_inverse=True)
    nu = np.ascontiguousarray(linelist.nu_cm1[order], dtype=np.float64)
    delta_air = np.ascontiguousarray(linelist.delta_air[order], dtype=np.float64)
    s296 = np.ascontiguousarray(linelist.s296[order], dtype=np.float64)
    gamma_air = np.ascontiguousarray(linelist.gamma_air[order], dtype=np.float64)
    gamma_self = np.ascontiguousarray(linelist.gamma_self[order], dtype=np.float64)
    e_lower = np.ascontiguousarray(linelist.e_lower[order], dtype=np.float64)
    n_air = np.ascontiguousarray(linelist.n_air[order], dtype=np.float64)
    m_kg_mol = _isotopologue_mass_g_mol(linelist)[order] * 1.0e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_doppler = s296 * _SQRT_LN2_OVER_PI / (
            nu * np.sqrt(2.0 * _LN2 * GAS_CONSTANT_R / (m_kg_mol * LIGHT_C**2)))
        peak_lorentz = np.where(gamma_air > 0.0, s296 / (np.pi * gamma_air), np.inf)
    prepared = _PreparedLines(
        source_arrays=arrays,
        molecule_id=int(linelist.molecule_id),
        nu=nu,
        s296=s296,
        gamma_air=gamma_air,
        gamma_self=gamma_self,
        e_lower=e_lower,
        n_air=n_air,
        delta_air=delta_air,
        iso_index=np.ascontiguousarray(iso_index.reshape(-1)[order]),
        unique_isos=tuple(int(i) for i in unique_isos),
        q_ref=np.array([partition_function(linelist.molecule_id, int(i), _T_REF_K)
                        for i in unique_isos]),
        m_kg_mol=m_kg_mol,
        stim_ref=np.expm1(-C2_RAD * nu / _T_REF_K),
        max_abs_shift=float(np.abs(delta_air).max()) if nu.size else 0.0,
        width_reference=_width_reference(linelist),
        peak_doppler=peak_doppler,
        peak_lorentz=peak_lorentz,
        bound_reference=((float(e_lower.min()), float(n_air.min()), float(n_air.max()),
                          bool(np.all(gamma_self >= 0.0))) if nu.size
                         else (0.0, 0.0, 0.0, True)),
    )
    with _PREPARED_LOCK:
        _PREPARED[key] = prepared
    weakref.finalize(linelist, _PREPARED.pop, key, None)
    return prepared


#: Q(T) per (molecule, isotopologues, T): every spectral chunk of a run
#: selects lines for the same layers (9500 table lookups, ~0.1 s, per full band).
_Q_CACHE: dict = {}
_Q_CACHE_SIZE = 1 << 16


def _partition_ratio(prepared: _PreparedLines, temperature_k: float) -> np.ndarray:
    """``Q(296)/Q(T)`` per isotopologue of ``prepared.unique_isos``."""
    key = (prepared.molecule_id, prepared.unique_isos, float(temperature_k))
    q_t = _Q_CACHE.get(key)
    if q_t is None:
        q_t = np.array([partition_function(prepared.molecule_id, iso, temperature_k)
                        for iso in prepared.unique_isos])
        if len(_Q_CACHE) >= _Q_CACHE_SIZE:
            _Q_CACHE.clear()
        _Q_CACHE[key] = q_t
    return prepared.q_ref / q_t


def _intensity(prepared: _PreparedLines, temperature_k: float,
               lines: slice | np.ndarray = slice(None),
               partition_ratio: Optional[np.ndarray] = None) -> np.ndarray:
    """S(T) for ``prepared`` lines (sorted order); see :func:`_line_intensity_at_t`."""
    if partition_ratio is None:
        partition_ratio = _partition_ratio(prepared, temperature_k)
    q_ratio = partition_ratio[prepared.iso_index[lines]]
    boltzmann = np.exp(-C2_RAD * prepared.e_lower[lines] * (1.0 / temperature_k - 1.0 / _T_REF_K))
    stim = np.expm1(-C2_RAD * prepared.nu[lines] / temperature_k) / prepared.stim_ref[lines]
    return prepared.s296[lines] * q_ratio * boltzmann * stim


def _line_intensity_at_t(linelist: LineList, temperature_k: float) -> np.ndarray:
    """Full HITRAN intensity temperature scaling S(T) from S(296 K).

    ``S(T) = S(296) * [Q(296)/Q(T)] * exp(-c2*E"*(1/T - 1/296))
             * (1 - exp(-c2*nu0/T)) / (1 - exp(-c2*nu0/296))``

    Returned in the LineList's own line order. Reduces exactly to
    ``S(296) == s296`` when ``temperature_k == 296.0`` (every factor above is
    identically 1 at T=296 K), which is asserted in ``tests/test_lbl.py``.
    """
    # (1 - exp(-x)) == -expm1(-x); the ratio of two such terms is the ratio
    # of the expm1(-x) values directly, and expm1 keeps precision for small x.
    if len(linelist) == 0:
        return np.zeros(0)
    prepared = _prepare(linelist)
    sorted_values = _intensity(prepared, temperature_k)
    out = np.empty_like(sorted_values)
    out[np.argsort(linelist.nu_cm1, kind="stable")] = sorted_values
    return out


def line_strength_reference(linelist: LineList, temperature_k: float) -> float:
    """Strongest line intensity S(T) in the whole list (the ``min_strength_fraction`` basis).

    Spectral chunks of one calculation pass this to
    :func:`absorption_cross_section` so every chunk prunes the same lines as
    an undivided evaluation would.
    """
    if len(linelist) == 0:
        return 0.0
    return float(_intensity(_prepare(linelist), temperature_k).max())


# =========================================================================
# Profile evaluation
# =========================================================================


@dataclass(frozen=True)
class _Lines:
    """Lines selected for one (layer, species) evaluation; all arrays per line."""

    center: np.ndarray        # pressure-shifted centre, cm^-1
    strength: np.ndarray      # S(T), cm^-1/(molecule cm^-2)
    alpha_d: np.ndarray       # Doppler HWHM, cm^-1
    gamma_l: np.ndarray       # Lorentz HWHM, cm^-1
    pedestal: Optional[np.ndarray]  # Lorentz value at 25 cm^-1, or None
    #: Voigt half-width (cm^-1) sizing the multigrid core, from the whole
    #: list's width statistics at these conditions (see _PreparedLines).
    plan_hwhm: float = 0.0


def _shape(dnu: np.ndarray, owner: np.ndarray, lines: _Lines) -> np.ndarray:
    """Voigt (optionally pedestal-subtracted) profile at offsets ``dnu``, cm.

    ``owner[k]`` is the index into ``lines`` of the line that ``dnu[k]`` belongs to.
    """
    alpha = lines.alpha_d[owner]
    x = _SQRT_LN2 * dnu / alpha
    y = (_SQRT_LN2 * lines.gamma_l / lines.alpha_d)[owner]
    shape = _SQRT_LN2_OVER_PI / alpha * wofz(x + 1j * y).real
    if lines.pedestal is not None:
        # HITRAN/MT_CKD local-line convention: the continuum already contains
        # this constant Lorentz wing pedestal. Do not renormalize the local line.
        shape = np.maximum(shape - lines.pedestal[owner], 0.0)
    return shape


def _ragged(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-owner integer ranges ``[starts, starts + counts)``.

    Returns ``(owner, value)``: for every element, which range it belongs to
    and its integer value. Owners appear in order, values ascending within each.
    """
    total = int(counts.sum())
    owner = np.repeat(np.arange(counts.size), counts)
    first = np.cumsum(counts) - counts
    value = np.arange(total, dtype=np.int64) + np.repeat(starts - first, counts)
    return owner, value


def _batches(counts: np.ndarray):
    """Consecutive owner ranges whose ragged size stays near ``_BATCH_ELEMENTS``."""
    ends = np.cumsum(counts)
    start = 0
    while start < counts.size:
        done = int(ends[start - 1]) if start else 0
        stop = int(np.searchsorted(ends, done + _BATCH_ELEMENTS, side="right"))
        stop = max(stop, start + 1)
        yield start, stop
        start = stop


def _subset(lines: _Lines, a: int, b: int) -> _Lines:
    return _Lines(lines.center[a:b], lines.strength[a:b], lines.alpha_d[a:b], lines.gamma_l[a:b],
                  None if lines.pedestal is None else lines.pedestal[a:b], lines.plan_hwhm)


def _lattice(grid: SpectralGrid) -> tuple[float, float, int, int]:
    """``(origin, step, first_index, n)``: grid point ``i`` sits at lattice index ``first_index + i``."""
    if grid.lattice_origin_cm1 is None:
        return float(grid.wavenumber_cm1[0]), float(grid.resolution_cm1), 0, grid.n
    return float(grid.lattice_origin_cm1), float(grid.resolution_cm1), int(grid.lattice_index0), grid.n


def _direct_sum(grid: SpectralGrid, lines: _Lines, cutoff: float) -> np.ndarray:
    """Exact truncated profiles summed at every grid point (strength-weighted)."""
    nu_grid = grid.wavenumber_cm1
    n = grid.n
    origin = nu_grid[0]
    step = grid.resolution_cm1
    left = np.maximum(np.ceil((lines.center - cutoff - origin) / step).astype(np.int64), 0)
    right = np.minimum(np.floor((lines.center + cutoff - origin) / step).astype(np.int64), n - 1)
    counts = np.maximum(right - left + 1, 0)
    out = np.zeros(n, dtype=np.float64)
    for a, b in _batches(counts):
        owner, idx = _ragged(left[a:b], counts[a:b])
        part = _subset(lines, a, b)
        dnu = nu_grid[idx] - part.center[owner]            # exact offset from centre
        contribution = part.strength[owner] * _shape(dnu, owner, part)
        contribution = np.where(np.abs(dnu) <= cutoff, contribution, 0.0)
        out += np.bincount(idx, weights=contribution, minlength=n)
    return out


@dataclass(frozen=True)
class _Plan:
    """Multigrid parameters: coarse mesh every ``m`` lattice points, core half-width."""

    m: int
    core_half_width: float


#: Coarse-mesh spacing is at most (core half-width)/_CORE_OVER_MESH. Cubic
#: interpolation of a Lorentzian wing at distance d from the centre has a
#: relative error of about 2.8*(H/d)**4, i.e. ~2e-4 of the local wing value
#: where the exact core ends, falling as d**-4 further out.
_CORE_OVER_MESH = 12.0
#: The exactly evaluated core spans at least this many Voigt half-widths.
_CORE_OVER_HWHM = 12.0
#: Balances fine-grid core cost against coarse-mesh wing cost (see _multigrid_plan).
_CORE_SCALE = 1.5


def _plan_for(step: float, cutoff: float, hwhm: float) -> Optional[_Plan]:
    """Multigrid parameters for a grid step, cutoff and line half-width (cm^-1),
    or None when direct evaluation is as cheap or cheaper.

    Per line, multigrid costs about ``2w/h`` exact core evaluations, ``2C/H``
    coarse wing evaluations and ``6m`` evaluations along the two cutoff edges
    (``h`` the grid step, ``H = m*h`` the mesh step, ``w`` the core half-width,
    ``C`` the cutoff). With ``H = w/12`` and core points costing ~3x a wing
    point in ``wofz``, the total is minimised near ``w = 1.5*sqrt(C*h)``.
    """
    core = max(_CORE_SCALE * np.sqrt(cutoff * step), _CORE_OVER_HWHM * hwhm)
    m = int(core / (_CORE_OVER_MESH * step))
    if m < 4 or core + 4.0 * m * step > cutoff:
        return None
    direct_cost = 2.0 * cutoff / step
    multigrid_cost = 3.0 * 2.0 * core / step + 2.0 * cutoff / (m * step) + 6.0 * m
    if multigrid_cost * 2.0 > direct_cost:
        return None
    return _Plan(m=m, core_half_width=core)


def evaluations_per_line(step_cm1: float, cutoff_cm1: float, hwhm_cm1: float = 0.1) -> float:
    """Approximate profile evaluations per line and layer, in wing-point units.

    Direct evaluation costs ``2C/h + 1``; where multigrid applies (see
    :func:`_plan_for`) its cheaper count is returned instead. Used by the CLI
    cost estimate; ``hwhm_cm1`` defaults to a typical near-surface width.
    """
    plan = _plan_for(step_cm1, cutoff_cm1, hwhm_cm1)
    direct = 2.0 * cutoff_cm1 / step_cm1 + 1.0
    if plan is None:
        return direct
    return (3.0 * 2.0 * plan.core_half_width / step_cm1
            + 2.0 * cutoff_cm1 / (plan.m * step_cm1) + 6.0 * plan.m)


def _multigrid_plan(grid: SpectralGrid, lines: _Lines, cutoff: float) -> Optional[_Plan]:
    """:func:`_plan_for` this grid, using the list-wide width (``lines.plan_hwhm``).

    The plan must not depend on which lines happen to lie near this grid --
    spectral chunks of one calculation would then disagree.
    """
    if lines.center.size == 0:
        return None
    return _plan_for(grid.resolution_cm1, cutoff, lines.plan_hwhm)


def _lagrange_weights(m: int) -> np.ndarray:
    """Four-point cubic Lagrange weights, shape ``(m, 4)``, for nodes -1, 0, 1, 2
    and evaluation points at fractional offsets ``t = 0, 1/m, ..., (m-1)/m``."""
    t = np.arange(m, dtype=np.float64) / m
    return np.stack([
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ], axis=1)


def _multigrid_sum(grid: SpectralGrid, lines: _Lines, cutoff: float, plan: _Plan) -> np.ndarray:
    """Truncated profiles summed on ``grid`` via a coarse wing mesh plus exact corrections.

    Every line ``l`` defines coarse samples ``W_l(j) = S*V(nu_j - c)`` at mesh
    nodes ``j`` with ``|nu_j - c| <= C`` (index-range test), zero elsewhere.
    The result is ``I[sum_l W_l] + sum_l (S*V*1_cutoff - I[W_l])`` where ``I``
    is the linear cubic-interpolation operator and each correction is applied
    over the line's core and cutoff-edge stretches. Outside those stretches
    the correction is the interpolation error of a smooth wing and is omitted.
    """
    origin, step, i0, n = _lattice(grid)
    m = plan.m
    mesh = m * step
    i1 = i0 + n - 1
    c = lines.center
    weights = _lagrange_weights(m)
    out = np.zeros(n, dtype=np.float64)

    def at(index):
        # Lattice wavenumber; identical expression to make_grid/chunk grids.
        return origin + step * index

    # Index range of mesh nodes inside each line's cutoff; defines W_l.
    j_lo = np.ceil((c - cutoff - origin) / mesh).astype(np.int64)
    j_hi = np.floor((c + cutoff - origin) / mesh).astype(np.int64)

    # --- 1. coarse wing mesh over the stencils of this grid's points ----------
    node0 = i0 // m - 1
    node1 = i1 // m + 2
    lo = np.maximum(j_lo, node0)
    counts = np.maximum(np.minimum(j_hi, node1) - lo + 1, 0)
    coarse = np.zeros(node1 - node0 + 1, dtype=np.float64)
    for a, b in _batches(counts):
        owner, node = _ragged(lo[a:b], counts[a:b])
        part = _subset(lines, a, b)
        values = part.strength[owner] * _shape(at(node * m) - part.center[owner], owner, part)
        coarse += np.bincount(node - node0, weights=values, minlength=coarse.size)

    g = np.arange(i0, i1 + 1, dtype=np.int64)
    j = g // m
    phase = g - j * m
    k = j - node0
    out += (weights[phase, 0] * coarse[k - 1] + weights[phase, 1] * coarse[k]
            + weights[phase, 2] * coarse[k + 1] + weights[phase, 3] * coarse[k + 2])

    # --- 2. exact corrections: core and both cutoff edges -----------------------
    core_lo = np.floor((c - plan.core_half_width - origin) / mesh).astype(np.int64)
    core_hi = np.floor((c + plan.core_half_width - origin) / mesh).astype(np.int64) + 1
    # Fine index windows [start, stop): where the stencil (nodes j-1..j+2) of
    # I[W_l] either spans the core or straddles the index-range edge of W_l.
    windows = (
        (core_lo * m, core_hi * m),
        ((j_lo - 2) * m, (j_lo + 1) * m),
        ((j_hi - 1) * m, (j_hi + 2) * m),
    )
    for start, stop in windows:
        start = np.maximum(start, i0)
        stop = np.minimum(stop, i1 + 1)
        counts = np.maximum(stop - start, 0)
        for a, b in _batches(counts):
            if not counts[a:b].any():
                continue
            part = _subset(lines, a, b)
            owner, gi = _ragged(start[a:b], counts[a:b])
            # This line's own mesh samples over its window's stencils.
            active = counts[a:b] > 0
            first_node = np.where(active, start[a:b] // m - 1, 0)
            node_counts = np.where(active, (np.maximum(stop[a:b], start[a:b] + 1) - 1) // m
                                   - start[a:b] // m + 4, 0)
            node_owner, node = _ragged(first_node, node_counts)
            inside = (node >= j_lo[a:b][node_owner]) & (node <= j_hi[a:b][node_owner])
            samples = _shape(at(node * m) - part.center[node_owner], node_owner, part)
            samples = np.where(inside, samples, 0.0)
            node_offset = np.cumsum(node_counts) - node_counts

            jj = gi // m
            ph = gi - jj * m
            base = node_offset[owner] + (jj - 1 - first_node[owner])
            interpolated = (weights[ph, 0] * samples[base] + weights[ph, 1] * samples[base + 1]
                            + weights[ph, 2] * samples[base + 2] + weights[ph, 3] * samples[base + 3])
            dnu = at(gi) - part.center[owner]
            exact = np.where(np.abs(dnu) <= cutoff, _shape(dnu, owner, part), 0.0)
            out += np.bincount(gi - i0, weights=part.strength[owner] * (exact - interpolated),
                               minlength=n)
    return out


# =========================================================================
# Line selection
# =========================================================================

#: Skip lines whose peak optical depth provably stays below the cutoff before
#: evaluating their widths and exact Voigt peak (see _peak_candidates). Off
#: only to check that the pre-filter changes nothing.
_PEAK_PREFILTER = True
#: Relative slack on the pre-filter bound, which is exact up to rounding.
_PREFILTER_MARGIN = 1.0e-6


def _peak_candidates(prepared: _PreparedLines, first: int, last: int, temperature_k: float,
                     p_air_atm: float, column_cm2: float, min_peak_optical_depth: float,
                     q_ratio: np.ndarray) -> Optional[np.ndarray]:
    """Indices in ``[first, last)`` of the lines that can reach ``min_peak_optical_depth``.

    Peak optical depth is ``column * S(T) * V(0)``, and the Voigt peak
    ``V(0) = sqrt(ln2/pi)/alpha_D * erfcx(y)`` is at most
    ``min(sqrt(ln2/pi)/alpha_D, 1/(pi*gamma_L))`` (``erfcx(y) <= 1`` and
    ``< 1/(y*sqrt(pi))``). With ``S(T) <= S296 * max Q(296)/Q(T) * (largest
    Boltzmann and stimulated-emission factors)`` and ``gamma_L >= gamma_air *
    p_air * min (296/T)**n``, both limits become comparisons of precomputed
    per-line numbers (``_PreparedLines.peak_doppler`` / ``peak_lorentz``) with
    one threshold per call, so S(T), the widths and ``erfcx`` are evaluated
    only for the lines that pass: ~16% of the scanned (line, layer) pairs of
    the default scenario, against ~10% finally kept. Every line the exact
    test keeps passes, so the selection is unchanged. Returns None when a
    bound does not apply; the caller then evaluates the whole window.
    """
    if last <= first or temperature_k <= 0.0:
        return None
    e_min, n_min, n_max, self_nonnegative = prepared.bound_reference
    nu_min = float(prepared.nu[first])
    if nu_min <= 0.0:
        return None
    inverse_delta = 1.0 / temperature_k - 1.0 / _T_REF_K
    # expm1(-c2 nu/T) / expm1(-c2 nu/296) is <= 1 for T >= 296, and otherwise
    # largest at the window's lowest wavenumber.
    stim = 1.0 if inverse_delta <= 0.0 else 1.0 / -np.expm1(-C2_RAD * nu_min / _T_REF_K)
    scale = column_cm2 * float(np.max(q_ratio)) * stim * (1.0 + _PREFILTER_MARGIN)
    window = slice(first, last)
    boltzmann = None
    if inverse_delta >= 0.0:
        # The Boltzmann factor falls with E'': the list's lowest E'' bounds it.
        scale *= float(np.exp(-C2_RAD * min(e_min, 0.0) * inverse_delta))
    else:
        boltzmann = np.exp(-C2_RAD * prepared.e_lower[window] * inverse_delta)
    if not (np.isfinite(scale) and scale > 0.0):
        return None

    doppler = prepared.peak_doppler[window]
    if boltzmann is not None:
        doppler = doppler * boltzmann
    keep = doppler >= min_peak_optical_depth * np.sqrt(temperature_k) / scale
    ratio = _T_REF_K / temperature_k
    width_floor = p_air_atm * ratio ** (n_min if ratio >= 1.0 else n_max)
    if self_nonnegative and width_floor > 0.0:
        lorentz = prepared.peak_lorentz[window]
        if boltzmann is not None:
            lorentz = lorentz * boltzmann
        keep &= lorentz >= min_peak_optical_depth * width_floor / scale
    return first + np.flatnonzero(keep)


def _pressures_and_window(prepared: _PreparedLines, grid_lo: float, grid_hi: float,
                          pressure_mbar: float, partial_pressure_mbar: float,
                          cutoff: float) -> tuple[float, float, float, int, int]:
    """``(p_total, p_self, p_air)`` in atm, and the ``[first, last)`` range of
    ``prepared`` lines whose shifted, truncated wings can reach ``[grid_lo, grid_hi]``."""
    p_total_atm = pressure_mbar / STD_PRESSURE_MBAR
    p_self_atm = max(partial_pressure_mbar, 0.0) / STD_PRESSURE_MBAR
    p_air_atm = max(p_total_atm - p_self_atm, 0.0)
    reach = cutoff + prepared.max_abs_shift * p_total_atm + 1.0e-9
    first = int(np.searchsorted(prepared.nu, grid_lo - reach, side="left"))
    last = int(np.searchsorted(prepared.nu, grid_hi + reach, side="right"))
    return p_total_atm, p_self_atm, p_air_atm, first, last


def _select_lines(prepared: _PreparedLines, grid: SpectralGrid, pressure_mbar: float,
                  temperature_k: float, partial_pressure_mbar: float, cutoff: float,
                  subtract_pedestal: bool, min_strength_fraction: float,
                  s_max: Optional[float], column_cm2: Optional[float],
                  min_peak_optical_depth: float) -> _Lines:
    """Lines whose truncated wings can reach ``grid``, with their profile parameters."""
    p_total_atm, p_self_atm, p_air_atm, first, last = _pressures_and_window(
        prepared, grid.wavenumber_cm1[0], grid.wavenumber_cm1[-1], pressure_mbar,
        partial_pressure_mbar, cutoff)
    q_ratio = _partition_ratio(prepared, temperature_k)
    window: slice | np.ndarray = slice(first, last)
    prefiltered = False
    if _PEAK_PREFILTER and min_peak_optical_depth > 0.0 and column_cm2 is not None:
        candidates = _peak_candidates(prepared, first, last, temperature_k, p_air_atm,
                                      column_cm2, min_peak_optical_depth, q_ratio)
        if candidates is not None:
            window, prefiltered = candidates, True

    s_t = _intensity(prepared, temperature_k, window, q_ratio)
    nu_shifted = prepared.nu[window] + prepared.delta_air[window] * p_total_atm
    keep = np.abs(nu_shifted - np.clip(nu_shifted, grid.wavenumber_cm1[0],
                                       grid.wavenumber_cm1[-1])) <= cutoff
    if min_strength_fraction > 0.0:
        if s_max is None:
            everything = not prefiltered and first == 0 and last == prepared.nu.size
            s_max = float((s_t if everything else _intensity(prepared, temperature_k)).max())
        if s_max > 0.0:
            keep &= s_t >= (min_strength_fraction * s_max)

    # Doppler HWHM, cm^-1: nu0 * sqrt(2 ln2 R T / (M c^2)), isotopologue mass.
    alpha_d = prepared.nu[window] * np.sqrt(
        2.0 * _LN2 * GAS_CONSTANT_R * temperature_k / (prepared.m_kg_mol[window] * LIGHT_C**2)
    )
    # Lorentz (pressure-broadened) HWHM.
    gamma_l = (prepared.gamma_air[window] * p_air_atm + prepared.gamma_self[window] * p_self_atm) * (
        _T_REF_K / temperature_k
    ) ** prepared.n_air[window]

    if min_peak_optical_depth > 0.0 and column_cm2 is not None:
        # Exact Voigt peak: sqrt(ln2/pi)/alpha_D * Re w(i*y) = ... * erfcx(y).
        peak = _SQRT_LN2_OVER_PI / alpha_d * erfcx(_SQRT_LN2 * gamma_l / alpha_d)
        keep &= s_t * column_cm2 * peak >= min_peak_optical_depth

    idx = np.flatnonzero(keep)
    gamma_l = gamma_l[idx]
    pedestal = (gamma_l / (np.pi * (_MT_CKD_CUTOFF_CM1**2 + gamma_l**2))
                if subtract_pedestal else None)

    g_air, g_self, n_lo, n_hi, doppler = prepared.width_reference
    exponent = n_hi if temperature_k < _T_REF_K else n_lo      # maximises (296/T)**n
    gamma_ref = (g_air * p_air_atm + g_self * p_self_atm) * (_T_REF_K / temperature_k) ** exponent
    alpha_ref = doppler * np.sqrt(temperature_k)
    plan_hwhm = float(0.5346 * gamma_ref + np.sqrt(0.2166 * gamma_ref**2 + alpha_ref**2))
    return _Lines(center=nu_shifted[idx], strength=s_t[idx],
                  alpha_d=alpha_d[idx], gamma_l=gamma_l, pedestal=pedestal, plan_hwhm=plan_hwhm)


# =========================================================================
# Public entry points
# =========================================================================


def absorption_cross_section(
    linelist: LineList,
    grid: SpectralGrid,
    pressure_mbar: float,
    temperature_k: float,
    partial_pressure_mbar: float,
    *,
    wing_cutoff_cm1: float = 25.0,
    min_strength_fraction: float = 0.0,
    subtract_pedestal: bool = False,
    method: str = "direct",
    strength_reference: Optional[float] = None,
    column_cm2: Optional[float] = None,
    min_peak_optical_depth: float = 0.0,
) -> np.ndarray:
    """Voigt cross-section, cm^2/molecule, shape ``(grid.n,)``.

    ``pressure_mbar`` is the total (broadening) pressure of the layer at the
    species' Curtis-Godson effective point (``PathLayer.p_eff_mbar[species]``);
    ``partial_pressure_mbar`` is the absorber's own partial (self) pressure
    at that same point. Air (foreign) broadening uses
    ``pressure_mbar - partial_pressure_mbar``, clamped at zero.

    Each line contributes only within ``+/-wing_cutoff_cm1`` of its
    (pressure-shifted) centre. ``method`` is ``"direct"`` (default: exact
    evaluation at every grid point), ``"multigrid"`` (coarse-mesh wings with
    exact core/edge corrections; falls back to direct where it would not pay
    off) or ``"auto"`` (same as ``"multigrid"``). See the module docstring.

    ``subtract_pedestal=True`` subtracts the Lorentz value at +/-25 cm^-1
    before summation, as required when combining local H2O lines with MT_CKD.
    It requires a 25 cm^-1 cutoff; the default returns an ordinary Voigt profile.

    ``min_strength_fraction`` (default 0: off) drops lines whose
    temperature-scaled intensity is below this fraction of the strongest line
    of the whole list at this temperature (``strength_reference``, computed if
    not given) before the Voigt evaluation. The full 1775-4325 cm^-1 HITRAN
    fetch for some species is dominated by extremely weak hot/combination-band
    lines: O3 and N2O each return 200k+ lines, but 99.9%+ of total line
    strength is concentrated in a few percent of them (at 1e-4, every species
    in this package's fetch loses < 1% of summed line strength). **Summed strength is the wrong measure in
    the windows**: at 1e-4 the rule discards every H2O line between 2440 and
    2860 cm^-1 (main isotopologue and HDO alike are weaker than 1e-4 of the
    2.7 um band's strongest line), which overstated 3.5-4.1 um transmittance
    by 24 percentage points on a 10 km tropical path. Prefer
    ``min_peak_optical_depth``.

    ``min_peak_optical_depth`` (with ``column_cm2``, the absorber column the
    cross-section will multiply) drops lines whose peak optical depth
    ``S(T) * column * V(0)`` is below the threshold -- the LBLRTM approach.
    The criterion depends on the actual amount of gas, so a thick humid layer
    keeps weak water lines that matter in the windows while a thin layer drops
    lines that cannot. The Voigt peak uses ``erfcx`` and is exact.
    """
    if subtract_pedestal and wing_cutoff_cm1 != _MT_CKD_CUTOFF_CM1:
        raise ValueError("MT_CKD pedestal subtraction requires a 25 cm^-1 cutoff")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if len(linelist) == 0:
        return np.zeros(grid.n, dtype=np.float64)

    lines = _select_lines(_prepare(linelist), grid, pressure_mbar, temperature_k,
                          partial_pressure_mbar, wing_cutoff_cm1, subtract_pedestal,
                          min_strength_fraction, strength_reference, column_cm2,
                          min_peak_optical_depth)
    if lines.center.size == 0:
        return np.zeros(grid.n, dtype=np.float64)
    plan = _multigrid_plan(grid, lines, wing_cutoff_cm1) if method != "direct" else None
    if plan is None:
        return _direct_sum(grid, lines, wing_cutoff_cm1)
    return _multigrid_sum(grid, lines, wing_cutoff_cm1, plan)


def molecular_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
    *,
    species: Sequence[Species] | None = None,
    linelists: dict[Species, LineList] | None = None,
    include_continuum: bool = True,
    wing_cutoff_cm1: float = 25.0,
    min_strength_fraction: float = 0.0,
    method: str = "auto",
    n_threads: Optional[int] = None,
    strength_references: Optional[Mapping[tuple[int, Species], float]] = None,
    min_peak_optical_depth: float = DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF,
    backend: Optional[str] = None,
) -> tuple[np.ndarray, dict[Species, np.ndarray]]:
    """THE headline function. Returns ``(tau_total, tau_by_species)``, each
    shape ``(len(layers), grid.n)``.

    ``linelists`` defaults to :func:`mwir_atmos.hitran.fetch.ensure_linelists`
    (loads the shipped cache; missing files raise without network access).
    ``species`` defaults to :data:`mwir_atmos.types.REQUIRED_SPECIES` (the
    six species every profile must supply -- excludes O2, which has no
    HITRAN line data fetched for this package and negligible MWIR line
    absorption anyway).

    With continuum enabled, H2O always uses MT_CKD's required 25 cm^-1
    local-line cutoff and pedestal subtraction. Other species use the requested
    ``wing_cutoff_cm1``. With continuum disabled, all profiles are ordinary Voigt.

    Per layer and species, broadening/shift use the Curtis-Godson effective
    pressure and temperature (``PathLayer.p_eff_mbar`` / ``t_eff_k``); the
    species' self partial pressure is derived from its column fraction of
    the total air column (``column_cm2[species] / air_column_cm2``), i.e.
    the path-mean volume mixing ratio, times that effective pressure.

    ``method`` is passed to :func:`absorption_cross_section` (default
    ``"auto"``). ``(layer, species)`` pairs run on ``n_threads`` threads (see
    :func:`resolve_thread_count`); results are independent of the thread count.
    ``strength_references`` optionally maps ``(layer row, species)`` to the
    ``min_strength_fraction`` basis (see :func:`line_strength_reference`).

    Line selection defaults to the per-layer peak-optical-depth criterion
    (``min_peak_optical_depth``, see :func:`absorption_cross_section`). It
    depends on each layer's column, so optical depth is additive over split
    layers and linear in path length only to within the dropped lines' peak
    optical depths. Pass ``min_peak_optical_depth=0.0`` (and the default
    ``min_strength_fraction=0.0``) for the exact, unpruned model.

    ``backend`` is ``"cpu"`` or ``"cuda"`` (default: :func:`resolve_backend`).
    With ``"cuda"`` the CPU selects lines exactly as above and a GPU evaluates
    every (layer, species) pair, always by direct summation
    (:mod:`mwir_atmos.optics.lbl_cuda_dispatch`), in groups of tasks so that
    selection overlaps the kernels; ``method`` then does not apply, and
    ``n_threads`` sets the line-selection threads (default: ``MWIR_THREADS``,
    else ``lbl_cuda_dispatch._CUDA_SELECTION_THREADS``). A grid whose ``wavenumber_cm1`` is a CuPy
    array (``"cuda"`` only) gives CuPy results that never leave the GPU.

    The per-species arrays are views into one ``(species, layer, point)``
    array, and ``tau_total`` is their sum.
    """
    if species is None:
        species = REQUIRED_SPECIES
    if linelists is None:
        linelists = ensure_linelists()
    backend = resolve_backend(backend)
    xp = array_namespace(grid.wavenumber_cm1)
    if xp is not np and backend != "cuda":
        raise ValueError("a grid of CuPy arrays needs backend='cuda'")

    n_layers = len(layers)
    species = list(dict.fromkeys(species))
    tasks = []
    for row, layer in enumerate(layers):
        for sp in species:
            column = layer.column_cm2.get(sp, 0.0)
            linelist = linelists.get(sp)
            if column > 0.0 and linelist is not None:
                tasks.append((row, sp, column, linelist))

    def arguments(task) -> dict:
        row, sp, column, linelist = task
        layer = layers[row]
        p_eff = layer.p_eff_mbar.get(sp, layer.p_mbar)
        vmr = column / layer.air_column_cm2 if layer.air_column_cm2 > 0 else 0.0
        water_convention = sp == Species.H2O and include_continuum
        return dict(
            pressure_mbar=p_eff,
            temperature_k=layer.t_eff_k.get(sp, layer.t_k),
            partial_pressure_mbar=p_eff * vmr,
            wing_cutoff_cm1=_MT_CKD_CUTOFF_CM1 if water_convention else wing_cutoff_cm1,
            subtract_pedestal=water_convention,
            min_strength_fraction=min_strength_fraction,
            strength_reference=(None if strength_references is None
                                else strength_references.get((row, sp))),
            column_cm2=column,
            min_peak_optical_depth=min_peak_optical_depth,
        )

    # One (species, layer, point) block; each species' array is a view into it.
    if backend == "cuda":
        from .lbl_cuda_dispatch import cuda_optical_depth
        block = cuda_optical_depth(tasks, arguments, grid, len(species) * n_layers,
                                   {sp: k * n_layers for k, sp in enumerate(species)}, n_threads)
        block = block.reshape(len(species), n_layers, grid.n)
        tau_by_species = {sp: block[k] for k, sp in enumerate(species)}
    else:
        block = np.zeros((len(species), n_layers, grid.n), dtype=np.float64)
        tau_by_species = {sp: block[k] for k, sp in enumerate(species)}

        def evaluate(task) -> None:
            row, sp, column, linelist = task
            xsec = absorption_cross_section(linelist, grid, method=method, **arguments(task))
            tau_by_species[sp][row] = xsec * column

        _run_on_threads(evaluate, tasks, n_threads)

    if include_continuum:
        if Species.H2O in tau_by_species:
            tau_by_species[Species.H2O] += h2o_continuum_optical_depth(layers, grid)
        if Species.CO2 in tau_by_species:
            tau_by_species[Species.CO2] += co2_continuum_optical_depth(layers, grid)

    # Same summation order as adding the species one by one.
    tau_total = block.sum(axis=0)
    return tau_total, tau_by_species


def _run_on_threads(evaluate, tasks: list, n_threads: Optional[int]) -> None:
    """``evaluate(task)`` for every ``(row, species, column, linelist)`` task."""
    workers = min(resolve_thread_count(n_threads), len(tasks))
    if workers <= 1:
        for task in tasks:
            evaluate(task)
        return
    # Largest line lists first, so no long task starts last.
    tasks = sorted(tasks, key=lambda task: -len(task[3]))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mwir-lbl") as pool:
        for _ in pool.map(evaluate, tasks):
            pass
