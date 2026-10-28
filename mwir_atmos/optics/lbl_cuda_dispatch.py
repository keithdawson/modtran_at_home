"""CUDA backend orchestration for :func:`mwir_atmos.optics.lbl.molecular_optical_depth`.

With ``backend="cuda"``, :func:`cuda_optical_depth` evaluates every
``(layer, species)`` task of a spectral chunk on the GPU:

1. **Line selection** -- by default on the GPU (:func:`_select_packed`, one
   pass per line list), with the arithmetic of the CPU backend's
   :func:`~mwir_atmos.optics.lbl._select_lines`. With
   ``MWIR_CUDA_SELECTION=cpu``, ``_select_lines`` itself runs on a few CPU
   threads, in groups of tasks so that the CPU selects group k+1 while the
   GPU evaluates group k.
2. **Evaluation** -- :func:`mwir_atmos.optics.lbl_cuda.evaluate_groups`
   launches the Voigt kernel on each packed group.

:mod:`mwir_atmos.optics.lbl_cuda` holds the kernel itself and
:mod:`mwir_atmos.optics.voigt_gpu` its Voigt profile.
"""

from __future__ import annotations

import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.special import erfcx

from .._array import to_host
from ..constants import C2_RAD, GAS_CONSTANT_R, LIGHT_C
from ..types import Species, SpectralGrid
from . import lbl_cuda, lbl_cuda_select
from .lbl import (
    _LN2,
    _MT_CKD_CUTOFF_CM1,
    _SQRT_LN2,
    _SQRT_LN2_OVER_PI,
    _T_REF_K,
    _PreparedLines,
    _intensity,
    _partition_ratio,
    _prepare,
    _pressures_and_window,
    _select_lines,
    resolve_thread_count,
)

__all__ = ["cuda_optical_depth", "release_selection_memory"]


#: The (layer, species) tasks of one CUDA evaluation are split into this many
#: groups: the CPU selects lines for group k+1 while the GPU evaluates group k,
#: so only the first group's selection is not hidden behind the kernel.
_CUDA_TASK_GROUPS = 8
#: Line-selection threads for the CUDA backend when neither ``n_threads`` nor
#: ``MWIR_THREADS`` says otherwise. Selection hides behind the kernels of a
#: resolved run even on one thread (full default band on an RTX 4090: 0.96 s
#: on 1 thread, 0.97 s on 16); point sampling, where the kernel is tiny, takes
#: 0.34 s on 4 threads and 0.32 s on 16. The rest of a shared machine stays free.
_CUDA_SELECTION_THREADS = 4


def _cuda_selection_threads(n_threads: Optional[int]) -> int:
    if n_threads is None and not os.environ.get("MWIR_THREADS", "").strip():
        return min(_CUDA_SELECTION_THREADS, os.cpu_count() or 1)
    return resolve_thread_count(n_threads)


def _gpu_selection_enabled() -> bool:
    """Where the CUDA backend selects lines: ``MWIR_CUDA_SELECTION``, ``gpu`` (default) or ``cpu``.

    GPU selection (:func:`_select_packed`) took the full default band from
    0.89 s to 0.55 s on an RTX 4090, with identical results; ``cpu`` keeps
    the threaded CPU selection (``_CUDA_SELECTION_THREADS``).
    """
    value = os.environ.get("MWIR_CUDA_SELECTION", "").strip().lower() or "gpu"
    if value not in ("gpu", "cpu"):
        raise ValueError(f"MWIR_CUDA_SELECTION must be 'gpu' or 'cpu', got {value!r}")
    return value == "gpu"


#: Largest number of (task, window line) pairs selected in one pass of
#: :func:`_select_packed`; its temporaries are 8 bytes per pair each.
_SELECT_PAIRS = 1 << 23
#: CuPy selection through the two fused kernels of :mod:`lbl_cuda_select`
#: rather than ~40 array operations per pass. Off only for comparisons.
_SELECT_FUSED = True
#: Per-line arrays of _PreparedLines that _select_packed reads.
_SELECT_FIELDS = ("nu", "delta_air", "s296", "gamma_air", "gamma_self", "e_lower", "n_air",
                  "m_kg_mol", "stim_ref", "iso_index")
_DEVICE_LINES: dict = {}
_DEVICE_LINES_LOCK = threading.Lock()


def release_selection_memory() -> None:
    """Forget the line lists uploaded for GPU selection (see :func:`_device_lines`)."""
    with _DEVICE_LINES_LOCK:
        _DEVICE_LINES.clear()


def _device_lines(xp, prepared: _PreparedLines) -> dict:
    """``prepared``'s :data:`_SELECT_FIELDS` as ``xp`` arrays, uploaded once per GPU."""
    if xp is np:
        return {name: getattr(prepared, name) for name in _SELECT_FIELDS}
    key = (id(prepared), xp.cuda.Device().id)
    with _DEVICE_LINES_LOCK:
        cached = _DEVICE_LINES.get(key)
        if cached is None or cached[0] is not prepared:
            cached = (prepared, {name: xp.asarray(getattr(prepared, name))
                                 for name in _SELECT_FIELDS})
            _DEVICE_LINES[key] = cached
            weakref.finalize(prepared, _DEVICE_LINES.pop, key, None)
    return cached[1]


@dataclass(frozen=True)
class _SelectionTask:
    """Host-side scalars of one (layer, species) selection; see :func:`_select_packed`."""

    first: int                 # window of prepared lines, as in _select_lines
    last: int
    temperature_k: float
    p_total_atm: float
    p_self_atm: float
    p_air_atm: float
    cutoff: float
    column_cm2: float
    pedestal: bool
    q_ratio: np.ndarray        # Q(296)/Q(T) per prepared.unique_isos
    strength_threshold: Optional[float]   # min_strength_fraction * s_max, if applied
    min_peak_optical_depth: Optional[float]  # if applied


def _selection_task(prepared: _PreparedLines, grid_lo, grid_hi, pressure_mbar: float,
                    temperature_k: float, partial_pressure_mbar: float, cutoff: float,
                    subtract_pedestal: bool, min_strength_fraction: float,
                    s_max: Optional[float], column_cm2: float,
                    min_peak_optical_depth: float) -> _SelectionTask:
    """The per-task part of :func:`_select_lines` (same arguments, same arithmetic)."""
    p_total_atm, p_self_atm, p_air_atm, first, last = _pressures_and_window(
        prepared, grid_lo, grid_hi, pressure_mbar, partial_pressure_mbar, cutoff)
    q_ratio = _partition_ratio(prepared, temperature_k)
    threshold = None
    if min_strength_fraction > 0.0:
        if s_max is None:
            s_max = float(_intensity(prepared, temperature_k, partition_ratio=q_ratio).max())
        if s_max > 0.0:
            threshold = min_strength_fraction * s_max
    return _SelectionTask(
        first=first, last=last, temperature_k=float(temperature_k), p_total_atm=p_total_atm,
        p_self_atm=p_self_atm, p_air_atm=p_air_atm, cutoff=float(cutoff),
        column_cm2=float(column_cm2), pedestal=bool(subtract_pedestal), q_ratio=q_ratio,
        strength_threshold=threshold,
        min_peak_optical_depth=(min_peak_optical_depth if min_peak_optical_depth > 0.0
                                else None))


def _erfcx(xp):
    if xp is np:
        return erfcx
    import cupyx.scipy.special
    return cupyx.scipy.special.erfcx


def _use_fused(xp) -> bool:
    """Whether :func:`_select_packed` uses the fused kernels (CuPy only; tests run them in C++)."""
    return xp is not np and _SELECT_FUSED


def _select_packed(xp, prepared: _PreparedLines, tasks: Sequence[_SelectionTask],
                   grid_lo: float, grid_hi: float):
    """:func:`_select_lines` then :func:`lbl_cuda.pack_task` for tasks of one line list, in ``xp``.

    Every (task, window line) pair is one element of a flat array; the same
    per-element operations as the CPU path, in the same order, decide which
    lines are kept and give their packed parameters. With NumPy the result
    equals the CPU path exactly; with CuPy, every basic operation is one
    correctly rounded kernel per op (no fused multiply-add across them), so
    only ``exp``, ``expm1``, ``pow`` and ``erfcx`` can differ by an ulp or so
    (a line within that of a threshold may then flip). The peak-optical-depth
    pre-filter is skipped: it removes nothing the exact test keeps.

    Returns an ``lbl_cuda.Packed`` whose line arrays and ``start``/``stop``
    are ``xp`` arrays (``cutoff``/``has_pedestal`` stay NumPy).
    """
    from .lbl_cuda import Packed

    n_tasks = len(tasks)
    counts = np.array([t.last - t.first for t in tasks], dtype=np.int64)
    ends = np.cumsum(counts)
    total = int(ends[-1]) if n_tasks else 0
    cutoff = np.array([t.cutoff for t in tasks], dtype=np.float64)
    has_pedestal = np.array([t.pedestal for t in tasks], dtype=np.int32)

    # Every per-task scalar in one upload: a host-to-device copy from pageable
    # memory waits for the work already queued (the previous group's kernels).
    columns = {
        "first_minus_offset": np.array([t.first for t in tasks], dtype=np.float64) - (ends - counts),
        "end": ends, "temperature": [t.temperature_k for t in tasks],
        "inverse_delta": [1.0 / t.temperature_k - 1.0 / _T_REF_K for t in tasks],
        "p_total": [t.p_total_atm for t in tasks], "cutoff": cutoff,
        "strength_threshold": [t.strength_threshold or 0.0 for t in tasks],
        "strength_applies": [t.strength_threshold is not None for t in tasks],
        "doppler": [2.0 * _LN2 * GAS_CONSTANT_R * t.temperature_k for t in tasks],
        "p_air": [t.p_air_atm for t in tasks], "p_self": [t.p_self_atm for t in tasks],
        "ratio": [_T_REF_K / t.temperature_k for t in tasks],
        "column": [t.column_cm2 for t in tasks],
        "peak_minimum": [t.min_peak_optical_depth or 0.0 for t in tasks],
        "peak_applies": [t.min_peak_optical_depth is not None for t in tasks],
        "pedestal": has_pedestal,
    }
    assert tuple(columns) == lbl_cuda_select.TASK_COLUMNS
    constants = dict(c2=C2_RAD, light_c_squared=LIGHT_C**2, sqrt_ln2=_SQRT_LN2,
                     sqrt_ln2_over_pi=_SQRT_LN2_OVER_PI, pi=np.pi,
                     mt_ckd_cutoff_squared=_MT_CKD_CUTOFF_CM1**2)
    assert tuple(constants) == lbl_cuda_select.CONSTANTS
    n_iso = len(prepared.unique_isos)
    q_ratio = np.concatenate([t.q_ratio for t in tasks]) if n_tasks else np.zeros(0)
    table = np.empty(len(columns) * n_tasks + q_ratio.size + len(constants), dtype=np.float64)
    for k, values in enumerate(columns.values()):
        table[k * n_tasks:(k + 1) * n_tasks] = values
    table[len(columns) * n_tasks:len(columns) * n_tasks + q_ratio.size] = q_ratio
    table[len(columns) * n_tasks + q_ratio.size:] = list(constants.values())
    table = xp.asarray(table)
    index = {name: k for k, name in enumerate(columns)}
    lines = _device_lines(xp, prepared)
    use_strength = any(t.strength_threshold is not None for t in tasks)
    use_peak = any(t.min_peak_optical_depth is not None for t in tasks)

    if _use_fused(xp):
        center, owner, inv_alpha, y, amp, pedestal = lbl_cuda_select.select(
            xp, table, n_tasks, n_iso, lines, total, grid_lo, grid_hi, use_strength, use_peak)
        order = xp.argsort(center, kind="stable")
        order = order[xp.argsort(owner[order], kind="stable")]
        owner = owner[order]
        n_kept = (xp.bincount(owner, minlength=n_tasks) if owner.size
                  else xp.zeros(n_tasks, dtype=np.int64))
        stop = xp.cumsum(n_kept).astype(np.int64)
        return Packed(center=center[order], inv_alpha=inv_alpha[order], y=y[order],
                      amp=amp[order], pedestal=pedestal[order],
                      start=stop - n_kept, stop=stop, cutoff=cutoff, has_pedestal=has_pedestal)

    def per_task(name, dtype=np.float64):
        k = index[name]
        values = table[k * n_tasks:(k + 1) * n_tasks]
        return values if dtype is np.float64 else values.astype(dtype)

    pair = xp.arange(total, dtype=np.int64)
    owner = xp.searchsorted(per_task("end", np.int64), pair, side="right")
    line = pair + per_task("first_minus_offset", np.int64)[owner]
    del pair

    temperature = per_task("temperature")[owner]
    nu = lines["nu"][line]
    q_ratio = table[len(columns) * n_tasks + owner * n_iso + lines["iso_index"][line]]
    # _intensity: S(296) * Q ratio * Boltzmann * stimulated emission.
    inverse_delta = per_task("inverse_delta")[owner]
    boltzmann = xp.exp(-C2_RAD * lines["e_lower"][line] * inverse_delta)
    stim = xp.expm1(-C2_RAD * nu / temperature) / lines["stim_ref"][line]
    s_t = lines["s296"][line] * q_ratio * boltzmann * stim
    del q_ratio, boltzmann, stim, inverse_delta

    nu_shifted = nu + lines["delta_air"][line] * per_task("p_total")[owner]
    keep = xp.abs(nu_shifted - xp.clip(nu_shifted, grid_lo, grid_hi)) <= per_task("cutoff")[owner]
    if use_strength:
        threshold = per_task("strength_threshold")[owner]
        applies = per_task("strength_applies", bool)[owner]
        keep &= ~applies | (s_t >= threshold)
        del threshold, applies

    doppler = per_task("doppler")[owner]
    alpha_d = nu * xp.sqrt(doppler / (lines["m_kg_mol"][line] * LIGHT_C**2))
    del doppler, nu, temperature
    gamma_l = (lines["gamma_air"][line] * per_task("p_air")[owner]
               + lines["gamma_self"][line] * per_task("p_self")[owner]) * (
        per_task("ratio")[owner] ** lines["n_air"][line])
    column = per_task("column")[owner]
    if use_peak:
        peak = _SQRT_LN2_OVER_PI / alpha_d * _erfcx(xp)(_SQRT_LN2 * gamma_l / alpha_d)
        minimum = per_task("peak_minimum")[owner]
        applies = per_task("peak_applies", bool)[owner]
        keep &= ~applies | (s_t * column * peak >= minimum)
        del peak, minimum, applies

    kept = xp.flatnonzero(keep)
    owner = owner[kept]
    center = nu_shifted[kept]
    # Per task, ascending centre; ties in the original (wavenumber) order, as
    # pack_task's stable argsort. Two stable sorts (radix sorts in CuPy).
    order = xp.argsort(center, kind="stable")
    order = order[xp.argsort(owner[order], kind="stable")]
    kept, owner, center = kept[order], owner[order], center[order]
    alpha = alpha_d[kept]
    gamma = gamma_l[kept]
    scale = column[kept] * s_t[kept]
    pedestal = gamma / (np.pi * (_MT_CKD_CUTOFF_CM1**2 + gamma * gamma))
    pedestal = xp.where(per_task("pedestal", bool)[owner], scale * pedestal, 0.0)

    n_kept = (xp.bincount(owner, minlength=n_tasks) if owner.size
              else xp.zeros(n_tasks, dtype=np.int64))
    stop = xp.cumsum(n_kept).astype(np.int64)
    return Packed(center=center,
                  inv_alpha=(_SQRT_LN2 / alpha).astype(np.float32),
                  y=(_SQRT_LN2 * gamma / alpha).astype(np.float32),
                  amp=(scale * _SQRT_LN2_OVER_PI / alpha).astype(np.float32),
                  pedestal=pedestal.astype(np.float32),
                  start=stop - n_kept, stop=stop, cutoff=cutoff, has_pedestal=has_pedestal)


def _gpu_selected_groups(xp, tasks: list, arguments, host_grid: SpectralGrid,
                         row0: Mapping[Species, int]):
    """``(packed, rows)`` groups for :func:`lbl_cuda.evaluate_groups`, selected in ``xp``.

    One group per line list (and per :data:`_SELECT_PAIRS` pairs); the CPU
    only finds each task's window and partition-function ratios.
    """
    grid_lo, grid_hi = host_grid.wavenumber_cm1[0], host_grid.wavenumber_cm1[-1]
    by_list: dict[int, list] = {}
    for task in tasks:
        by_list.setdefault(id(task[3]), []).append(task)
    for same_list in by_list.values():
        prepared = _prepare(same_list[0][3])
        batch, rows, pairs = [], [], 0
        for task in same_list:
            a = arguments(task)
            spec = _selection_task(
                prepared, grid_lo, grid_hi, a["pressure_mbar"], a["temperature_k"],
                a["partial_pressure_mbar"], a["wing_cutoff_cm1"], a["subtract_pedestal"],
                a["min_strength_fraction"], a["strength_reference"], a["column_cm2"],
                a["min_peak_optical_depth"])
            if batch and pairs + (spec.last - spec.first) > _SELECT_PAIRS:
                yield _select_packed(xp, prepared, batch, grid_lo, grid_hi), rows
                batch, rows, pairs = [], [], 0
            batch.append(spec)
            rows.append(row0[task[1]] + task[0])
            pairs += spec.last - spec.first
        if batch:
            yield _select_packed(xp, prepared, batch, grid_lo, grid_hi), rows


def cuda_optical_depth(tasks: list, arguments, grid: SpectralGrid, n_rows: int,
                        row0: Mapping[Species, int], n_threads: Optional[int]):
    """Optical depths of every ``(row, species, column, linelist)`` task on the GPU.

    Returns ``(n_rows, grid.n)`` with task ``(row, sp)`` in row ``row0[sp] + row``,
    other rows zero; a CuPy array when the grid is on the device, NumPy otherwise.
    Lines are selected on the GPU (:func:`_gpu_selected_groups`), or with
    ``MWIR_CUDA_SELECTION=cpu`` on the CPU exactly as for the CPU backend.
    """
    tasks = [task for task in tasks if len(task[3])]
    wavenumber = to_host(grid.wavenumber_cm1)
    host_grid = SpectralGrid(wavenumber_cm1=wavenumber, resolution_cm1=grid.resolution_cm1,
                             lattice_origin_cm1=grid.lattice_origin_cm1,
                             lattice_index0=grid.lattice_index0)
    if _gpu_selection_enabled() and tasks:
        return lbl_cuda.evaluate_groups(
            grid.wavenumber_cm1,
            _gpu_selected_groups(lbl_cuda.selection_module(), tasks, arguments, host_grid, row0),
            n_rows)

    def select(task):
        a = arguments(task)
        lines = _select_lines(
            _prepare(task[3]), host_grid, a["pressure_mbar"], a["temperature_k"],
            a["partial_pressure_mbar"], a["wing_cutoff_cm1"], a["subtract_pedestal"],
            a["min_strength_fraction"], a["strength_reference"], a["column_cm2"],
            a["min_peak_optical_depth"])
        return lbl_cuda.pack_task(lines, a["wing_cutoff_cm1"], task[2])

    # Costliest line lists dealt round-robin, so every group has a similar mix.
    tasks.sort(key=lambda task: -len(task[3]))
    n_groups = max(1, min(_CUDA_TASK_GROUPS, len(tasks)))
    groups = [tasks[g::n_groups] for g in range(n_groups)] if tasks else []
    workers = min(_cuda_selection_threads(n_threads), max(len(groups[0]) if groups else 1, 1))

    def produce(pool):
        for group in groups:
            packed = list(pool.map(select, group)) if pool else [select(t) for t in group]
            yield lbl_cuda.concatenate(packed), [row0[sp] + row for row, sp, _, _ in group]

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mwir-select") as pool:
            return lbl_cuda.evaluate_groups(grid.wavenumber_cm1, produce(pool), n_rows)
    return lbl_cuda.evaluate_groups(grid.wavenumber_cm1, produce(None), n_rows)
