"""CUDA line-by-line backend (optional; needs CuPy and an NVIDIA GPU).

Line-by-line Voigt summation is ~98% of a resolved run's time, and every
(line, grid point) pair is independent, so it maps directly onto a GPU. The
CPU selects lines (:func:`mwir_atmos.optics.lbl._select_lines`, which
applies the peak-optical-depth cutoff and pressure shift/broadening), or with
``MWIR_CUDA_SELECTION=gpu`` the GPU does the same arithmetic in CuPy
(:func:`mwir_atmos.optics.lbl._select_packed`); this module evaluates the selected lines of a spectral chunk's (layer, species)
tasks in a few kernel launches (:func:`evaluate_groups`: the CPU selects the
next group of tasks while the GPU evaluates the current one), each task
writing its own row of one ``(species x layer, grid point)`` device array.
That array stays on the GPU when the grid does (resolved runs, see
``mwir_atmos.api._device_resident``); otherwise it is copied back once.

Kernel layout: one CUDA block per (task, 256 consecutive grid points). The
block binary-searches the task's lines (sorted by shifted centre) for those
whose cutoff window reaches its points, stages them through shared memory in
tiles, and every thread accumulates its own point. Each thread sums lines in
the same order on every run, so results are deterministic.

Precision (see :mod:`mwir_atmos.optics.voigt_gpu`): the profile is float32 in
the far wings and float64 in the cores. Nothing else per (line, point) pair
is float64, which runs at 1/64 rate on consumer and workstation GPUs: the
offset from line centre is formed from float32 hi/lo pairs relative to the
block's first point (:func:`_offset32`, ~1e-7 relative), the cutoff is a
per-line range of block points found once per tile with the float64 test
``|nu - c| <= cut``, and each tile is summed with compensated float32
arithmetic before adding to the float64 total. Evaluation is always *direct*
(every pair within the cutoff) -- the GPU needs none of the CPU's multigrid
shortcuts.

:func:`evaluate_emulated` runs the identical algorithm in NumPy, so packing,
pedestal handling and precision are tested on machines without a GPU.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .._array import is_device
from .voigt_gpu import CF_RADIUS, CF_TERMS, re_faddeeva, weideman_coefficients

__all__ = ["available", "concatenate", "device_summary", "evaluate", "evaluate_emulated",
           "evaluate_groups", "evaluate_groups_emulated", "launch", "pack", "pack_task"]

_SQRT_LN2 = float(np.sqrt(np.log(2.0)))
_SQRT_LN2_OVER_PI = float(np.sqrt(np.log(2.0) / np.pi))
#: Grid points per block, and lines per shared-memory tile.
_TILE = 256
#: Largest task-rows x grid-points output held on the GPU per launch (float64).
_MAX_OUTPUT_BYTES = 1 << 30


@dataclass(frozen=True)
class Packed:
    """Every task's lines, sorted by centre within each task, ready for upload."""

    center: np.ndarray      # float64, pressure-shifted centre, cm^-1
    inv_alpha: np.ndarray   # float32, sqrt(ln2) / Doppler HWHM
    y: np.ndarray           # float32, sqrt(ln2) * Lorentz / Doppler HWHM
    amp: np.ndarray         # float32, column * S(T) * sqrt(ln2/pi) / Doppler HWHM
    pedestal: np.ndarray    # float32, column * S(T) * MT_CKD pedestal (0 if none)
    start: np.ndarray       # int64 per task: first line
    stop: np.ndarray        # int64 per task: one past the last line
    cutoff: np.ndarray      # float64 per task, cm^-1
    has_pedestal: np.ndarray  # int32 per task


_LINE_ARRAYS = (("center", np.float64), ("inv_alpha", np.float32), ("y", np.float32),
                ("amp", np.float32), ("pedestal", np.float32))


def pack_task(lines, cutoff: float, column: float) -> tuple:
    """One task's kernel inputs: ``(line arrays in _LINE_ARRAYS order, cutoff, has_pedestal)``.

    Runs in the line-selection threads, so the serial :func:`pack` only concatenates.
    """
    order = np.argsort(lines.center, kind="stable")
    alpha = lines.alpha_d[order]
    scale = column * lines.strength[order]
    arrays = (lines.center[order],
              (_SQRT_LN2 / alpha).astype(np.float32),
              (_SQRT_LN2 * lines.gamma_l[order] / alpha).astype(np.float32),
              (scale * _SQRT_LN2_OVER_PI / alpha).astype(np.float32),
              (scale * lines.pedestal[order] if lines.pedestal is not None
               else np.zeros(order.size)).astype(np.float32))
    return arrays, float(cutoff), lines.pedestal is not None


def pack(tasks: Sequence[tuple]) -> Packed:
    """Pack ``(lines, cutoff, column)`` tasks (``lines`` is an ``lbl._Lines``)."""
    return concatenate([pack_task(*task) for task in tasks])


def concatenate(tasks: Sequence[tuple]) -> Packed:
    """Join :func:`pack_task` results into one :class:`Packed`, in order."""
    counts = np.array([arrays[0].size for arrays, _, _ in tasks], dtype=np.int64)
    stop = np.cumsum(counts)
    columns = {}
    for k, (name, dtype) in enumerate(_LINE_ARRAYS):
        columns[name] = (np.concatenate([arrays[k] for arrays, _, _ in tasks]).astype(dtype, copy=False)
                         if tasks else np.zeros(0, dtype))
    return Packed(**columns, start=stop - counts, stop=stop,
                  cutoff=np.array([cut for _, cut, _ in tasks], dtype=np.float64),
                  has_pedestal=np.array([ped for _, _, ped in tasks], dtype=np.int32))


# =========================================================================
# NumPy twin of the kernel
# =========================================================================


def _split32(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``value`` (float64) as float32 ``(hi, lo)`` with ``hi + lo`` ~ value to ~2**-48."""
    hi = value.astype(np.float32)
    return hi, (value - hi.astype(np.float64)).astype(np.float32)


def _offset32(nu: np.ndarray, point: np.ndarray, center: np.ndarray) -> np.ndarray:
    """``nu[point] - center`` in float32 exactly as the kernel forms it.

    ``nu0`` is the first point of the point's block of ``_TILE``; the point's
    offset from it (exact in float64) and ``nu0 - center`` are split into
    float32 pairs and added as ``(a_hi + b_hi) + (a_lo + b_lo)``: when the
    large parts nearly cancel their sum is exact, otherwise it dominates, so
    the offset is accurate to ~1e-7 relative however close to line centre.
    """
    nu0 = nu[(point // _TILE) * _TILE]
    a_hi, a_lo = _split32(nu[point] - nu0)
    b_hi, b_lo = _split32(nu0 - center)
    return (a_hi + b_hi) + (a_lo + b_lo)


def evaluate_emulated(nu: np.ndarray, packed: Packed) -> np.ndarray:
    """The kernel's algorithm and precision in NumPy; shape ``(n_tasks, nu.size)``."""
    nu = np.asarray(nu, dtype=np.float64)
    out = np.zeros((packed.start.size, nu.size))
    for t in range(packed.start.size):
        a, b = int(packed.start[t]), int(packed.stop[t])
        cut = float(packed.cutoff[t])
        centers = packed.center[a:b]
        lo = np.searchsorted(centers, nu - cut, side="left")
        hi = np.searchsorted(centers, nu + cut, side="right")
        counts = hi - lo
        point = np.repeat(np.arange(nu.size), counts)
        line = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
                + np.repeat(lo, counts)) + a
        dnu = nu[point] - packed.center[line]
        keep = np.abs(dnu) <= cut
        point, line = point[keep], line[keep]
        x = _offset32(nu, point, packed.center[line]) * packed.inv_alpha[line]
        value = packed.amp[line] * re_faddeeva(x, packed.y[line])
        if packed.has_pedestal[t]:
            value = np.maximum(value - packed.pedestal[line], np.float32(0.0))
        out[t] = np.bincount(point, weights=value.astype(np.float64), minlength=nu.size)
    return out


def evaluate_groups_emulated(nu: np.ndarray, groups: Iterable[tuple], n_rows: int) -> np.ndarray:
    """NumPy twin of :func:`evaluate_groups` (host arrays only)."""
    nu = np.asarray(nu, dtype=np.float64)
    out = np.zeros((n_rows, nu.size))
    for packed, rows in groups:
        out[np.asarray(rows, dtype=np.int64)] = evaluate_emulated(nu, packed)
    return out


# =========================================================================
# CUDA
# =========================================================================


def _source() -> str:
    big_l, coefficients = weideman_coefficients()
    coeffs = ", ".join(repr(float(c)) for c in coefficients)
    return r"""
#define TILE %(tile)d
__device__ __constant__ double WEID_A[%(n)d] = {%(coeffs)s};
#define WEID_N %(n)d
#define WEID_L %(big_l)r
#define INV_SQRT_PI 0.56418958354775628695
#define SQRT_PI_F 1.7724538509055160273f
#define CF_R2 %(cf_r2)rf
#define CF_TERMS %(cf_terms)d

__device__ __forceinline__ float re_w_cf(float x, float y) {
    float rr = x, ri = y;
#pragma unroll
    for (int k = CF_TERMS; k >= 1; --k) {
        const float h = (0.5f * (float)k) / (rr * rr + ri * ri);
        const float nr = x - h * rr;
        const float ni = y + h * ri;
        rr = nr;
        ri = ni;
    }
    return ri / (SQRT_PI_F * (rr * rr + ri * ri));
}

__device__ double re_w_weideman(double x, double y) {
    // den = L - i z, num = L + i z, z = x + i y
    const double dr = WEID_L + y, di = -x;
    const double nr = WEID_L - y, ni = x;
    const double dd = dr * dr + di * di;
    const double zr = (nr * dr + ni * di) / dd, zi = (ni * dr - nr * di) / dd;
    double pr = 0.0, pim = 0.0;
    for (int k = 0; k < WEID_N; ++k) {
        const double t = pr * zr - pim * zi + WEID_A[k];
        pim = pr * zi + pim * zr;
        pr = t;
    }
    const double ir = dr / dd, ii = -di / dd;             // 1/den
    const double i2r = ir * ir - ii * ii, i2i = 2.0 * ir * ii;  // 1/den^2
    return 2.0 * (pr * i2r - pim * i2i) + INV_SQRT_PI * ir;
}

__device__ __forceinline__ float re_w(float x, float y) {
    x = fabsf(x);
    if (x * x + y * y >= CF_R2) return re_w_cf(x, y);
    return (float)re_w_weideman((double)x, (double)y);
}

__device__ long long lower_bound(const double* a, long long lo, long long hi, double v) {
    while (lo < hi) { const long long m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
    return lo;
}

__device__ long long upper_bound(const double* a, long long lo, long long hi, double v) {
    while (lo < hi) { const long long m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
    return lo;
}

extern "C" __global__ void voigt_tau(
        const double* __restrict__ nu, const int n,
        const double* __restrict__ center, const float* __restrict__ inv_alpha,
        const float* __restrict__ yv, const float* __restrict__ amp,
        const float* __restrict__ ped,
        const long long* __restrict__ task_start, const long long* __restrict__ task_stop,
        const double* __restrict__ cutoff, const int* __restrict__ has_ped,
        const long long* __restrict__ out_row, const int task0, double* __restrict__ out) {
    // Float64 runs at 1/64 rate on consumer and workstation GPUs, so the
    // per-(line, point) work is float32 (see the module docstring):
    //  - offsets: nu_i - c = (nu_i - nu0) + (nu0 - c), both parts as float32
    //    hi/lo pairs (nu0 = the block's first point), summed as
    //    (a_hi + b_hi) + (a_lo + b_lo);
    //  - the cutoff |fl(nu_i - c)| <= cut: a per-line range of block points,
    //    found once per tile by binary search with that float64 test;
    //  - sums: compensated (Kahan) float32 per tile, then float64.
    const int task = task0 + blockIdx.y;
    const int i0 = blockIdx.x * TILE;
    const int t = threadIdx.x;
    const int i = i0 + t;
    const int m_pts = min(i0 + TILE, n) - i0;
    const double cut = cutoff[task];
    const bool subtract = has_ped[task] != 0;

    __shared__ long long range[2];
    __shared__ double s_nu[TILE];
    __shared__ float s_hi[TILE], s_lo[TILE], s_ia[TILE], s_y[TILE], s_amp[TILE], s_ped[TILE];
    __shared__ int s_first[TILE], s_last[TILE];
    if (t < m_pts) s_nu[t] = nu[i];
    if (t == 0) {
        range[0] = lower_bound(center, task_start[task], task_stop[task], nu[i0] - cut);
        range[1] = upper_bound(center, task_start[task], task_stop[task], nu[i0 + m_pts - 1] + cut);
    }
    __syncthreads();
    const long long lo = range[0], hi = range[1];
    const double nu0 = s_nu[0];
    float a_hi = 0.0f, a_lo = 0.0f;
    if (t < m_pts) {
        const double a = s_nu[t] - nu0;          // exact (Sterbenz)
        a_hi = (float)a;
        a_lo = (float)(a - (double)a_hi);
    }
    double acc = 0.0;
    for (long long base = lo; base < hi; base += TILE) {
        const long long j = base + t;
        if (j < hi) {
            const double c = center[j];
            const double off = nu0 - c;
            const float b_hi = (float)off;
            s_hi[t] = b_hi;
            s_lo[t] = (float)(off - (double)b_hi);
            int a = 0, b = m_pts;                // first point with fl(nu - c) >= -cut
            while (a < b) { const int mid = (a + b) >> 1; if (s_nu[mid] - c < -cut) a = mid + 1; else b = mid; }
            s_first[t] = a;
            b = m_pts;                           // one past the last with fl(nu - c) <= cut
            while (a < b) { const int mid = (a + b) >> 1; if (s_nu[mid] - c <= cut) a = mid + 1; else b = mid; }
            s_last[t] = a - 1;
            s_ia[t] = inv_alpha[j];
            s_y[t] = yv[j];
            s_amp[t] = amp[j];
            s_ped[t] = ped[j];
        }
        __syncthreads();
        const int m = (int)min((long long)TILE, hi - base);
        if (t < m_pts) {
            float sum = 0.0f, comp = 0.0f;
            for (int k = 0; k < m; ++k) {
                if (t < s_first[k] || t > s_last[k]) continue;
                const float dnu = (a_hi + s_hi[k]) + (a_lo + s_lo[k]);
                float v = s_amp[k] * re_w(dnu * s_ia[k], s_y[k]);
                if (subtract) v = fmaxf(v - s_ped[k], 0.0f);
                const float term = v - comp;
                const float next = sum + term;
                comp = (next - sum) - term;
                sum = next;
            }
            acc += (double)sum - (double)comp;
        }
        __syncthreads();
    }
    if (t < m_pts) out[(size_t)out_row[task] * n + i] = acc;
}
""" % dict(tile=_TILE, n=len(coefficients), coeffs=coeffs, big_l=big_l,
           cf_r2=float(CF_RADIUS**2), cf_terms=CF_TERMS)


_LOCK = threading.Lock()
_KERNELS: dict = {}


def _cupy():
    import cupy  # noqa: F401  (optional dependency)
    return cupy


def selection_module():
    """The array module GPU line selection runs in (CuPy; tests substitute NumPy)."""
    return _cupy()


def available() -> bool:
    """True when CuPy imports and sees at least one CUDA device."""
    try:
        return _cupy().cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def device_summary() -> str:
    cp = _cupy()
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    return (f"{name} (device {dev.id}, compute {props['major']}.{props['minor']}, "
            f"{props['totalGlobalMem'] / 2**30:.0f} GiB)")


def _kernel():
    cp = _cupy()
    device = cp.cuda.Device().id
    with _LOCK:
        kernel = _KERNELS.get(device)
        if kernel is None:
            kernel = cp.RawKernel(_source(), "voigt_tau", options=("--std=c++14",))
            kernel.compile()
            _KERNELS[device] = kernel
    return kernel


def launch(out, nu, packed: Packed, rows) -> None:
    """Enqueue the kernel for every task of ``packed``; task ``t`` fills row ``rows[t]`` of ``out``.

    ``out`` (float64, shape ``(R, nu.size)``) and ``nu`` are CuPy arrays. The
    call returns once the work is queued, without waiting for the GPU.
    """
    cp = _cupy()
    n_tasks, n = packed.start.size, int(nu.size)
    if n == 0 or n_tasks == 0 or packed.center.size == 0:
        return
    kernel = _kernel()
    d = {name: cp.asarray(getattr(packed, name)) for name in
         ("center", "inv_alpha", "y", "amp", "pedestal", "start", "stop", "cutoff", "has_pedestal")}
    d_rows = cp.asarray(np.asarray(rows, dtype=np.int64))
    blocks_x = (n + _TILE - 1) // _TILE
    for task0 in range(0, n_tasks, 65535):
        count = min(65535, n_tasks - task0)
        kernel((blocks_x, count), (_TILE,),
               (nu, np.int32(n), d["center"], d["inv_alpha"], d["y"], d["amp"], d["pedestal"],
                d["start"], d["stop"], d["cutoff"], d["has_pedestal"], d_rows, np.int32(task0),
                out))
    # Submit now: Windows (WDDM) drivers may otherwise hold queued launches back
    # until the next synchronising call, and the CPU prepares the next group meanwhile.
    cp.cuda.get_current_stream().done


def evaluate_groups(nu, groups: Iterable[tuple], n_rows: int):
    """Optical depths of packed task groups in one ``(n_rows, nu.size)`` array.

    ``groups`` yields ``(packed, rows)``: task ``t`` of ``packed`` goes to row
    ``rows[t]``; rows no task names stay zero. Each group's kernel is queued
    before the next group is requested, so a lazy ``groups`` (the CPU line
    selection in :func:`mwir_atmos.optics.lbl.molecular_optical_depth`)
    prepares group ``k+1`` while the GPU evaluates group ``k``. A CuPy ``nu``
    gives a CuPy result that stays on the device; a NumPy ``nu`` a NumPy copy.
    """
    cp = _cupy()
    select_device()
    on_device = is_device(nu)
    d_nu = nu if on_device else cp.asarray(np.ascontiguousarray(nu, dtype=np.float64))
    out = cp.zeros((n_rows, int(d_nu.size)), dtype=cp.float64)
    for packed, rows in groups:
        launch(out, d_nu, packed, rows)
    return out if on_device else out.get()


def evaluate(nu: np.ndarray, packed: Packed) -> np.ndarray:
    """Optical depth of every packed task on ``nu``: shape ``(n_tasks, nu.size)``."""
    nu = np.ascontiguousarray(nu, dtype=np.float64)
    n_tasks, n = packed.start.size, nu.size
    result = np.zeros((n_tasks, n))
    if n == 0 or n_tasks == 0 or packed.center.size == 0:
        return result
    # Bound the device output: at most _MAX_OUTPUT_BYTES of rows at a time.
    rows = max(1, min(n_tasks, _MAX_OUTPUT_BYTES // (8 * n)))
    for task0 in range(0, n_tasks, rows):
        part = _task_slice(packed, task0, min(task0 + rows, n_tasks))
        result[task0:task0 + part.start.size] = evaluate_groups(
            nu, [(part, np.arange(part.start.size))], part.start.size)
    return result


def _task_slice(packed: Packed, a: int, b: int) -> Packed:
    """Tasks ``a..b-1`` of ``packed`` as their own :class:`Packed`."""
    if a == 0 and b == packed.start.size:
        return packed
    lo, hi = int(packed.start[a]), int(packed.stop[b - 1])
    return Packed(**{name: getattr(packed, name)[lo:hi] for name, _ in _LINE_ARRAYS},
                  start=packed.start[a:b] - lo, stop=packed.stop[a:b] - lo,
                  cutoff=packed.cutoff[a:b], has_pedestal=packed.has_pedestal[a:b])


def to_device(array):
    """``array`` as a CuPy array on the selected GPU (``MWIR_CUDA_DEVICE``)."""
    select_device()
    return _cupy().asarray(array)


def release_memory() -> None:
    """Return CuPy's cached device and pinned memory to the driver (the GPU may be shared)."""
    try:
        cp = _cupy()
    except ImportError:
        return
    from . import lbl
    with lbl._DEVICE_LINES_LOCK:
        lbl._DEVICE_LINES.clear()     # line lists uploaded for GPU selection
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def select_device() -> Optional[int]:
    """Use ``MWIR_CUDA_DEVICE`` if set (for pinning a process to one GPU)."""
    value = os.environ.get("MWIR_CUDA_DEVICE", "").strip()
    if not value:
        return None
    device = int(value)
    _cupy().cuda.Device(device).use()
    return device
