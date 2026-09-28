"""CUDA line-by-line backend (optional; needs CuPy and an NVIDIA GPU).

Line-by-line Voigt summation is ~98% of a resolved run's time, and every
(line, grid point) pair is independent, so it maps directly onto a GPU. The
CPU still selects lines (:func:`mwir_atmos.optics.lbl._select_lines`, which
applies the peak-optical-depth cutoff and pressure shift/broadening); this
module then evaluates every selected line of every (layer, species) task of a
spectral chunk in **one kernel launch** and returns optical depths.

Kernel layout: one CUDA block per (task, 256 consecutive grid points). The
block binary-searches the task's lines (sorted by shifted centre) for those
whose cutoff window reaches its points, stages them through shared memory in
tiles, and every thread accumulates its own point. Each thread sums lines in
the same order on every run, so results are deterministic.

Precision (see :mod:`mwir_atmos.optics.voigt_gpu`): offsets from line centre
and the per-point sums are float64; the profile itself is float32 in the far
wings and float64 in the cores. Evaluation is always *direct* (every pair
within the cutoff) -- the GPU needs none of the CPU's multigrid shortcuts.

:func:`evaluate_emulated` runs the identical algorithm in NumPy, so packing,
pedestal handling and precision are tested on machines without a GPU.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .voigt_gpu import CF_RADIUS, CF_TERMS, re_faddeeva, weideman_coefficients

__all__ = ["available", "device_summary", "evaluate", "evaluate_emulated", "pack"]

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


def pack(tasks: Sequence[tuple]) -> Packed:
    """Pack ``(lines, cutoff, column)`` tasks (``lines`` is an ``lbl._Lines``)."""
    parts = {k: [] for k in ("center", "inv_alpha", "y", "amp", "pedestal")}
    start, stop, cutoff, has_ped = [], [], [], []
    offset = 0
    for lines, cut, column in tasks:
        order = np.argsort(lines.center, kind="stable")
        alpha = lines.alpha_d[order]
        scale = column * lines.strength[order]
        parts["center"].append(lines.center[order])
        parts["inv_alpha"].append(_SQRT_LN2 / alpha)
        parts["y"].append(_SQRT_LN2 * lines.gamma_l[order] / alpha)
        parts["amp"].append(scale * _SQRT_LN2_OVER_PI / alpha)
        parts["pedestal"].append(scale * lines.pedestal[order] if lines.pedestal is not None
                                 else np.zeros(order.size))
        start.append(offset)
        offset += order.size
        stop.append(offset)
        cutoff.append(cut)
        has_ped.append(lines.pedestal is not None)
    cat = lambda name, dtype: (np.concatenate(parts[name]).astype(dtype) if parts[name]
                               else np.zeros(0, dtype))
    return Packed(center=cat("center", np.float64), inv_alpha=cat("inv_alpha", np.float32),
                  y=cat("y", np.float32), amp=cat("amp", np.float32),
                  pedestal=cat("pedestal", np.float32),
                  start=np.asarray(start, np.int64), stop=np.asarray(stop, np.int64),
                  cutoff=np.asarray(cutoff, np.float64),
                  has_pedestal=np.asarray(has_ped, np.int32))


# =========================================================================
# NumPy twin of the kernel
# =========================================================================


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
        point, line, dnu = point[keep], line[keep], dnu[keep]
        x = dnu.astype(np.float32) * packed.inv_alpha[line]
        value = packed.amp[line] * re_faddeeva(x, packed.y[line])
        if packed.has_pedestal[t]:
            value = np.maximum(value - packed.pedestal[line], np.float32(0.0))
        out[t] = np.bincount(point, weights=value.astype(np.float64), minlength=nu.size)
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
        const int task0, double* __restrict__ out) {
    const int task = task0 + blockIdx.y;
    const int i0 = blockIdx.x * TILE;
    const int i = i0 + threadIdx.x;
    const int i1 = min(i0 + TILE, n) - 1;
    const double cut = cutoff[task];
    const bool subtract = has_ped[task] != 0;

    __shared__ long long range[2];
    __shared__ double s_c[TILE];
    __shared__ float s_ia[TILE], s_y[TILE], s_amp[TILE], s_ped[TILE];
    if (threadIdx.x == 0) {
        range[0] = lower_bound(center, task_start[task], task_stop[task], nu[i0] - cut);
        range[1] = upper_bound(center, task_start[task], task_stop[task], nu[i1] + cut);
    }
    __syncthreads();
    const long long lo = range[0], hi = range[1];
    const double nu_i = (i < n) ? nu[i] : 0.0;
    double acc = 0.0;
    for (long long base = lo; base < hi; base += TILE) {
        const long long j = base + threadIdx.x;
        if (j < hi) {
            s_c[threadIdx.x] = center[j];
            s_ia[threadIdx.x] = inv_alpha[j];
            s_y[threadIdx.x] = yv[j];
            s_amp[threadIdx.x] = amp[j];
            s_ped[threadIdx.x] = ped[j];
        }
        __syncthreads();
        const int m = (int)min((long long)TILE, hi - base);
        if (i < n) {
            for (int k = 0; k < m; ++k) {
                const double dnu = nu_i - s_c[k];
                if (fabs(dnu) > cut) continue;
                float v = s_amp[k] * re_w((float)dnu * s_ia[k], s_y[k]);
                if (subtract) v = fmaxf(v - s_ped[k], 0.0f);
                acc += (double)v;
            }
        }
        __syncthreads();
    }
    if (i < n) out[(size_t)(task - task0) * n + i] = acc;
}
""" % dict(tile=_TILE, n=len(coefficients), coeffs=coeffs, big_l=big_l,
           cf_r2=float(CF_RADIUS**2), cf_terms=CF_TERMS)


_LOCK = threading.Lock()
_KERNELS: dict = {}


def _cupy():
    import cupy  # noqa: F401  (optional dependency)
    return cupy


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


def evaluate(nu: np.ndarray, packed: Packed) -> np.ndarray:
    """Optical depth of every packed task on ``nu``: shape ``(n_tasks, nu.size)``."""
    cp = _cupy()
    nu = np.ascontiguousarray(nu, dtype=np.float64)
    n_tasks, n = packed.start.size, nu.size
    result = np.zeros((n_tasks, n))
    if n == 0 or n_tasks == 0 or packed.center.size == 0:
        return result
    select_device()
    kernel = _kernel()
    d = {name: cp.asarray(getattr(packed, name)) for name in
         ("center", "inv_alpha", "y", "amp", "pedestal", "start", "stop", "cutoff", "has_pedestal")}
    d_nu = cp.asarray(nu)
    rows = max(1, min(n_tasks, _MAX_OUTPUT_BYTES // (8 * n), 65535))
    blocks_x = (n + _TILE - 1) // _TILE
    for task0 in range(0, n_tasks, rows):
        count = min(rows, n_tasks - task0)
        d_out = cp.empty((count, n), dtype=cp.float64)
        kernel((blocks_x, count), (_TILE,),
               (d_nu, np.int32(n), d["center"], d["inv_alpha"], d["y"], d["amp"], d["pedestal"],
                d["start"], d["stop"], d["cutoff"], d["has_pedestal"], np.int32(task0), d_out))
        result[task0:task0 + count] = cp.asnumpy(d_out)
    return result


def select_device() -> Optional[int]:
    """Use ``MWIR_CUDA_DEVICE`` if set (for pinning a process to one GPU)."""
    value = os.environ.get("MWIR_CUDA_DEVICE", "").strip()
    if not value:
        return None
    device = int(value)
    _cupy().cuda.Device(device).use()
    return device
