"""Fused CUDA kernels for GPU line selection (see ``lbl_cuda_dispatch._select_packed``).

The array-operation version of the selection runs ~40 CuPy kernels per pass,
each reading and writing float64 temporaries for every (task, window line)
pair (~2 ns per pair on an RTX 4090). Here two kernels do the per-pair work:

``select_keep``
    one thread per pair: S(T), shifted centre, widths and the keep tests;
    writes one byte per pair.
``select_values``
    one thread per kept pair: the shifted centre, the task, and the packed
    float32 parameters.

Compaction (``flatnonzero``) and the two stable sorts by (task, centre) stay
CuPy calls. Each per-pair quantity is computed with the same operations in
the same order as the NumPy expressions in ``_select_packed``, compiled with
``--fmad=false`` (no fused multiply-add contraction) and IEEE division and
square root, so only ``exp``, ``expm1``, ``pow`` and ``erfcx`` round
differently from NumPy.

:func:`source` is plain CUDA C++ that also compiles as host C++ (see
``tests/test_lbl_cuda_select.py``, which runs it on the CPU against NumPy).
"""
from __future__ import annotations

import threading

import numpy as np

__all__ = ["CONSTANTS", "TASK_COLUMNS", "select", "source"]

#: Per-task columns of the table ``_select_packed`` uploads, in order.
TASK_COLUMNS = ("first_minus_offset", "end", "temperature", "inverse_delta", "p_total",
                "cutoff", "strength_threshold", "strength_applies", "doppler", "p_air",
                "p_self", "ratio", "column", "peak_minimum", "peak_applies", "pedestal")
#: Constants appended to the table (exact Python floats, so no literal rounding).
CONSTANTS = ("c2", "light_c_squared", "sqrt_ln2", "sqrt_ln2_over_pi", "pi", "mt_ckd_cutoff_squared")

#: Threads per block, and blocks per launch (grid-stride loops).
_BLOCK = 256
_MAX_BLOCKS = 65535


def source() -> str:
    columns = "\n".join(f"#define COL_{name.upper()} {k}" for k, name in enumerate(TASK_COLUMNS))
    constants = "\n".join(f"#define K_{name.upper()} {k}" for k, name in enumerate(CONSTANTS))
    return columns + "\n" + constants + "\n" + r"""
#define N_COLUMNS %(n_columns)d

struct Pair {
    long long task, line;
    double center, s_t, alpha, gamma, column;
    bool keep;
};

__device__ __forceinline__ double task_value(const double* table, int n_tasks, int column,
                                             long long task) {
    return table[(long long)column * n_tasks + task];
}

// The pair's task: the first whose cumulative end exceeds p (searchsorted, side="right").
__device__ __forceinline__ long long task_of(const double* table, int n_tasks, long long p) {
    int lo = 0, hi = n_tasks;
    const double v = (double)p;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (task_value(table, n_tasks, COL_END, mid) <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// _select_packed's per-pair arithmetic, operation for operation.
__device__ Pair evaluate_pair(
        long long p, const double* __restrict__ table, int n_tasks, int n_iso,
        const double* __restrict__ nu, const double* __restrict__ delta_air,
        const double* __restrict__ s296, const double* __restrict__ gamma_air,
        const double* __restrict__ gamma_self, const double* __restrict__ e_lower,
        const double* __restrict__ n_air, const double* __restrict__ m_kg_mol,
        const double* __restrict__ stim_ref, const long long* __restrict__ iso_index,
        double grid_lo, double grid_hi, int use_strength, int use_peak) {
    const double* k = table + (long long)N_COLUMNS * n_tasks + (long long)n_tasks * n_iso;
    Pair r;
    const long long t = task_of(table, n_tasks, p);
    const long long j = p + (long long)task_value(table, n_tasks, COL_FIRST_MINUS_OFFSET, t);
    r.task = t;
    r.line = j;
    const double temperature = task_value(table, n_tasks, COL_TEMPERATURE, t);
    const double nu_j = nu[j];
    const double q_ratio = table[(long long)N_COLUMNS * n_tasks + t * n_iso + iso_index[j]];
    const double inverse_delta = task_value(table, n_tasks, COL_INVERSE_DELTA, t);
    const double boltzmann = exp(-k[K_C2] * e_lower[j] * inverse_delta);
    const double stim = expm1(-k[K_C2] * nu_j / temperature) / stim_ref[j];
    r.s_t = s296[j] * q_ratio * boltzmann * stim;

    r.center = nu_j + delta_air[j] * task_value(table, n_tasks, COL_P_TOTAL, t);
    const double clipped = fmin(fmax(r.center, grid_lo), grid_hi);
    bool keep = fabs(r.center - clipped) <= task_value(table, n_tasks, COL_CUTOFF, t);
    if (use_strength && task_value(table, n_tasks, COL_STRENGTH_APPLIES, t) != 0.0)
        keep = keep && r.s_t >= task_value(table, n_tasks, COL_STRENGTH_THRESHOLD, t);

    const double doppler = task_value(table, n_tasks, COL_DOPPLER, t);
    r.alpha = nu_j * sqrt(doppler / (m_kg_mol[j] * k[K_LIGHT_C_SQUARED]));
    r.gamma = (gamma_air[j] * task_value(table, n_tasks, COL_P_AIR, t)
               + gamma_self[j] * task_value(table, n_tasks, COL_P_SELF, t))
              * pow(task_value(table, n_tasks, COL_RATIO, t), n_air[j]);
    r.column = task_value(table, n_tasks, COL_COLUMN, t);
    if (use_peak && task_value(table, n_tasks, COL_PEAK_APPLIES, t) != 0.0) {
        const double peak = k[K_SQRT_LN2_OVER_PI] / r.alpha * erfcx(k[K_SQRT_LN2] * r.gamma / r.alpha);
        keep = keep && r.s_t * r.column * peak >= task_value(table, n_tasks, COL_PEAK_MINIMUM, t);
    }
    r.keep = keep;
    return r;
}

#define LINE_ARGS \
        const double* __restrict__ table, const int n_tasks, const int n_iso, \
        const double* __restrict__ nu, const double* __restrict__ delta_air, \
        const double* __restrict__ s296, const double* __restrict__ gamma_air, \
        const double* __restrict__ gamma_self, const double* __restrict__ e_lower, \
        const double* __restrict__ n_air, const double* __restrict__ m_kg_mol, \
        const double* __restrict__ stim_ref, const long long* __restrict__ iso_index, \
        const double grid_lo, const double grid_hi, const int use_strength, const int use_peak
#define PAIR(p) evaluate_pair(p, table, n_tasks, n_iso, nu, delta_air, s296, gamma_air, \
        gamma_self, e_lower, n_air, m_kg_mol, stim_ref, iso_index, grid_lo, grid_hi, \
        use_strength, use_peak)

extern "C" __global__ void select_keep(const long long total, LINE_ARGS,
                                       unsigned char* __restrict__ keep) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < total; p += stride)
        keep[p] = PAIR(p).keep ? 1 : 0;
}

extern "C" __global__ void select_values(const long long n, const long long* __restrict__ pairs,
                                         LINE_ARGS,
                                         double* __restrict__ center, long long* __restrict__ owner,
                                         float* __restrict__ inv_alpha, float* __restrict__ y,
                                         float* __restrict__ amp, float* __restrict__ pedestal) {
    const double* k = table + (long long)N_COLUMNS * n_tasks + (long long)n_tasks * n_iso;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const Pair r = PAIR(pairs[i]);
        center[i] = r.center;
        owner[i] = r.task;
        // pack_task: scale = column * S(T); pedestal = gamma / (pi (25^2 + gamma^2)).
        const double scale = r.column * r.s_t;
        inv_alpha[i] = (float)(k[K_SQRT_LN2] / r.alpha);
        y[i] = (float)(k[K_SQRT_LN2] * r.gamma / r.alpha);
        amp[i] = (float)(scale * k[K_SQRT_LN2_OVER_PI] / r.alpha);
        const double ped = r.gamma / (k[K_PI] * (k[K_MT_CKD_CUTOFF_SQUARED] + r.gamma * r.gamma));
        pedestal[i] = task_value(table, n_tasks, COL_PEDESTAL, r.task) != 0.0
                      ? (float)(scale * ped) : 0.0f;
    }
}
""" % dict(n_columns=len(TASK_COLUMNS))


#: --fmad=false keeps every multiply and add separately rounded, as in NumPy.
COMPILE_OPTIONS = ("--std=c++14", "--fmad=false", "--prec-div=true", "--prec-sqrt=true")
_LOCK = threading.Lock()
_MODULES: dict = {}


def _kernels(cp):
    device = cp.cuda.Device().id
    with _LOCK:
        module = _MODULES.get(device)
        if module is None:
            module = cp.RawModule(code=source(), options=COMPILE_OPTIONS)
            module.compile()
            _MODULES[device] = module
    return module.get_function("select_keep"), module.get_function("select_values")


def _blocks(n: int) -> int:
    return max(1, min(_MAX_BLOCKS, (n + _BLOCK - 1) // _BLOCK))


def select(cp, table, n_tasks: int, n_iso: int, lines: dict, total: int, grid_lo: float,
           grid_hi: float, use_strength: bool, use_peak: bool):
    """Kept pairs of one pass: ``(center, owner, inv_alpha, y, amp, pedestal)`` in pair order.

    ``table`` is the device table (task columns, Q ratios, :data:`CONSTANTS`);
    ``lines`` the device line arrays of ``_device_lines``.
    """
    keep_kernel, values_kernel = _kernels(cp)
    args = (table, np.int32(n_tasks), np.int32(n_iso), lines["nu"], lines["delta_air"],
            lines["s296"], lines["gamma_air"], lines["gamma_self"], lines["e_lower"],
            lines["n_air"], lines["m_kg_mol"], lines["stim_ref"], lines["iso_index"],
            np.float64(grid_lo), np.float64(grid_hi), np.int32(use_strength), np.int32(use_peak))
    keep = cp.empty(total, dtype=cp.uint8)
    if total:
        keep_kernel((_blocks(total),), (_BLOCK,), (np.int64(total),) + args + (keep,))
    pairs = cp.flatnonzero(keep)
    n = int(pairs.size)
    out = (cp.empty(n, cp.float64), cp.empty(n, cp.int64), cp.empty(n, cp.float32),
           cp.empty(n, cp.float32), cp.empty(n, cp.float32), cp.empty(n, cp.float32))
    if n:
        values_kernel((_blocks(n),), (_BLOCK,), (np.int64(n), pairs) + args + out)
    return out
