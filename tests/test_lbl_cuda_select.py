"""The fused selection kernels (``lbl_cuda_select``), compiled as host C++.

The CUDA source has no device-only constructs beyond qualifiers and the
thread indices, so with those defined away it compiles with ``g++`` and runs
on the CPU, one "thread" covering every element. ``-ffp-contract=off`` is
the host equivalent of ``--fmad=false``. Run through ``_select_packed``, it
must give the same lines and packed parameters as the NumPy array path:
this checks the kernels' indexing, table layout and operation order on
machines without a GPU. ``erfcx`` (not in the C++ library) is a stand-in
accurate to ~1e-13, so a line right at the peak-optical-depth threshold may
flip; none does in these cases.
"""
from __future__ import annotations

import ctypes
import dataclasses
import shutil
import subprocess

import numpy as np
import pytest

from mwir_atmos.api import build_profile, build_scenario_path
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics import lbl, lbl_cuda, lbl_cuda_dispatch, lbl_cuda_select
from mwir_atmos.spectral import make_grid
from mwir_atmos.types import ModelAtmosphere, Scenario, Species

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

_HOST_PRELUDE = r"""
#include <cmath>
using std::exp; using std::expm1; using std::pow; using std::sqrt; using std::fabs;
using std::fmin; using std::fmax; using std::erfc;
#define __global__
#define __device__
#define __forceinline__ inline
struct Dim { unsigned x; };
static const Dim blockIdx = {0}, threadIdx = {0}, blockDim = {1}, gridDim = {1};
static double erfcx(double x) {
    if (x < 0.0) return 2.0 * exp(x * x) - erfcx(-x);
    if (x < 26.0) return exp(x * x) * erfc(x);
    const double z = 1.0 / (2.0 * x * x);   // asymptotic series
    return (1.0 - z + 3.0 * z * z - 15.0 * z * z * z + 105.0 * z * z * z * z)
           / (x * 1.7724538509055160273);
}
"""


@pytest.fixture(scope="module")
def host_kernels(tmp_path_factory):
    folder = tmp_path_factory.mktemp("select")
    (folder / "select.cpp").write_text(_HOST_PRELUDE + lbl_cuda_select.source())
    library = folder / "select.so"
    subprocess.run(["g++", "-O2", "-std=c++14", "-ffp-contract=off", "-shared", "-fPIC",
                    str(folder / "select.cpp"), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    for name in ("select_keep", "select_values"):
        getattr(lib, name).restype = None
    return lib


def _host_select(lib):
    """``lbl_cuda_select.select`` with the host-compiled kernels and NumPy arrays."""
    def pointer(array):
        return ctypes.c_void_p(array.ctypes.data)

    def select(xp, table, n_tasks, n_iso, lines, total, grid_lo, grid_hi, use_strength, use_peak):
        arrays = [np.ascontiguousarray(lines[name]) for name in
                  ("nu", "delta_air", "s296", "gamma_air", "gamma_self", "e_lower", "n_air",
                   "m_kg_mol", "stim_ref")]
        iso = np.ascontiguousarray(lines["iso_index"], dtype=np.int64)
        table = np.ascontiguousarray(table)
        args = ([pointer(table), ctypes.c_int(n_tasks), ctypes.c_int(n_iso)]
                + [pointer(a) for a in arrays] + [pointer(iso),
                ctypes.c_double(grid_lo), ctypes.c_double(grid_hi),
                ctypes.c_int(int(use_strength)), ctypes.c_int(int(use_peak))])
        keep = np.zeros(total, dtype=np.uint8)
        lib.select_keep(ctypes.c_longlong(total), *args, pointer(keep))
        pairs = np.flatnonzero(keep).astype(np.int64)
        n = pairs.size
        out = (np.empty(n), np.empty(n, np.int64), np.empty(n, np.float32),
               np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, np.float32))
        lib.select_values(ctypes.c_longlong(n), pointer(pairs), *args, *[pointer(a) for a in out])
        return out
    return select


def _passes(grid, scenario, min_strength_fraction=0.0):
    """``(prepared, tasks)`` for every line list over every layer of ``scenario``."""
    path = build_scenario_path(scenario, build_profile(scenario))
    linelists = ensure_linelists()
    lo, hi = grid.wavenumber_cm1[0], grid.wavenumber_cm1[-1]
    for sp in (sp for sp in lbl.REQUIRED_SPECIES if sp in linelists):
        prepared = lbl._prepare(linelists[sp])
        tasks = []
        for layer in path.layers:
            column = layer.column_cm2.get(sp, 0.0)
            if column > 0.0:
                p = layer.p_eff_mbar.get(sp, layer.p_mbar)
                tasks.append(lbl_cuda_dispatch._selection_task(
                    prepared, lo, hi, p, layer.t_eff_k.get(sp, layer.t_k),
                    p * column / layer.air_column_cm2, 25.0, sp == Species.H2O,
                    min_strength_fraction, None, column, 3e-6))
        yield sp, prepared, tasks


@pytest.mark.parametrize("band, min_strength_fraction, atmosphere", [
    ((2140.0, 2145.0), 0.0, ModelAtmosphere.US_STANDARD),
    ((2140.0, 2145.0), 1e-4, ModelAtmosphere.MIDLAT_SUMMER),
    ((3300.0, 3600.0), 0.0, ModelAtmosphere.TROPICAL),
])
def test_fused_kernels_match_the_array_path(monkeypatch, host_kernels, band,
                                            min_strength_fraction, atmosphere):
    grid = make_grid(*band, 0.05)
    lo, hi = grid.wavenumber_cm1[0], grid.wavenumber_cm1[-1]
    kept = 0
    for sp, prepared, tasks in _passes(grid, Scenario(atmosphere=atmosphere),
                                       min_strength_fraction):
        monkeypatch.setattr(lbl_cuda_dispatch, "_use_fused", lambda xp: False)
        expected = lbl_cuda_dispatch._select_packed(np, prepared, tasks, lo, hi)
        monkeypatch.setattr(lbl_cuda_dispatch, "_use_fused", lambda xp: True)
        monkeypatch.setattr(lbl_cuda_select, "select", _host_select(host_kernels))
        got = lbl_cuda_dispatch._select_packed(np, prepared, tasks, lo, hi)
        for field in dataclasses.fields(lbl_cuda.Packed):
            np.testing.assert_array_equal(getattr(got, field.name), getattr(expected, field.name),
                                          err_msg=f"{sp}: {field.name}")
        kept += expected.center.size
    assert kept > 1000
