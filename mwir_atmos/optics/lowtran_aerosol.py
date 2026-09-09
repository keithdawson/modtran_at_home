"""LOWTRAN 7 table aerosols; see docs/lowtran_aerosols.md for scope and sources.

EXABIN humidity interpolation precedes AEREXT wavelength interpolation.
AERPRF supplies 550-nm profiles. Layer integration uses log-linear extinction
between those altitude nodes, with explicit splits at optical-region boundaries.
"""
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..data_paths import aerosol_data_dir
from ..types import AerosolModel, PathLayer, SpectralGrid

TABLE_MODELS = {
    AerosolModel.LOWTRAN_RURAL: "RUR",
    AerosolModel.LOWTRAN_URBAN: "URB",
    AerosolModel.LOWTRAN_MARITIME: "OCN",
    AerosolModel.LOWTRAN_TROPOSPHERIC: "TRO",
}
RH_NODES = np.array([0., 70., 80., 99.])


@lru_cache(maxsize=1)
def _tables() -> dict[str, np.ndarray]:
    with np.load(aerosol_data_dir() / "lowtran7.npz", allow_pickle=False) as data:
        tables = {name: data[name] for name in data.files}
    for array in tables.values():
        array.flags.writeable = False
    return tables


def optical_properties(prefix: str, rh_pct: float, wavelength_um: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return normalized extinction, absorption and g (0.2–40 µm).

    RH is bounded to the original table's 0–99% range. At fixed visibility,
    RH changes spectral shape and albedo, not the 550-nm reference loading.
    """
    lam = np.asarray(wavelength_um, dtype=float)
    if not np.isfinite(lam).all() or np.any((lam < .2) | (lam > 40.)):
        raise ValueError("LOWTRAN aerosol tables support wavelengths from 0.2 to 40 micrometers")
    if not np.isfinite(rh_pct):
        raise ValueError("rh_pct must be finite")
    data = _tables()
    native = data["VX2"][:40]
    if prefix in ("BST", "DME"):
        values = [data[prefix + field][:40] for field in ("EXT", "ABS", "SYM")]
    elif prefix in TABLE_MODELS.values():
        rh = np.clip(rh_pct, 0., 99.)
        j = int(np.clip(np.searchsorted(RH_NODES, rh, side="right"), 1, 3))
        x = np.log(100. - RH_NODES)
        weight = (np.log(100. - rh) - x[j - 1]) / (x[j] - x[j - 1])
        values = []
        for field in "EAG":
            lo, hi = (data[f"{prefix}{field}{k}"][:40] for k in (j, j + 1))
            values.append(np.exp(np.log(lo) * (1. - weight) + np.log(hi) * weight))
    else:
        raise ValueError(f"Unknown LOWTRAN optical table: {prefix}")
    return tuple(np.interp(lam, native, value) for value in values)


def extinction_profile(visibility_km: float, *, winter: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """AERPRF background profile at native nodes, altitude in km MSL.

    Return altitude and aerosol extinction (km^-1 at 550 nm). No molecular
    extinction is added here. Only the tabulated 2–50 km visibility range is
    supported; the original program's unbounded extrapolation is excluded.
    """
    if not np.isfinite(visibility_km) or not 2. <= visibility_km <= 50.:
        raise ValueError("LOWTRAN table aerosols require visibility_km between 2 and 50")
    data = _tables()
    z = data["ZHT"][:-1]  # omit the original 99999-km sentinel
    result = np.zeros_like(z)
    vs = np.array([50., 23., 10., 5., 2.])
    j = int(np.clip(np.searchsorted(-vs, -visibility_km), 1, 4))
    fraction = (1. / visibility_km - 1. / vs[j - 1]) / (1. / vs[j] - 1. / vs[j - 1])
    result[:3] = data["HZ2K"][:, j - 1] * (1. - fraction) + data["HZ2K"][:, j] * fraction
    season = "FAWI" if winter else "SPSU"
    weight = np.clip((1. / visibility_km - 1. / 50.) / (1. / 23. - 1. / 50.), 0., 1.)
    result[3:11] = data[season + "50"][3:11] * (1. - weight) + data[season + "23"][3:11] * weight
    result[11:27] = data["BASTFW" if winter else "BASTSS"][11:27]
    result[27:] = data["UPNATM"][27:-1]
    return z, result


def table_optical_depth(
    layers: Sequence[PathLayer], grid: SpectralGrid, model: AerosolModel,
    *, visibility_km: float = 23., winter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate background aerosol extinction, keeping scattering-weighted g.

    Geometry within a PathLayer is linear in altitude; each segment is split
    at profile nodes and at 2, 10, 30 and 100 km. Above 100 km aerosol is zero.
    Humidity is the layer's mean RH; finer geometry layers resolve RH gradients.
    """
    z, sigma = extinction_profile(visibility_km, winter=winter)
    log_sigma = np.log(sigma)
    # Validate spectral bounds even for an empty path.
    optical_properties(TABLE_MODELS[model], 70., grid.wavelength_um)
    shape = (len(layers), grid.n)
    absorption, scattering, g_numerator = (np.zeros(shape) for _ in range(3))
    for i, layer in enumerate(layers):
        lo, hi = sorted((layer.z_bottom_km, layer.z_top_km))
        if lo < 0. or not np.isfinite([lo, hi, layer.ds_km, layer.rh_pct]).all() or layer.ds_km < 0.:
            raise ValueError("LOWTRAN aerosol layers require finite nonnegative altitude and path length")
        cuts = np.r_[lo, z[(z > lo) & (z < hi)], hi]
        tau550 = 0.
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = (a + b) / 2.
            if mid > 100.:
                continue
            ds = layer.ds_km if hi == lo else layer.ds_km * (b - a) / (hi - lo)
            log_a, log_b = np.interp([a, b], z, log_sigma)
            delta = log_b - log_a
            average = np.exp(log_a) * (np.expm1(delta) / delta if abs(delta) > 1e-10 else 1.)
            reference_od = ds * average
            prefix = TABLE_MODELS[model] if mid <= 2. else "TRO" if mid <= 10. else "BST" if mid <= 30. else "DME"
            rh = 70. if 2. < mid <= 10. else layer.rh_pct
            ext, ab, g = optical_properties(prefix, rh, grid.wavelength_um)
            sca = np.maximum(ext - ab, 0.)  # source rounding can make tiny differences
            absorption[i] += reference_od * ab
            scattering[i] += reference_od * sca
            g_numerator[i] += reference_od * sca * g
            tau550 += reference_od
        layer.aerosol_od_550nm = float(tau550)
    g = np.divide(g_numerator, scattering, out=np.zeros(shape), where=scattering > 0.)
    return absorption, scattering, g
