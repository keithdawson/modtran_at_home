"""Physical benchmark and validation test suite for mwir_atmos.

Validates the radiative transfer, spectroscopy, aerosol, geometry, and turbulence
physics against canonical physical laws, standard reference atmospheres (AFGL /
US Standard 1976), and published MODTRAN 6 benchmark cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.api import run_scenario
from mwir_atmos.constants import BACKGROUND_VMR_PPMV
from mwir_atmos.geometry.path import build_path
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics.lbl import molecular_optical_depth
from mwir_atmos.optics.transmittance import compute_layer_optics, path_transmittance
from mwir_atmos.radiometry.radiance import compute_radiance
from mwir_atmos.spectral import band_average, make_grid, planck
from mwir_atmos.turbulence.metrics import (
    fried_parameter_r0_m,
    isoplanatic_angle_rad,
    scintillation_index,
)
from mwir_atmos.types import (
    AerosolModel,
    AtmosphereProfile,
    ModelAtmosphere,
    Scenario,
    SkyPosition,
    Species,
    SurfaceType,
)


@pytest.fixture(scope="module")
def linelists():
    return ensure_linelists()


# =========================================================================
# 1. Kirchhoff's Law & Isothermal Blackbody Cavity Radiative Invariance
# =========================================================================


def test_isothermal_cavity_radiates_exact_planck_blackbody():
    """In an enclosed isothermal absorbing atmosphere at temperature T0 and blackbody
    surface at T0 without external sources (night), the radiance along ANY
    path must identically equal the Planck blackbody radiance B(nu, T0).

    By Kirchhoff's law:
        L_total(nu) = B(nu, T0) * tau_total + sum_i B(nu, T0) * (1 - exp(-tau_i)) * tau_accum
                    = B(nu, T0) [ tau_total + (1 - tau_total) ] = B(nu, T0)
    """
    t_iso = 300.0
    alt_km = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=np.float64)
    # Hydrostatic-like pressure exponential
    p_mbar = 1013.25 * np.exp(-alt_km / 7.0)
    t_k = np.full_like(alt_km, t_iso)

    iso_profile = AtmosphereProfile(
        model=ModelAtmosphere.US_STANDARD,
        altitude_km=alt_km,
        pressure_mbar=p_mbar,
        temperature_k=t_k,
        vmr_ppmv={
            Species.H2O: np.full_like(alt_km, 10000.0),
            Species.CO2: np.full_like(alt_km, BACKGROUND_VMR_PPMV[Species.CO2]),
            Species.O3: np.full_like(alt_km, 0.05),
            Species.N2O: np.full_like(alt_km, BACKGROUND_VMR_PPMV[Species.N2O]),
            Species.CO: np.full_like(alt_km, BACKGROUND_VMR_PPMV[Species.CO]),
            Species.CH4: np.full_like(alt_km, BACKGROUND_VMR_PPMV[Species.CH4]),
            Species.O2: np.full_like(alt_km, BACKGROUND_VMR_PPMV[Species.O2]),
        },
    )

    grid = make_grid(wavenumber_min_cm1=2200.0, wavenumber_max_cm1=2800.0, resolution_cm1=10.0)

    for zenith in (120.0, 150.0, 180.0):  # Down-looking paths to surface
        path = build_path(
            iso_profile,
            sensor_alt_km=10.0,
            zenith_deg=zenith,
            max_layers=20,
        )
        # Verify ground intersection
        assert path.hits_ground

        # Pure absorbing medium (no scattering loss)
        optics = compute_layer_optics(
            path, grid, aerosol=AerosolModel.NONE, include_rayleigh=False
        )
        trans = path_transmittance(optics)

        scenario = Scenario(
            atmosphere=ModelAtmosphere.US_STANDARD,
            surface_type=SurfaceType.BLACKBODY,
            ground_temp_k=t_iso,
            sensor_alt_km=10.0,
            sensor_zenith_deg=zenith,
            wavenumber_min_cm1=2200.0,
            wavenumber_max_cm1=2800.0,
            resolution_cm1=10.0,
            aerosol=AerosolModel.NONE,
            local_datetime=None,
        )

        no_body = SkyPosition(
            zenith_deg=180.0,
            azimuth_deg=0.0,
            scattering_angle_deg=180.0,
            illuminated_fraction=0.0,
            is_above_horizon=False,
        )

        radiance = compute_radiance(
            scenario=scenario,
            grid=grid,
            profile=iso_profile,
            path=path,
            optics=optics,
            transmittance=trans,
            sun=no_body,
            moon=no_body,
        )

        expected_bb = planck(grid.wavenumber_cm1, t_iso)
        # Path thermal + surface emitted must equal exact blackbody
        np.testing.assert_allclose(radiance.total, expected_bb, rtol=1e-5)


# =========================================================================
# 2. MODTRAN 6 / US Standard 1976 Band Transmission Benchmarks
# =========================================================================


def test_us_standard_10km_horizontal_transmission_benchmarks():
    """Verify transmission across classical MWIR bands for a 10 km sea-level
    horizontal path in US Standard Atmosphere (23 km rural visibility).

    Matches standard MODTRAN 6 / HITRAN benchmark ranges:
      - 4.2-4.4 um CO2 band: fully opaque (tau < 1e-6)
      - 3.6-4.0 um clean MWIR window: high transmission (0.50 - 0.85)
      - 4.6-4.8 um window: moderate transmission (0.35 - 0.70)
    """
    sc = Scenario(
        atmosphere=ModelAtmosphere.US_STANDARD,
        sensor_alt_km=0.01,
        sensor_zenith_deg=90.0,
        source_range_km=10.0,
        wavenumber_min_cm1=2000.0,
        wavenumber_max_cm1=3000.0,
        resolution_cm1=4.0,
        aerosol=AerosolModel.RURAL,
        visibility_km=23.0,
    )
    res = run_scenario(sc)

    # 1. 4.25 - 4.35 um CO2 asymmetric stretch fundamental band
    tau_co2 = band_average(res.grid, res.transmittance.total, (2300.0, 2350.0))
    assert tau_co2 < 1e-6, f"CO2 band must be opaque, got {tau_co2}"

    # 2. 3.6 - 4.0 um clean atmospheric window (2500 - 2777 cm^-1)
    tau_win_3um = band_average(res.grid, res.transmittance.total, (2550.0, 2750.0))
    assert 0.50 <= tau_win_3um <= 0.85, f"3.7 um window expected ~0.50-0.85, got {tau_win_3um}"

    # 3. 4.6 - 4.8 um window (2083 - 2174 cm^-1)
    tau_win_4um = band_average(res.grid, res.transmittance.total, (2100.0, 2170.0))
    assert 0.35 <= tau_win_4um <= 0.75, f"4.7 um window expected ~0.35-0.75, got {tau_win_4um}"


# =========================================================================
# 3. Doppler vs. Lorentz Line-Shape Transition
# =========================================================================


def test_doppler_gaussian_vs_lorentz_power_law_broadening(linelists):
    """At sea-level (1013 mbar), collisional Lorentz broadening gives a broad profile
    with slow wing decay. In the stratosphere (p < 5 mbar, e.g. 35 km), thermal Doppler broadening
    dominates with an exp(-(delta nu)^2 / gamma_D^2) Gaussian profile that collapses extremely rapidly
    away from line centre.
    """
    # Single CO line at 2172.76 cm^-1
    nu0 = 2172.76
    # High resolution grid around the line core
    grid = make_grid(
        wavenumber_min_cm1=nu0 - 0.05,
        wavenumber_max_cm1=nu0 + 0.05,
        resolution_cm1=0.0005,
    )

    from tests.test_physics_regression import _homogeneous_layer

    # Sea-level layer (Lorentz dominated: gamma_L ~ 0.05 cm^-1)
    sea_layer = _homogeneous_layer(ds_km=1.0, p_mbar=1013.25, t_k=296.0, h2o_vmr=0.01)
    # Stratospheric layer at 35 km (Doppler dominated: gamma_D ~ 0.002 cm^-1, p ~ 5.7 mbar)
    strat_layer = _homogeneous_layer(ds_km=10.0, p_mbar=5.7, t_k=236.0, h2o_vmr=5e-6)

    tau_sea_tot, _ = molecular_optical_depth([sea_layer], grid, species=[Species.CO], linelists=linelists)
    tau_strat_tot, _ = molecular_optical_depth([strat_layer], grid, species=[Species.CO], linelists=linelists)

    tau_sea = tau_sea_tot[0]
    tau_strat = tau_strat_tot[0]

    # Find centre and detuned positions
    idx_center = int(np.argmin(np.abs(grid.wavenumber_cm1 - nu0)))
    idx_detuned = int(np.argmin(np.abs(grid.wavenumber_cm1 - (nu0 + 0.015))))

    # Normalized line shapes at delta nu = 0.015 cm^-1
    sea_retention = tau_sea[idx_detuned] / tau_sea[idx_center]
    strat_retention = tau_strat[idx_detuned] / tau_strat[idx_center]

    # At 0.015 cm^-1 (~7 Doppler widths), Doppler line core has fallen by > 10^5,
    # whereas Lorentz line core at 0.015 cm^-1 is still > 90% of its peak.
    assert sea_retention > 100.0 * strat_retention, (
        f"Sea-level Lorentz retention ({sea_retention:.3e}) must vastly exceed stratospheric Doppler retention ({strat_retention:.3e})"
    )


# =========================================================================
# 4. Extreme Earth Climatological Soundings
# =========================================================================


def test_climatological_soundings_polar_vs_tropical():
    """Compare cold-dry subarctic winter vs warm-humid tropical atmospheres for 15 km horizontal paths.
    Warm-humid tropical atmosphere should experience significantly stronger continuum + molecular
    attenuation than cold subarctic winter.
    """
    sc_sub = Scenario(
        atmosphere=ModelAtmosphere.SUBARCTIC_WINTER,
        sensor_alt_km=0.01,
        sensor_zenith_deg=90.0,
        source_range_km=15.0,
        wavenumber_min_cm1=2500.0,
        wavenumber_max_cm1=2750.0,
        resolution_cm1=5.0,
        aerosol=AerosolModel.NONE,  # isolate molecular + continuum
    )
    res_sub = run_scenario(sc_sub)
    tau_subarctic = float(res_sub.transmittance.total.mean())

    sc_trop = Scenario(
        atmosphere=ModelAtmosphere.TROPICAL,
        sensor_alt_km=0.01,
        sensor_zenith_deg=90.0,
        source_range_km=15.0,
        wavenumber_min_cm1=2500.0,
        wavenumber_max_cm1=2750.0,
        resolution_cm1=5.0,
        aerosol=AerosolModel.NONE,  # isolate molecular + continuum
    )
    res_trop = run_scenario(sc_trop)
    tau_tropical = float(res_trop.transmittance.total.mean())

    # Cold subarctic winter has very little water vapor -> high transmission
    assert tau_subarctic > 0.90, f"Subarctic winter should be very transparent, got {tau_subarctic}"
    # Tropical has strong water vapor absorption and continuum attenuation
    assert tau_tropical < tau_subarctic - 0.15, (
        f"Tropical ({tau_tropical}) must be significantly lower than subarctic ({tau_subarctic})"
    )


# =========================================================================
# 5. Real-World Ingestion with Sharp Inversions
# =========================================================================


def test_custom_inversion_sounding_stability():
    """Verify that a non-monotonic sounding (e.g. strong 14 K nocturnal boundary layer
    temperature inversion) solves stably with non-negative, well-ordered optical depths.
    """
    alt_km = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0])
    p_mbar = 1015.0 * np.exp(-alt_km / 7.2)
    # Surface at 280 K, sharp inversion to 294 K at 300 m, then standard lapse rate
    t_k = np.array([280.0, 286.0, 294.0, 291.0, 288.0, 282.0, 260.0, 230.0, 220.0, 190.0])
    h2o_ppmv = np.array([8000.0, 7500.0, 6000.0, 4000.0, 2500.0, 1000.0, 200.0, 20.0, 4.0, 0.4])
    o3_ppmv = np.full_like(alt_km, 0.04)

    profile = AtmosphereProfile(
        model=ModelAtmosphere.US_STANDARD,
        altitude_km=alt_km,
        pressure_mbar=p_mbar,
        temperature_k=t_k,
        vmr_ppmv={
            Species.H2O: h2o_ppmv,
            Species.CO2: np.full_like(alt_km, 423.0),
            Species.O3: o3_ppmv,
            Species.N2O: np.full_like(alt_km, 0.337),
            Species.CO: np.full_like(alt_km, 0.10),
            Species.CH4: np.full_like(alt_km, 1.92),
            Species.O2: np.full_like(alt_km, 209500.0),
        },
    )

    grid = make_grid(wavenumber_min_cm1=2200.0, wavenumber_max_cm1=2600.0, resolution_cm1=10.0)
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=45.0, max_layers=30)
    optics = compute_layer_optics(path, grid, aerosol=AerosolModel.RURAL, visibility_km=15.0)

    # All optical depths must be positive and finite
    assert np.all(optics.tau_total >= 0.0)
    assert np.all(np.isfinite(optics.tau_total))
    # Transmission must be strictly bounded in (0, 1]
    trans_total = np.exp(-optics.tau_total.sum(axis=0))
    assert np.all(trans_total >= 0.0)
    assert np.all(trans_total <= 1.0)


# =========================================================================
# 6. Turbulence Wavelength Scaling & Scintillation Saturation Limit
# =========================================================================


def test_turbulence_wavelength_scaling_power_law():
    """Fried parameter r0 and isoplanatic angle theta0 scale as lambda^(6/5) = lambda^1.2.
    Ratio r0(4.0 um) / r0(0.5 um) = (4.0 / 0.5)^1.2 = 8^1.2 = 12.128355.
    """
    r0_vis = fried_parameter_r0_m(
        cn2_m23=np.array([1e-14]),
        ds_m=np.array([1000.0]),
        wavelength_um=0.5,
    )
    r0_mwir = fried_parameter_r0_m(
        cn2_m23=np.array([1e-14]),
        ds_m=np.array([1000.0]),
        wavelength_um=4.0,
    )
    expected_ratio = (4.0 / 0.5) ** 1.2
    assert r0_mwir / r0_vis == pytest.approx(expected_ratio, rel=1e-5)

    theta0_vis = isoplanatic_angle_rad(
        cn2_m23=np.array([1e-14]),
        sensor_distance_m=np.array([500.0]),
        ds_m=np.array([1000.0]),
        wavelength_um=0.5,
    )
    theta0_mwir = isoplanatic_angle_rad(
        cn2_m23=np.array([1e-14]),
        sensor_distance_m=np.array([500.0]),
        ds_m=np.array([1000.0]),
        wavelength_um=4.0,
    )
    assert theta0_mwir / theta0_vis == pytest.approx(expected_ratio, rel=1e-5)


def test_scintillation_index_strong_turbulence_saturation():
    """In strong turbulence (Rytov variance sigma_R^2 >> 1), the scintillation index
    must saturate towards 1.0 rather than growing unbounded (Andrews & Phillips 2005).
    """
    # Weak regime: sigma_I^2 approx sigma_R^2
    sigma_r2_weak = 0.05
    sigma_i2_weak = scintillation_index(sigma_r2_weak)
    assert sigma_i2_weak == pytest.approx(sigma_r2_weak, rel=0.1)

    # Strong saturated regime: sigma_R^2 = 100 -> sigma_I^2 saturates to ~1.0
    sigma_r2_strong = 100.0
    sigma_i2_strong = scintillation_index(sigma_r2_strong)
    assert 0.9 <= sigma_i2_strong <= 1.5
