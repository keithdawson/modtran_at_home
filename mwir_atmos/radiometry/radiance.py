"""Spectral radiance at the sensor aperture, decomposed by origin.

The transfer model is single-scattering: thermal emission is integrated along
the path with full absorption, and solar/lunar photons are allowed exactly one
scattering event. That is a good approximation in the MWIR, where scattering
optical depths are small compared with molecular absorption, and it is why
:class:`LayerOptics` keeps absorption and scattering separate.

Illumination beam geometry is approximated. The layer optical depths on hand
belong to the *sensor* line of sight, not to the sun-to-layer path, so the beam
transmittance is reconstructed by de-slanting each layer to a vertical optical
depth (scaling by ``dz/ds``) and re-slanting along the illumination direction
with a Kasten-Young airmass. This is exact for a plane-parallel atmosphere and
degrades gracefully at large zenith angles; it does not capture the horizontal
inhomogeneity a true 3D solar path would see.

Airmass reference: Kasten, F. and Young, A.T. (1989), "Revised optical air mass
tables and approximation formula", Applied Optics 28(22), 4735-4738.
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import (
    LUNAR_SOLID_ANGLE_SR,
    SOLAR_ANGULAR_RADIUS_RAD,
    SOLAR_CONSTANT_W_M2,
    SOLAR_EFFECTIVE_TEMP_K,
    SOLAR_SOLID_ANGLE_SR,
    STEFAN_BOLTZMANN,
)
from ..spectral import planck
from ..types import (
    AtmosphereProfile,
    LayerOptics,
    PathGeometry,
    RadianceResult,
    Scenario,
    SkyPosition,
    SpectralGrid,
    TransmittanceResult,
)
from .surface import surface_emissivity

__all__ = ["compute_radiance", "solar_irradiance", "lunar_irradiance", "airmass"]

#: Subsolar brightness temperature of the lunar surface. The night side falls to
#: ~100 K, which is negligible in the MWIR, so only the lit fraction is modelled.
LUNAR_SUBSOLAR_TEMP_K = 390.0
LUNAR_EMISSIVITY = 0.95
#: Broadband lunar geometric albedo used for the (small) reflected-solar term.
LUNAR_ALBEDO = 0.12


def airmass(zenith_deg: float) -> float:
    """Relative airmass, finite at the horizon (Kasten & Young 1989)."""
    z = min(max(zenith_deg, 0.0), 90.0)
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def solar_irradiance(grid: SpectralGrid, distance_factor: float = 1.0) -> np.ndarray:
    """Top-of-atmosphere solar spectral irradiance, W cm^-2 / cm^-1.

    A 5778 K blackbody filling the solar solid angle, renormalised so that the
    broadband integral equals the solar constant. The true solar spectrum
    departs from a blackbody -- notably Fraunhofer structure and a cooler
    effective temperature in the infrared -- but across the MWIR the blackbody
    shape is adequate, and solar terms are small there in any case.
    """
    raw = planck(grid.wavenumber_cm1, SOLAR_EFFECTIVE_TEMP_K) * SOLAR_SOLID_ANGLE_SR
    broadband = (
        STEFAN_BOLTZMANN * SOLAR_EFFECTIVE_TEMP_K**4 / math.pi * 1.0e-4 * SOLAR_SOLID_ANGLE_SR
    )
    return raw * (SOLAR_CONSTANT_W_M2 * 1.0e-4 / broadband) * distance_factor


def lunar_irradiance(grid: SpectralGrid, moon: SkyPosition) -> tuple[np.ndarray, np.ndarray]:
    """Lunar spectral irradiance at the top of atmosphere, W cm^-2 / cm^-1.

    Returns ``(thermal, reflected)`` separately. In the MWIR the Moon's own
    thermal emission at ~390 K dominates its reflected sunlight by orders of
    magnitude -- a lunar model built only from albedo-scaled sunlight is badly
    wrong here, which is the usual mistake.
    """
    lit = float(moon.illuminated_fraction)
    thermal = (
        LUNAR_EMISSIVITY
        * planck(grid.wavenumber_cm1, LUNAR_SUBSOLAR_TEMP_K)
        * LUNAR_SOLID_ANGLE_SR
        * lit
    )
    reflected = LUNAR_ALBEDO * solar_irradiance(grid) / math.pi * LUNAR_SOLID_ANGLE_SR * lit
    return thermal, reflected


def _henyey_greenstein(g: np.ndarray, theta_deg: float) -> np.ndarray:
    """Henyey-Greenstein phase function, normalised to 1 over 4*pi steradians."""
    mu = math.cos(math.radians(theta_deg))
    g = np.clip(g, -0.99, 0.99)
    return (1.0 - g**2) / np.power(1.0 + g**2 - 2.0 * g * mu, 1.5)


def _rayleigh_phase(theta_deg: float) -> float:
    mu = math.cos(math.radians(theta_deg))
    return 0.75 * (1.0 + mu * mu)


def _vertical_optical_depth(path: PathGeometry, optics: LayerOptics) -> np.ndarray:
    """De-slant each layer's extinction to a vertical optical depth."""
    tau = optics.tau_total
    scale = np.array(
        [
            (abs(l.z_top_km - l.z_bottom_km) / l.ds_km) if l.ds_km > 0 else 0.0
            for l in path.layers
        ]
    )
    return tau * scale[:, None]


def _beam_transmittance_to_layers(
    path: PathGeometry, optics: LayerOptics, body_zenith_deg: float
) -> np.ndarray:
    """Transmittance from space down to the top of each layer along the beam.

    Shape ``(n_layers, n_wavenumbers)``. Layers are ordered along the line of
    sight, which may run either up or down, so the column above each layer is
    accumulated by altitude rather than by index.
    """
    tau_vert = _vertical_optical_depth(path, optics)
    mid_alt = np.array([0.5 * (l.z_bottom_km + l.z_top_km) for l in path.layers])
    order = np.argsort(-mid_alt)                 # highest first

    above = np.zeros_like(tau_vert)
    running = np.zeros(tau_vert.shape[1])
    for idx in order:
        above[idx] = running
        running = running + tau_vert[idx]

    return np.exp(-above * airmass(body_zenith_deg))


def _beam_transmittance_to_surface(
    path: PathGeometry, optics: LayerOptics, body_zenith_deg: float
) -> np.ndarray:
    """Transmittance from space to the ground along the illumination beam."""
    tau_vert = _vertical_optical_depth(path, optics)
    return np.exp(-tau_vert.sum(axis=0) * airmass(body_zenith_deg))


def _single_scatter(
    grid: SpectralGrid,
    path: PathGeometry,
    optics: LayerOptics,
    transmittance: TransmittanceResult,
    body: SkyPosition,
    irradiance: np.ndarray,
) -> np.ndarray:
    """Single-scattered radiance from one illumination source."""
    if not body.is_above_horizon:
        return np.zeros(grid.n)

    tau_total = optics.tau_total
    tau_sca = optics.tau_scattering
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(tau_total > 0, tau_sca / tau_total, 0.0)

    g = (
        optics.aerosol_asymmetry
        if optics.aerosol_asymmetry is not None
        else np.zeros_like(tau_total)
    )
    p_aer = _henyey_greenstein(g, body.scattering_angle_deg)
    p_ray = _rayleigh_phase(body.scattering_angle_deg)

    aer_sca, ray_sca = optics.tau_aerosol_sca, optics.tau_rayleigh
    denom = aer_sca + ray_sca
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.where(denom > 0, (p_aer * aer_sca + p_ray * ray_sca) / denom, 0.0)

    beam = _beam_transmittance_to_layers(path, optics, body.zenith_deg)
    to_sensor = transmittance.cumulative[:-1]
    emitted = 1.0 - np.exp(-tau_total)

    contrib = omega * phase / (4.0 * math.pi) * irradiance[None, :] * beam * to_sensor * emitted
    return contrib.sum(axis=0)


def compute_radiance(
    *,
    scenario: Scenario,
    grid: SpectralGrid,
    profile: AtmosphereProfile,
    path: PathGeometry,
    optics: LayerOptics,
    transmittance: TransmittanceResult,
    sun: SkyPosition,
    moon: SkyPosition,
) -> RadianceResult:
    """Assemble sensor-aperture spectral radiance, W cm^-2 sr^-1 / cm^-1."""
    n = grid.n
    zeros = np.zeros(n)
    total_trans = transmittance.total
    cumulative = transmittance.cumulative

    # --- path thermal emission (Schwarzschild) --------------------------------
    layer_t = np.array([l.t_k for l in path.layers])
    b_layers = planck(grid.wavenumber_cm1, layer_t)          # (n_layers, n)
    layer_emissivity = 1.0 - np.exp(-optics.tau_absorption)
    path_thermal = (b_layers * layer_emissivity * cumulative[:-1]).sum(axis=0)

    # --- surface terms --------------------------------------------------------
    surface_emitted = zeros.copy()
    surface_reflected_solar = zeros.copy()
    surface_reflected_lunar = zeros.copy()

    if path.hits_ground:
        skin_t = scenario.surface_skin_temp_k
        if skin_t is None:
            skin_t = scenario.ground_temp_k
        if skin_t is None:
            skin_t = path.ground_temp_k
        if skin_t is None:
            skin_t = float(profile.temperature_k[0])

        incidence = abs(180.0 - scenario.sensor_zenith_deg)
        eps = surface_emissivity(
            scenario.surface_type,
            incidence_deg=incidence,
            override=scenario.surface_emissivity,
        )
        surface_emitted = eps * planck(grid.wavenumber_cm1, skin_t) * total_trans

        if scenario.include_solar and sun.is_above_horizon:
            e_sun = solar_irradiance(grid, sun.distance_factor)
            t_beam = _beam_transmittance_to_surface(path, optics, sun.zenith_deg)
            surface_reflected_solar = (
                (1.0 - eps) / math.pi
                * e_sun
                * math.cos(math.radians(min(sun.zenith_deg, 89.9)))
                * t_beam
                * total_trans
            )

        if scenario.include_lunar and moon.is_above_horizon:
            m_thermal, m_reflected = lunar_irradiance(grid, moon)
            t_beam = _beam_transmittance_to_surface(path, optics, moon.zenith_deg)
            surface_reflected_lunar = (
                (1.0 - eps) / math.pi
                * (m_thermal + m_reflected)
                * math.cos(math.radians(min(moon.zenith_deg, 89.9)))
                * t_beam
                * total_trans
            )

    # --- scattered and direct illumination ------------------------------------
    solar_single_scatter = zeros.copy()
    lunar_single_scatter = zeros.copy()
    direct_solar = zeros.copy()
    direct_lunar = zeros.copy()

    if scenario.include_solar and sun.is_above_horizon:
        e_sun = solar_irradiance(grid, sun.distance_factor)
        solar_single_scatter = _single_scatter(
            grid, path, optics, transmittance, sun, e_sun
        )
        if sun.scattering_angle_deg <= math.degrees(SOLAR_ANGULAR_RADIUS_RAD):
            direct_solar = e_sun / SOLAR_SOLID_ANGLE_SR * total_trans

    if scenario.include_lunar and moon.is_above_horizon:
        m_thermal, m_reflected = lunar_irradiance(grid, moon)
        e_moon = m_thermal + m_reflected
        lunar_single_scatter = _single_scatter(
            grid, path, optics, transmittance, moon, e_moon
        )
        if moon.scattering_angle_deg <= 0.2586:      # lunar angular radius, deg
            direct_lunar = e_moon / LUNAR_SOLID_ANGLE_SR * total_trans

    # --- an explicit target at the far end of the path ------------------------
    source_emitted = zeros.copy()
    if scenario.source_temp_k is not None:
        source_emitted = (
            scenario.source_emissivity
            * planck(grid.wavenumber_cm1, scenario.source_temp_k)
            * total_trans
        )

    total = (
        path_thermal
        + surface_emitted
        + surface_reflected_solar
        + surface_reflected_lunar
        + solar_single_scatter
        + lunar_single_scatter
        + direct_solar
        + direct_lunar
        + source_emitted
    )

    return RadianceResult(
        grid=grid,
        path_thermal=path_thermal,
        surface_emitted=surface_emitted,
        surface_reflected_solar=surface_reflected_solar,
        surface_reflected_lunar=surface_reflected_lunar,
        solar_single_scatter=solar_single_scatter,
        lunar_single_scatter=lunar_single_scatter,
        direct_solar=direct_solar,
        direct_lunar=direct_lunar,
        source_emitted=source_emitted,
        total=total,
    )
