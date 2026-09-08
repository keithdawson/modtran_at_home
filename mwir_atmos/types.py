"""Shared data contracts between the physics core and the radiometry/UI layer.

OVERSEER-OWNED. Worker agents must NOT edit this file. Every cross-module value
flows through the dataclasses defined here; if you believe a field is missing or
wrong, implement around it and say so in your report -- do not change it
unilaterally, because the other worker is coding against it concurrently.

All arrays are ``numpy.ndarray`` of dtype float64. Spectral arrays are indexed
on the *same* wavenumber grid (``SpectralGrid.wavenumber_cm1``) everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


# =========================================================================
# Enumerations
# =========================================================================


class ModelAtmosphere(str, Enum):
    """The six standard LOWTRAN/MODTRAN vertical profiles."""

    TROPICAL = "tropical"
    MIDLAT_SUMMER = "midlat_summer"
    MIDLAT_WINTER = "midlat_winter"
    SUBARCTIC_SUMMER = "subarctic_summer"
    SUBARCTIC_WINTER = "subarctic_winter"
    US_STANDARD = "us_standard"


class SurfaceType(str, Enum):
    """Lower boundary condition: drives emissivity and BRDF, not the profile."""

    SEA_WATER = "sea_water"
    LAND_VEGETATED = "land_vegetated"
    DESERT_SAND = "desert_sand"
    SNOW = "snow"
    BLACKBODY = "blackbody"


class AerosolModel(str, Enum):
    """LOWTRAN-style boundary-layer aerosol types."""

    RURAL = "rural"
    URBAN = "urban"
    NAVY_MARITIME = "navy_maritime"
    DESERT = "desert"
    TROPOSPHERIC = "tropospheric"
    NONE = "none"


class Species(str, Enum):
    """Radiatively active species tracked per layer."""

    H2O = "H2O"
    CO2 = "CO2"
    O3 = "O3"
    N2O = "N2O"
    CO = "CO"
    CH4 = "CH4"
    O2 = "O2"


#: Species whose column amounts every profile must supply.
REQUIRED_SPECIES: tuple[Species, ...] = (
    Species.H2O,
    Species.CO2,
    Species.O3,
    Species.N2O,
    Species.CO,
    Species.CH4,
)


class TurbulenceModel(str, Enum):
    """Cn^2 vertical profile models (stretch goal)."""

    NONE = "none"
    HV57 = "hv57"
    SLC_DAY = "slc_day"
    SLC_NIGHT = "slc_night"
    CONSTANT = "constant"


# =========================================================================
# Spectral grid
# =========================================================================


@dataclass(frozen=True)
class SpectralGrid:
    """Monotonically increasing wavenumber grid, uniform spacing.

    Every spectral array in the package is defined on this grid. Construct with
    :func:`mwir_atmos.spectral.make_grid`.
    """

    wavenumber_cm1: np.ndarray
    resolution_cm1: float

    @property
    def wavelength_um(self) -> np.ndarray:
        return 1.0e4 / self.wavenumber_cm1

    @property
    def n(self) -> int:
        return int(self.wavenumber_cm1.size)


# =========================================================================
# Atmospheric state
# =========================================================================


@dataclass
class AtmosphereProfile:
    """Vertical state of the atmosphere on a set of levels.

    ``altitude_km`` is strictly increasing, starts at the surface (which may be
    above sea level) and ends at :data:`constants.TOA_ALTITUDE_KM`. All arrays
    share shape ``(n_levels,)``.

    ``vmr`` maps every member of :data:`REQUIRED_SPECIES` (plus optionally O2)
    to a volume mixing ratio profile in **ppmv**.
    """

    altitude_km: np.ndarray
    pressure_mbar: np.ndarray
    temperature_k: np.ndarray
    vmr_ppmv: dict[Species, np.ndarray]
    model: ModelAtmosphere
    surface_altitude_km: float = 0.0
    #: Relative humidity (%) derived from T and H2O vmr; used by aerosol growth.
    relative_humidity_pct: Optional[np.ndarray] = None

    @property
    def n_levels(self) -> int:
        return int(self.altitude_km.size)


# =========================================================================
# Path geometry
# =========================================================================


@dataclass
class PathLayer:
    """One homogenised slab along the slant path, sensor-to-source ordered.

    Layer 0 is adjacent to the sensor. ``column`` holds the slant-path column
    density (molecules cm^-2) for each species -- already multiplied by the
    slant length, so no further airmass factor is applied downstream.

    ``p_eff_mbar`` / ``t_eff_k`` are the Curtis-Godson effective pressure and
    temperature per species (column-weighted), which the band model needs to
    handle pressure broadening along an inhomogeneous path.
    """

    index: int
    z_bottom_km: float
    z_top_km: float
    ds_km: float
    p_mbar: float
    t_k: float
    rh_pct: float
    air_column_cm2: float
    column_cm2: dict[Species, float]
    p_eff_mbar: dict[Species, float]
    t_eff_k: dict[Species, float]
    #: Aerosol extinction path integral at 0.55 um, dimensionless optical depth.
    aerosol_od_550nm: float = 0.0


@dataclass
class PathGeometry:
    """A resolved sensor-to-source path through the layered spherical atmosphere.

    Layers run from the sensor outward. When ``hits_ground`` the last layer ends
    at the surface and the surface terms in the radiance model are active. When
    ``tangent_alt_km`` is not None the path is a limb view that turned over
    without reaching the ground.
    """

    layers: list[PathLayer]
    sensor_alt_km: float
    sensor_zenith_deg: float
    sensor_azimuth_deg: float
    source_alt_km: Optional[float]
    total_length_km: float
    hits_ground: bool
    reaches_space: bool
    tangent_alt_km: Optional[float] = None
    #: Ground intersection temperature, populated when ``hits_ground``.
    ground_temp_k: Optional[float] = None

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# =========================================================================
# Optical results
# =========================================================================


@dataclass
class LayerOptics:
    """Per-layer optical depth, split by mechanism.

    Every array has shape ``(n_layers, n_wavenumbers)``, row-aligned with
    ``PathGeometry.layers``. Radiance integration needs this decomposition:
    scattering source terms use ``tau_aerosol_sca`` + ``tau_rayleigh`` while
    thermal emission uses absorption only.
    """

    tau_molecular: np.ndarray
    tau_aerosol_abs: np.ndarray
    tau_aerosol_sca: np.ndarray
    tau_rayleigh: np.ndarray
    grid: SpectralGrid
    #: Optional per-species molecular breakdown, same shape, for diagnostics.
    tau_by_species: dict[Species, np.ndarray] = field(default_factory=dict)
    #: Aerosol scattering asymmetry parameter g per layer, shape (n_layers, n_wn).
    aerosol_asymmetry: Optional[np.ndarray] = None

    @property
    def tau_absorption(self) -> np.ndarray:
        return self.tau_molecular + self.tau_aerosol_abs

    @property
    def tau_scattering(self) -> np.ndarray:
        return self.tau_aerosol_sca + self.tau_rayleigh

    @property
    def tau_total(self) -> np.ndarray:
        return self.tau_absorption + self.tau_scattering


@dataclass
class TransmittanceResult:
    """Sensor-to-source transmittance, total and decomposed."""

    grid: SpectralGrid
    total: np.ndarray
    molecular: np.ndarray
    aerosol: np.ndarray
    rayleigh: np.ndarray
    by_species: dict[Species, np.ndarray] = field(default_factory=dict)
    #: Cumulative transmittance from the sensor to each layer *boundary*,
    #: shape ``(n_layers + 1, n_wavenumbers)``. Row 0 is all ones. Required by
    #: the radiance integration -- do not omit.
    cumulative: Optional[np.ndarray] = None

    @property
    def wavelength_um(self) -> np.ndarray:
        return self.grid.wavelength_um


# =========================================================================
# Sky geometry (sun / moon)
# =========================================================================


@dataclass
class SkyPosition:
    """Apparent position of a celestial body in the sensor's local horizon frame.

    ``azimuth_deg`` is measured clockwise from true north. ``zenith_deg`` > 90
    means the body is below the horizon.
    """

    zenith_deg: float
    azimuth_deg: float
    #: Angle between the sensor line of sight and the body, degrees.
    scattering_angle_deg: float
    is_above_horizon: bool
    #: Body-to-observer distance scaling applied to the exo-atmospheric
    #: irradiance (1.0 = mean distance).
    distance_factor: float = 1.0
    #: Illuminated fraction of the disc, 0-1. Always 1.0 for the sun.
    illuminated_fraction: float = 1.0


# =========================================================================
# Radiance result
# =========================================================================


@dataclass
class RadianceResult:
    """Spectral radiance at the sensor aperture, decomposed by origin.

    Units: W cm^-2 sr^-1 / cm^-1. Every component array has shape
    ``(n_wavenumbers,)``. ``total`` is the sum of the components that are
    physically present for the scenario (components that do not apply are
    zero-filled, never None).
    """

    grid: SpectralGrid
    path_thermal: np.ndarray
    surface_emitted: np.ndarray
    surface_reflected_solar: np.ndarray
    surface_reflected_lunar: np.ndarray
    solar_single_scatter: np.ndarray
    lunar_single_scatter: np.ndarray
    direct_solar: np.ndarray
    direct_lunar: np.ndarray
    source_emitted: np.ndarray
    total: np.ndarray

    @property
    def wavelength_um(self) -> np.ndarray:
        return self.grid.wavelength_um


# =========================================================================
# Turbulence result (stretch goal)
# =========================================================================


@dataclass
class TurbulenceResult:
    """Optical turbulence metrics along the path, at a reference wavelength."""

    reference_wavelength_um: float
    cn2_profile_m23: np.ndarray          # per layer, m^-2/3
    fried_parameter_r0_cm: float
    isoplanatic_angle_urad: float
    rytov_variance: float
    scintillation_index: float
    greenwood_frequency_hz: Optional[float] = None
    strehl_ratio: Optional[float] = None

    # --- point-source / aperture response -------------------------------
    #: Aperture the response quantities below were evaluated for, metres.
    aperture_diameter_m: Optional[float] = None
    #: Diffraction-limited angular resolution, lambda/D, microradians.
    diffraction_limit_urad: Optional[float] = None
    #: Turbulence-limited ("seeing") angular blur, lambda/r0, microradians.
    seeing_limit_urad: Optional[float] = None
    #: Actual long-exposure spot size, the quadrature sum of the two above.
    effective_blur_urad: Optional[float] = None
    #: True when D > r0, i.e. turbulence -- not the aperture -- sets the spot
    #: size. Below this the aperture is still diffraction-limited and
    #: turbulence barely matters.
    is_seeing_limited: Optional[bool] = None
    #: On-axis peak irradiance of an unresolved source relative to the
    #: turbulence-free diffraction-limited case. Equal to the Strehl ratio.
    #: Turbulence conserves total power -- this is redistribution, not loss.
    peak_irradiance_ratio: Optional[float] = None


# =========================================================================
# Scenario (the complete user-facing input)
# =========================================================================


@dataclass
class Scenario:
    """Everything the CLI or GUI collects. Serialisable to/from YAML.

    This is the single input object to :func:`mwir_atmos.api.run_scenario`.
    """

    # --- atmosphere -------------------------------------------------------
    atmosphere: ModelAtmosphere = ModelAtmosphere.MIDLAT_SUMMER
    surface_type: SurfaceType = SurfaceType.LAND_VEGETATED
    aerosol: AerosolModel = AerosolModel.RURAL
    visibility_km: float = 23.0
    #: Surface air temperature (K). None keeps the model-atmosphere value.
    ground_temp_k: Optional[float] = None
    #: Surface relative humidity (%). None keeps the model-atmosphere value.
    ground_rh_pct: Optional[float] = None
    #: Skin temperature of the ground/sea surface (K). Defaults to ground_temp_k.
    surface_skin_temp_k: Optional[float] = None
    surface_emissivity: Optional[float] = None
    co2_ppmv: Optional[float] = None
    #: Navy-maritime model only: 1 (open ocean) to 10 (heavy surf).
    wind_speed_m_s: float = 5.0
    #: Desert model only: surface wind driving lofted dust.
    surface_altitude_km: float = 0.0

    # --- sensor -----------------------------------------------------------
    sensor_alt_km: float = 0.0
    #: 0 = straight up, 90 = horizontal, 180 = straight down.
    sensor_zenith_deg: float = 0.0
    #: Degrees clockwise from true north.
    sensor_azimuth_deg: float = 0.0
    latitude_deg: float = 35.0
    longitude_deg: float = 0.0

    # --- source (target) --------------------------------------------------
    #: None means "integrate to the end of the path" (ground or space).
    source_alt_km: Optional[float] = None
    source_range_km: Optional[float] = None
    source_temp_k: Optional[float] = None
    source_emissivity: float = 1.0
    source_area_m2: Optional[float] = None

    # --- time / illumination ---------------------------------------------
    #: Local clock time. Timezone is approximated from longitude unless
    #: ``utc_offset_hours`` is given, so the sun position is intentionally
    #: "loose" -- good to a degree or so, not an ephemeris-grade fix.
    local_datetime: Optional[datetime] = None
    utc_offset_hours: Optional[float] = None
    #: Observed local sunrise/sunset at the sensor, in local clock hours
    #: (6.75 == 06:45). Supplying BOTH calibrates the sun exactly and overrides
    #: the timezone entirely: their midpoint is local solar noon on the
    #: observer's own clock, which removes the timezone error, and their
    #: separation gives the day length, which fixes the declination. Without
    #: them the timezone is guessed from longitude and can be an hour or more
    #: off, which is tens of degrees of solar position error.
    sunrise_local_hours: Optional[float] = None
    sunset_local_hours: Optional[float] = None
    #: 0.0 = new moon, 0.5 = full moon, 1.0 = new moon again.
    moon_phase: float = 0.5
    include_solar: bool = True
    include_lunar: bool = True

    # --- spectral ---------------------------------------------------------
    wavenumber_min_cm1: float = 1800.0
    wavenumber_max_cm1: float = 4300.0
    resolution_cm1: float = 1.0
    #: Line-by-line accuracy controls. The defaults are tuned for full-band runs
    #: at ~1 cm^-1. For high-resolution work over a narrow window, lower
    #: ``line_strength_cutoff`` (1e-7, or 0.0 for every line) -- at 0.001 cm^-1
    #: the default 1e-4 can shift individual line cores by a large fraction of
    #: their depth, even though the band average is barely affected.
    line_strength_cutoff: float = 1.0e-4
    line_wing_cutoff_cm1: float = 25.0

    # --- turbulence (stretch) --------------------------------------------
    turbulence: TurbulenceModel = TurbulenceModel.NONE
    turbulence_cn2_ground: Optional[float] = None
    #: Clear aperture diameter of the receiving optic, metres. Drives the
    #: Strehl ratio and the point-source blur reported in TurbulenceResult.
    #: Turbulence only degrades a point source once D exceeds the Fried
    #: parameter r0 -- which in the MWIR is ~60 cm at zenith, so modest
    #: apertures stay diffraction-limited.
    sensor_aperture_diameter_m: float = 0.10


@dataclass
class ScenarioResult:
    """Bundle returned by :func:`mwir_atmos.api.run_scenario`."""

    scenario: Scenario
    grid: SpectralGrid
    profile: AtmosphereProfile
    path: PathGeometry
    transmittance: TransmittanceResult
    #: None when radiance was not requested, or before the radiometry layer
    #: exists (wave 1 produces transmittance only).
    radiance: Optional[RadianceResult] = None
    sun: Optional[SkyPosition] = None
    moon: Optional[SkyPosition] = None
    turbulence: Optional[TurbulenceResult] = None
