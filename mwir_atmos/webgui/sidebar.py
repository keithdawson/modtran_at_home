"""GUI sidebar: widgets -> :class:`~mwir_atmos.types.Scenario`.

The sidebar groups the same fields as the CLI's flags (atmosphere / sensor /
source / illumination / spectral / accuracy). Scenario-affecting controls use
``number_input``/``selectbox``/``checkbox`` rather than continuous-drag
sliders, so an in-progress drag does not fire a cascade of intermediate
(uncached) physics runs -- only a committed value does.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime
from typing import Optional

import streamlit as st
import yaml

from mwir_atmos import constants
from mwir_atmos.scenario_io import scenario_from_dict
from mwir_atmos.setups import with_overrides
from mwir_atmos.types import (
    AerosolModel,
    InstrumentFunction,
    ModelAtmosphere,
    RadianceSolver,
    Scenario,
    SpectralSampling,
    SurfaceType,
    TurbulenceModel,
)

__all__ = ["AEROSOL_LABELS", "MODEL_LABELS", "build_scenario_from_sidebar"]

MODEL_LABELS = {m: m.value.replace("_", " ").title() for m in ModelAtmosphere}
SURFACE_LABELS = {s: s.value.replace("_", " ").title() for s in SurfaceType}
AEROSOL_LABELS = {a: a.value.replace("_", " ").title() for a in AerosolModel}
AEROSOL_LABELS.update({a: "LOWTRAN table: " + a.value.removeprefix("lowtran_").title()
                       for a in AerosolModel if a.value.startswith("lowtran_")})

#: Quick spectral-range presets, so a narrow window is quick to select.
#: Selecting one pins wavenumber-min/max/resolution and the accuracy
#: controls; switching to "Custom" (or just editing again) frees the fields
#: for hand-tuning -- see _spectral_and_accuracy_controls.
_SPECTRAL_PRESETS: dict[str, dict[str, float]] = {
    "Full band (default)": dict(
        wn_min=constants.MWIR_GRID_MIN_CM1, wn_max=constants.MWIR_GRID_MAX_CM1,
        resolution=4.0, wing=25.0, strength=0.0, tau=3.0e-6,
    ),
    "MWIR core, 3-5 um": dict(wn_min=2000.0, wn_max=3333.33, resolution=2.0, wing=25.0,
                              strength=0.0, tau=3.0e-6),
    "Narrow + fine (CO line demo)": dict(wn_min=2171.5, wn_max=2174.0, resolution=0.01, wing=3.0,
                                         strength=0.0, tau=1.0e-7),
}

_SAMPLING_LABELS = {
    SpectralSampling.POINT: "Point (fast: physics at output points only)",
    SpectralSampling.RESOLVED: "Line-resolved + instrument function (accurate)",
}

_SOLVER_LABELS = {
    RadianceSolver.SINGLE_SCATTER: "Single scattering (spherical paths)",
    RadianceSolver.DISORT: "DISORT multiple scattering (plane-parallel)",
    RadianceSolver.SPHERICAL_MC: "Spherical multiple scattering (Monte Carlo)",
}


def build_scenario_from_sidebar() -> Scenario:
    """The scenario the sidebar describes: from the controls, or a loaded YAML setup."""
    st.sidebar.title("mwir_atmos")
    source = st.sidebar.radio("Setup source", ["Current controls", "Saved YAML"], key="setup_source")
    if source == "Saved YAML":
        return _scenario_from_uploaded_yaml()
    fields: dict = {}
    fields.update(_atmosphere_controls())
    fields.update(_sensor_controls())
    fields.update(_source_controls())
    fields.update(_illumination_controls())
    fields.update(_spectral_and_accuracy_controls())
    fields.update(_solver_controls())
    return Scenario(**fields)


# =========================================================================
# Small widget helpers
# =========================================================================


def _optional_number(label: str, *, key: str, default: float, help: Optional[str] = None,
                     **kwargs) -> Optional[float]:
    """A checkbox-gated number_input: unchecked -> None (keep the model/scenario
    default), matching the CLI's "flag omitted -> None" semantics exactly."""
    enabled = st.checkbox(f"Override {label}", key=f"{key}__on")
    if not enabled:
        return None
    return float(st.number_input(label, value=default, key=key, help=help, **kwargs))


def _enum_select(label: str, options: dict, current, *, key: str):
    keys = list(options.keys())
    idx = keys.index(current) if current in keys else 0
    return st.sidebar.selectbox(label, keys, index=idx, format_func=lambda k: options[k], key=key)


# =========================================================================
# Control groups, each returning the Scenario fields it sets
# =========================================================================


def _scenario_from_uploaded_yaml() -> Scenario:
    uploaded = st.sidebar.file_uploader("Open saved setup", type=["yaml", "yml"], key="setup_upload")
    st.sidebar.caption("Use a downloaded setup or an existing CLI configuration. Overrides replace only the named values.")
    overrides = st.sidebar.text_area("Overrides (YAML)", value="{}", key="setup_overrides",
                                     help="Example: source_range_km: 5.0. Use null to clear an optional value.")
    if uploaded is None:
        st.info("Choose a YAML setup in the sidebar to continue.")
        st.stop()
    try:
        data = yaml.safe_load(uploaded.getvalue().decode("utf-8-sig"))
        changes = yaml.safe_load(overrides)
        if not isinstance(data, dict) or not isinstance(changes, dict):
            raise ValueError("Setup and overrides must be YAML mappings")
        return with_overrides(scenario_from_dict(data), **changes)
    except (ValueError, TypeError, UnicodeError, yaml.YAMLError) as exc:
        st.error(f"Could not load saved setup: {exc}")
        st.stop()


def _atmosphere_controls() -> dict:
    st.sidebar.header("Atmosphere")
    atmosphere = _enum_select("Model atmosphere", MODEL_LABELS, ModelAtmosphere.MIDLAT_SUMMER, key="atmosphere")
    surface_type = _enum_select("Surface type", SURFACE_LABELS, SurfaceType.LAND_VEGETATED, key="surface_type")
    aerosol = _enum_select("Aerosol model", AEROSOL_LABELS, AerosolModel.RURAL, key="aerosol")
    if aerosol.value.startswith("lowtran_"):
        st.sidebar.caption(
            "LOWTRAN 7 optical tables, RH 0–99%, background altitude profiles. "
            "Visibility: 2–50 km. Winter profiles follow winter atmospheres; "
            "otherwise spring/summer. Rayleigh is included separately."
        )
    visibility_km = st.sidebar.number_input(
        "Visibility (km)", min_value=0.1, value=23.0, step=1.0, key="visibility_km",
        help="Legacy models use Koschmieder extinction. LOWTRAN tables use their original 550 nm profiles (2–50 km visibility).",
    )
    wind_speed_m_s = st.sidebar.number_input(
        "Wind speed (m/s)", min_value=0.0, value=5.0, step=0.5, key="wind_speed_m_s",
        help="navy_maritime aerosol only.",
    )
    with st.sidebar.expander("Advanced atmosphere overrides", expanded=False):
        ground_temp_k = _optional_number("ground temperature (K)", key="ground_temp_k", default=288.2, min_value=150.0, max_value=340.0)
        ground_rh_pct = _optional_number("ground RH (%)", key="ground_rh_pct", default=50.0, min_value=0.0, max_value=100.0)
        surface_skin_temp_k = _optional_number("surface skin temperature (K)", key="surface_skin_temp_k", default=288.2, min_value=150.0, max_value=340.0)
        surface_emissivity = _optional_number("surface emissivity", key="surface_emissivity", default=0.95, min_value=0.0, max_value=1.0, step=0.01)
        co2_ppmv = _optional_number("CO2 (ppmv)", key="co2_ppmv", default=constants.BACKGROUND_VMR_PPMV["CO2"], min_value=0.0)
        surface_altitude_km = st.number_input(
            "Re-anchor surface altitude (km)", value=0.0, min_value=0.0, step=0.1, key="surface_altitude_km",
            help="Raises the model atmosphere's base level, e.g. for a plateau site.",
        )
    return dict(
        atmosphere=atmosphere, surface_type=surface_type, aerosol=aerosol,
        visibility_km=visibility_km, wind_speed_m_s=wind_speed_m_s,
        ground_temp_k=ground_temp_k, ground_rh_pct=ground_rh_pct,
        surface_skin_temp_k=surface_skin_temp_k, surface_emissivity=surface_emissivity,
        co2_ppmv=co2_ppmv, surface_altitude_km=surface_altitude_km,
    )


def _sensor_controls() -> dict:
    st.sidebar.header("Sensor")
    sensor_alt_km = st.sidebar.number_input("Sensor altitude (km)", value=0.0, min_value=0.0, step=0.1, key="sensor_alt_km")
    sensor_zenith_deg = st.sidebar.number_input(
        "Sensor zenith (deg)", value=0.0, min_value=0.0, max_value=180.0, step=1.0, key="sensor_zenith_deg",
        help="0 = straight up, 90 = horizontal, 180 = straight down.",
    )
    sensor_azimuth_deg = st.sidebar.number_input(
        "Sensor azimuth (deg)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="sensor_azimuth_deg",
        help="Clockwise from true north. Does not affect the (horizontally stratified) geometry, only sun/moon relative angles.",
    )
    latitude_deg = st.sidebar.number_input("Latitude (deg)", value=35.0, min_value=-90.0, max_value=90.0, step=1.0, key="latitude_deg")
    longitude_deg = st.sidebar.number_input("Longitude (deg)", value=0.0, min_value=-180.0, max_value=180.0, step=1.0, key="longitude_deg")
    sensor_aperture_diameter_m = st.sidebar.number_input(
        "Aperture diameter (m)", value=0.10, min_value=0.001, max_value=10.0, step=0.05,
        format="%.3f", key="sensor_aperture_diameter_m",
        help="Clear aperture of the receiving optic. Sets the diffraction limit "
             "and, with a turbulence model selected, the Strehl ratio and "
             "point-source blur. Turbulence only degrades a point source once "
             "this exceeds the Fried parameter r0 (~60 cm at 4 um, zenith).",
    )
    return dict(
        sensor_alt_km=sensor_alt_km, sensor_zenith_deg=sensor_zenith_deg,
        sensor_azimuth_deg=sensor_azimuth_deg, latitude_deg=latitude_deg, longitude_deg=longitude_deg,
        sensor_aperture_diameter_m=sensor_aperture_diameter_m,
    )


def _source_controls() -> dict:
    with st.sidebar.expander("Source / target", expanded=False):
        st.caption("A discrete target along the path (e.g. a hot spot). Leave everything off to integrate to the end of the path (ground or space).")
        source_alt_km = _optional_number("source altitude (km)", key="source_alt_km", default=0.0, min_value=0.0)
        source_range_km = _optional_number("source range (km)", key="source_range_km", default=5.0, min_value=0.01)
        source_temp_k = _optional_number("source temperature (K)", key="source_temp_k", default=650.0, min_value=0.0)
        source_area_m2 = _optional_number("source area (m^2)", key="source_area_m2", default=1.0, min_value=0.0)
        source_emissivity = st.number_input("Source emissivity", value=1.0, min_value=0.0, max_value=1.0, step=0.01, key="source_emissivity")
    return dict(
        source_alt_km=source_alt_km, source_range_km=source_range_km, source_temp_k=source_temp_k,
        source_area_m2=source_area_m2, source_emissivity=source_emissivity,
    )


def _illumination_controls() -> dict:
    with st.sidebar.expander("Illumination", expanded=False):
        use_dt = st.checkbox("Set local date/time", key="use_local_datetime")
        local_datetime = None
        if use_dt:
            d = st.date_input("Local date", value=date(2026, 7, 15), key="local_date")
            t = st.time_input("Local time", value=dtime(13, 30), key="local_time")
            local_datetime = datetime.combine(d, t)
        utc_offset_hours = _optional_number("UTC offset (hours)", key="utc_offset_hours", default=0.0, min_value=-14.0, max_value=14.0)
        st.caption(
            "Without an explicit UTC offset the timezone is guessed from longitude, "
            "which can be an hour or more out (Madrid keeps CET on the Greenwich "
            "meridian) — tens of degrees of solar position error. Supplying observed "
            "sunrise **and** sunset below overrides the timezone entirely and pins "
            "the sun to ~0.1°."
        )
        sunrise_local_hours = _optional_number(
            "sunrise (local clock hours)", key="sunrise_local_hours", default=6.75,
            min_value=0.0, max_value=24.0,
        )
        sunset_local_hours = _optional_number(
            "sunset (local clock hours)", key="sunset_local_hours", default=21.80,
            min_value=0.0, max_value=24.0,
        )
        moon_phase = st.number_input(
            "Moon phase", value=0.5, min_value=0.0, max_value=1.0, step=0.05, key="moon_phase",
            help="0 = new, 0.5 = full, 1 = new again.",
        )
        include_solar = st.checkbox("Include solar illumination", value=True, key="include_solar")
        include_lunar = st.checkbox("Include lunar illumination", value=True, key="include_lunar")

    with st.sidebar.expander("Optical turbulence", expanded=False):
        st.caption(
            "Diagnostic only — turbulence redistributes a point source in angle "
            "and time, it does not absorb, so no radiance value changes."
        )
        turbulence = st.selectbox(
            "Cn² profile model", options=list(TurbulenceModel),
            format_func=lambda m: m.value, index=0, key="turbulence",
        )
        turbulence_cn2_ground = _optional_number(
            "ground Cn² (m^-2/3)", key="turbulence_cn2_ground", default=1.7e-14,
            min_value=0.0, format="%.3e",
        )

    return dict(
        local_datetime=local_datetime, utc_offset_hours=utc_offset_hours, moon_phase=moon_phase,
        include_solar=include_solar, include_lunar=include_lunar,
        sunrise_local_hours=sunrise_local_hours, sunset_local_hours=sunset_local_hours,
        turbulence=turbulence, turbulence_cn2_ground=turbulence_cn2_ground,
    )


def _apply_spectral_preset() -> None:
    """Seed the spectral/accuracy widgets' session state, and overwrite it
    when the user picks a different preset."""
    for k, v in dict(
        wn_min=constants.MWIR_GRID_MIN_CM1, wn_max=constants.MWIR_GRID_MAX_CM1,
        resolution_cm1=4.0,  # coarser than the CLI/Scenario default (1.0) -- keeps the GUI interactive
        line_wing_cutoff_cm1=25.0, line_strength_cutoff=0.0, line_optical_depth_cutoff=3.0e-6,
    ).items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("_last_spectral_preset", None)

    preset = st.sidebar.selectbox("Quick preset", list(_SPECTRAL_PRESETS) + ["Custom"], index=0, key="spectral_preset")
    if preset != "Custom" and st.session_state["_last_spectral_preset"] != preset:
        p = _SPECTRAL_PRESETS[preset]
        st.session_state["wn_min"] = p["wn_min"]
        st.session_state["wn_max"] = p["wn_max"]
        st.session_state["resolution_cm1"] = p["resolution"]
        st.session_state["line_wing_cutoff_cm1"] = p["wing"]
        st.session_state["line_strength_cutoff"] = p["strength"]
        st.session_state["line_optical_depth_cutoff"] = p["tau"]
    st.session_state["_last_spectral_preset"] = None if preset == "Custom" else preset


def _spectral_and_accuracy_controls() -> dict:
    st.sidebar.header("Spectral range & resolution")
    _apply_spectral_preset()

    wn_min = st.sidebar.number_input("Wavenumber min (cm^-1)", min_value=400.0, max_value=8000.0, step=1.0, key="wn_min")
    wn_max = st.sidebar.number_input("Wavenumber max (cm^-1)", min_value=400.0, max_value=8000.0, step=1.0, key="wn_max")
    if wn_max > wn_min:
        st.sidebar.caption(f"-> {1.0e4 / wn_max:.4f} - {1.0e4 / wn_min:.4f} um")
    resolution_cm1 = st.sidebar.number_input(
        "Resolution (cm^-1)", min_value=1.0e-4, max_value=50.0, step=0.1, format="%.4f", key="resolution_cm1",
        help="Never capped -- set it as fine as your data needs. Cost scales with points x layers; "
             "see the cost estimate below before running a wide window at fine resolution.",
    )

    sampling = _enum_select("Spectral sampling", _SAMPLING_LABELS, SpectralSampling.RESOLVED,
                            key="spectral_sampling")
    resolved_fields: dict = {}
    if sampling == SpectralSampling.RESOLVED:
        resolved_fields["instrument_function"] = _enum_select(
            "Instrument function", {f: f.value.title() for f in InstrumentFunction},
            InstrumentFunction.BOX, key="instrument_function")
        fwhm = st.sidebar.number_input(
            "Instrument FWHM (cm^-1, 0 = resolution)", min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.4f", key="instrument_fwhm_cm1")
        fine = st.sidebar.number_input(
            "Fine step (cm^-1, 0 = automatic)", min_value=0.0, max_value=1.0, value=0.0,
            step=0.001, format="%.4f", key="fine_step_cm1")
        resolved_fields["instrument_fwhm_cm1"] = fwhm if fwhm > 0 else None
        resolved_fields["fine_step_cm1"] = fine if fine > 0 else None
        st.sidebar.caption(
            "The physics runs on a grid fine enough to resolve absorption lines, then is "
            "averaged through the instrument function onto the output grid. Cost follows the "
            "bandwidth and the fine step, not the output resolution.")

    st.sidebar.subheader("Accuracy")
    # value= is deliberately omitted on the widgets below: their keys are
    # pre-seeded into session_state by _apply_spectral_preset, and Streamlit
    # warns when a widget gets both a value= and a session_state-backed key.
    line_optical_depth_cutoff = st.sidebar.number_input(
        "Line optical-depth cutoff", min_value=0.0, max_value=1.0, step=1.0e-6, format="%.2e",
        key="line_optical_depth_cutoff",
        help="Drop a line from a layer when its peak optical depth there is below this. "
             "Tied to the actual gas amount, so humid layers keep the weak water lines that "
             "matter in the windows. 0.0 keeps every line.",
    )
    line_strength_cutoff = st.sidebar.number_input(
        "Legacy line strength cutoff", min_value=0.0, max_value=1.0, step=1.0e-5, format="%.2e",
        key="line_strength_cutoff",
        help="Drop lines weaker than this fraction of each species' strongest line. Off (0.0) "
             "by default: the former default of 1e-4 discarded every water line in the "
             "3.5-4.1 um window.",
    )
    line_wing_cutoff_cm1 = st.sidebar.number_input(
        "Line wing cutoff (cm^-1)", min_value=0.1, max_value=200.0, step=0.5, key="line_wing_cutoff_cm1",
        help="Local-line half-width for gases other than water. H2O with MT_CKD always uses "
             "25 cm^-1 with pedestal subtraction. Smaller cutoffs can omit absorption; "
             "check convergence. Evaluation windows are clipped to the requested grid.",
    )
    return dict(
        wavenumber_min_cm1=wn_min, wavenumber_max_cm1=wn_max, resolution_cm1=resolution_cm1,
        line_strength_cutoff=line_strength_cutoff, line_wing_cutoff_cm1=line_wing_cutoff_cm1,
        line_optical_depth_cutoff=line_optical_depth_cutoff, spectral_sampling=sampling,
        **resolved_fields,
    )


def _solver_controls() -> dict:
    solver = _enum_select("Radiance solver", _SOLVER_LABELS, RadianceSolver.SINGLE_SCATTER,
                          key="radiance_solver")
    fields: dict = {"radiance_solver": solver}
    if solver == RadianceSolver.DISORT:
        fields["disort_streams"] = st.sidebar.selectbox("DISORT streams", [8, 16, 32, 64], index=2, key="disort_streams")
        st.sidebar.caption("Solves the full vertical column and repeated surface reflection. "
                           "Plane-parallel geometry replaces spherical paths. Horizontal views need a finite target range. "
                           "Increase streams to check angular convergence; near-horizon curvature is not modeled.")
    if solver == RadianceSolver.SPHERICAL_MC:
        fields["spherical_photons"] = st.sidebar.number_input("Photon histories per spectral point", min_value=128, max_value=1000000, value=4096, step=1024)
        fields["spherical_seed"] = st.sidebar.number_input("Monte Carlo seed", min_value=0, max_value=2**32-1, value=0)
        st.sidebar.caption("Experimental solver: curved atmospheric shells for every photon path; scalar, no refraction. Sampling noise decreases approximately as 1/sqrt(histories). Start with a narrow spectrum.")
    return fields
