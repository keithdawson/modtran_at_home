"""GUI sidebar: widgets -> :class:`~mwir_atmos.types.Scenario`.

The sidebar groups the same fields as the CLI's flags (atmosphere / sensor /
source / illumination / spectral / accuracy). Scenario-affecting controls use
``number_input``/``selectbox``/``checkbox`` rather than continuous-drag
sliders, so an in-progress drag does not fire a cascade of intermediate
(uncached) physics runs -- only a committed value does.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime
from typing import Any, Optional

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

__all__ = ["AEROSOL_LABELS", "MODEL_LABELS", "backend_from_sidebar", "build_scenario_from_sidebar", "populate_controls_from_scenario"]

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


_BACKEND_LABELS = {
    "auto": "Auto (GPU if available)",
    "cpu": "CPU",
    "cuda": "GPU (CUDA)",
}


def backend_from_sidebar() -> str:
    """The compute backend the "Performance" controls select (a run option, not a Scenario field)."""
    from mwir_atmos.optics import lbl, lbl_cuda
    with st.sidebar.expander("Performance", expanded=False):
        choice = st.selectbox("Compute backend", list(_BACKEND_LABELS), index=0,
                              format_func=_BACKEND_LABELS.get, key="compute_backend",
                              help="Results agree to ~1e-6 between backends. The GPU needs "
                                   "CuPy and an NVIDIA GPU (pip install mwir_atmos[cuda]).")
        gpu = lbl_cuda.available()
        if gpu:
            st.caption(f"GPU: {lbl_cuda.device_summary()}")
        else:
            st.caption("No CUDA GPU found: Auto runs on the CPU.")
        if choice == "cuda" and not gpu:
            st.error("GPU (CUDA) selected, but CuPy or a CUDA device is not available.")
        try:
            resolved = lbl.resolve_backend(choice)
        except RuntimeError:
            resolved = "cpu"
        if choice == "auto":
            st.caption(f"Running on: {_BACKEND_LABELS[resolved]}")
    return choice


def populate_controls_from_scenario(sc: Scenario) -> None:
    """Drop all values from a Scenario directly into st.session_state so dropdowns reflect them."""
    def _set(k: str, v: Any) -> None:
        st.session_state[k] = v
        st.session_state[f"_saved_{k}"] = v

    # Atmosphere
    _set("atmosphere", sc.atmosphere)
    _set("surface_type", sc.surface_type)
    _set("aerosol", sc.aerosol)
    _set("visibility_km", float(sc.visibility_km))
    _set("wind_speed_m_s", float(sc.wind_speed_m_s))
    _set("surface_altitude_km", float(sc.surface_altitude_km))

    # Optional atmosphere overrides
    _set("ground_temp_k__on", sc.ground_temp_k is not None)
    if sc.ground_temp_k is not None:
        _set("ground_temp_k", float(sc.ground_temp_k))

    _set("ground_rh_pct__on", sc.ground_rh_pct is not None)
    if sc.ground_rh_pct is not None:
        _set("ground_rh_pct", float(sc.ground_rh_pct))

    _set("surface_skin_temp_k__on", sc.surface_skin_temp_k is not None)
    if sc.surface_skin_temp_k is not None:
        _set("surface_skin_temp_k", float(sc.surface_skin_temp_k))

    _set("surface_emissivity__on", sc.surface_emissivity is not None)
    if sc.surface_emissivity is not None:
        _set("surface_emissivity", float(sc.surface_emissivity))

    _set("co2_ppmv__on", sc.co2_ppmv is not None)
    if sc.co2_ppmv is not None:
        _set("co2_ppmv", float(sc.co2_ppmv))

    # Sensor
    _set("sensor_alt_km", float(sc.sensor_alt_km))
    _set("sensor_zenith_deg", float(sc.sensor_zenith_deg))
    _set("sensor_azimuth_deg", float(sc.sensor_azimuth_deg))
    _set("latitude_deg", float(sc.latitude_deg))
    _set("longitude_deg", float(sc.longitude_deg))
    _set("sensor_aperture_diameter_m", float(sc.sensor_aperture_diameter_m))

    # Source / Target
    _set("source_alt_km__on", sc.source_alt_km is not None)
    if sc.source_alt_km is not None:
        _set("source_alt_km", float(sc.source_alt_km))

    _set("source_range_km__on", sc.source_range_km is not None)
    if sc.source_range_km is not None:
        _set("source_range_km", float(sc.source_range_km))

    _set("source_temp_k__on", sc.source_temp_k is not None)
    if sc.source_temp_k is not None:
        _set("source_temp_k", float(sc.source_temp_k))

    _set("source_area_m2__on", sc.source_area_m2 is not None)
    if sc.source_area_m2 is not None:
        _set("source_area_m2", float(sc.source_area_m2))

    _set("source_emissivity", float(sc.source_emissivity))

    # Illumination / Time
    _set("use_local_datetime", sc.local_datetime is not None)
    if sc.local_datetime is not None:
        _set("local_date", sc.local_datetime.date())
        _set("local_time", sc.local_datetime.time())

    _set("utc_offset_hours__on", sc.utc_offset_hours is not None)
    if sc.utc_offset_hours is not None:
        _set("utc_offset_hours", float(sc.utc_offset_hours))

    _set("sunrise_local_hours__on", sc.sunrise_local_hours is not None)
    if sc.sunrise_local_hours is not None:
        _set("sunrise_local_hours", float(sc.sunrise_local_hours))

    _set("sunset_local_hours__on", sc.sunset_local_hours is not None)
    if sc.sunset_local_hours is not None:
        _set("sunset_local_hours", float(sc.sunset_local_hours))

    _set("moon_phase", float(sc.moon_phase))
    _set("include_solar", bool(sc.include_solar))
    _set("include_lunar", bool(sc.include_lunar))

    # Turbulence
    _set("turbulence", sc.turbulence)
    _set("turbulence_cn2_ground__on", sc.turbulence_cn2_ground is not None)
    if sc.turbulence_cn2_ground is not None:
        _set("turbulence_cn2_ground", float(sc.turbulence_cn2_ground))

    # Spectral & Accuracy
    _set("spectral_preset", "Custom")
    _set("_last_spectral_preset", "Custom")
    _set("wn_min", float(sc.wavenumber_min_cm1))
    _set("wn_max", float(sc.wavenumber_max_cm1))
    _set("_saved_wn_min", float(sc.wavenumber_min_cm1))
    _set("_saved_wn_max", float(sc.wavenumber_max_cm1))

    wl_min_val = round(10000.0 / float(sc.wavenumber_max_cm1), 3)
    wl_max_val = round(10000.0 / float(sc.wavenumber_min_cm1), 3)
    _set("wl_min", wl_min_val)
    _set("wl_max", wl_max_val)
    _set("_saved_wl_min", wl_min_val)
    _set("_saved_wl_max", wl_max_val)

    _set("resolution_cm1", float(sc.resolution_cm1))
    _set("_saved_resolution_cm1", float(sc.resolution_cm1))
    _set("spectral_sampling", sc.spectral_sampling)

    if sc.spectral_sampling == SpectralSampling.RESOLVED:
        _set("instrument_function", sc.instrument_function)
        _set("instrument_fwhm_cm1", float(sc.instrument_fwhm_cm1 or 0.0))
        _set("fine_step_cm1", float(sc.fine_step_cm1 or 0.0))

    _set("line_optical_depth_cutoff", float(sc.line_optical_depth_cutoff))
    _set("line_strength_cutoff", float(sc.line_strength_cutoff))
    _set("line_wing_cutoff_cm1", float(sc.line_wing_cutoff_cm1))

    # Solver
    _set("radiance_solver", sc.radiance_solver)
    if sc.disort_streams in (8, 16, 32, 64):
        _set("disort_streams", int(sc.disort_streams))


def build_scenario_from_sidebar() -> Scenario:
    """The scenario the sidebar describes: from the controls, or a loaded YAML setup."""
    st.sidebar.title("mwir_atmos")
    source = st.sidebar.radio("Setup source", ["Current controls", "Saved YAML"], key="setup_source")

    if source == "Saved YAML":
        _scenario_from_uploaded_yaml()

    # The controls below hold the active values (either configured manually or populated from YAML)
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

def _checkbox(label: str, *, key: str, default: bool = False, container=None, **kwargs) -> bool:
    target = container if container is not None else st.sidebar
    saved = st.session_state.get(f"_saved_{key}")
    if key not in st.session_state and saved is not None:
        st.session_state[key] = saved
    if key in st.session_state:
        val = bool(target.checkbox(label, key=key, **kwargs))
    else:
        val = bool(target.checkbox(label, value=default, key=key, **kwargs))
    st.session_state[f"_saved_{key}"] = val
    return val


def _number_input(label: str, *, key: str, default: float, container=None, **kwargs) -> float:
    target = container if container is not None else st.sidebar
    saved = st.session_state.get(f"_saved_{key}")
    if key not in st.session_state and saved is not None:
        st.session_state[key] = saved
    if key in st.session_state:
        val = float(target.number_input(label, key=key, **kwargs))
    else:
        val = float(target.number_input(label, value=default, key=key, **kwargs))
    st.session_state[f"_saved_{key}"] = val
    return val


def _optional_number(label: str, *, key: str, default: float, help: Optional[str] = None,
                     container=None, **kwargs) -> Optional[float]:
    """A checkbox-gated number_input: unchecked -> None (keep the model/scenario
    default), matching the CLI's "flag omitted -> None" semantics exactly."""
    target = container if container is not None else st.sidebar
    on_key = f"{key}__on"
    if on_key not in st.session_state and f"_saved_{on_key}" in st.session_state:
        st.session_state[on_key] = st.session_state[f"_saved_{on_key}"]
    enabled = target.checkbox(f"Override {label}", key=on_key)
    st.session_state[f"_saved_{on_key}"] = enabled
    if not enabled:
        return None
    if key not in st.session_state and f"_saved_{key}" in st.session_state:
        st.session_state[key] = st.session_state[f"_saved_{key}"]
    if key in st.session_state:
        val = float(target.number_input(label, key=key, help=help, **kwargs))
    else:
        val = float(target.number_input(label, value=default, key=key, help=help, **kwargs))
    st.session_state[f"_saved_{key}"] = val
    return val


def _enum_select(label: str, options: dict, current, *, key: str, container=None):
    target = container if container is not None else st.sidebar
    keys = list(options.keys())
    saved = st.session_state.get(f"_saved_{key}")
    if key not in st.session_state and saved in keys:
        st.session_state[key] = saved
    if key in st.session_state and st.session_state[key] in keys:
        val = target.selectbox(label, keys, format_func=lambda k: options[k], key=key)
    else:
        idx = keys.index(current) if current in keys else 0
        val = target.selectbox(label, keys, index=idx, format_func=lambda k: options[k], key=key)
    st.session_state[f"_saved_{key}"] = val
    return val


# =========================================================================
# Control groups, each returning the Scenario fields it sets
# =========================================================================


def _scenario_from_uploaded_yaml() -> None:
    uploaded = st.sidebar.file_uploader("Open saved setup", type=["yaml", "yml"], key="setup_upload")
    st.sidebar.caption("Loads setup parameters directly into the control dropdowns below.")

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        if st.session_state.get("_last_loaded_yaml_bytes") != raw_bytes:
            try:
                data = yaml.safe_load(raw_bytes.decode("utf-8-sig"))
                if not isinstance(data, dict):
                    raise ValueError("Setup must be a YAML mapping")
                sc = scenario_from_dict(data)
                populate_controls_from_scenario(sc)
                st.session_state["_last_loaded_yaml_bytes"] = raw_bytes
                st.session_state["current_result"] = None
                st.session_state["current_result_key"] = None
                st.sidebar.success(f"Loaded '{uploaded.name}' into dropdowns below!")
            except Exception as exc:
                st.sidebar.error(f"Could not load saved setup: {exc}")

        if st.sidebar.button("Re-load YAML into dropdowns", key="btn_reload_yaml"):
            try:
                data = yaml.safe_load(uploaded.getvalue().decode("utf-8-sig"))
                sc = scenario_from_dict(data)
                populate_controls_from_scenario(sc)
                st.session_state["current_result"] = None
                st.session_state["current_result_key"] = None
                st.sidebar.success(f"Re-loaded '{uploaded.name}' into dropdowns!")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Could not load saved setup: {exc}")
    else:
        st.sidebar.info("Upload a YAML setup above to populate the controls below.")


def _atmosphere_controls() -> dict:
    st.sidebar.header("Atmosphere")
    c1, c2 = st.sidebar.columns(2)
    atmosphere = _enum_select("Model atmosphere", MODEL_LABELS, ModelAtmosphere.MIDLAT_SUMMER, key="atmosphere", container=c1)
    surface_type = _enum_select("Surface type", SURFACE_LABELS, SurfaceType.LAND_VEGETATED, key="surface_type", container=c2)

    c1, c2 = st.sidebar.columns(2)
    aerosol = _enum_select("Aerosol model", AEROSOL_LABELS, AerosolModel.RURAL, key="aerosol", container=c1)
    visibility_km = _number_input(
        "Visibility (km)", min_value=0.0, default=23.0, step=1.0, format="%.1f", key="visibility_km",
        help="Legacy models use Koschmieder extinction. LOWTRAN tables use their original 550 nm profiles (2–50 km visibility).",
        container=c2,
    )
    if aerosol.value.startswith("lowtran_"):
        st.sidebar.caption(
            "LOWTRAN 7 optical tables, RH 0–99%, background altitude profiles. "
            "Visibility: 2–50 km. Winter profiles follow winter atmospheres; "
            "otherwise spring/summer. Rayleigh is included separately."
        )
    c1, c2 = st.sidebar.columns(2)
    wind_speed_m_s = _number_input(
        "Wind speed (m/s)", min_value=0.0, default=5.0, step=0.5, format="%.1f", key="wind_speed_m_s",
        help="navy_maritime aerosol only.", container=c1,
    )
    surface_altitude_km = _number_input(
        "Surface altitude (km)", default=0.0, min_value=0.0, step=0.1, format="%.2f", key="surface_altitude_km",
        help="Raises the model atmosphere's base level, e.g. for a plateau site.", container=c2,
    )
    exp = st.sidebar.expander("Advanced atmosphere overrides", expanded=False)
    with exp:
        c1, c2 = exp.columns(2)
        ground_temp_k = _optional_number("ground temperature (K)", key="ground_temp_k", default=288.2, min_value=150.0, max_value=340.0, step=0.01, format="%.2f", container=c1)
        ground_rh_pct = _optional_number("ground RH (%)", key="ground_rh_pct", default=50.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", container=c2)
        c1, c2 = exp.columns(2)
        surface_skin_temp_k = _optional_number("surface skin temperature (K)", key="surface_skin_temp_k", default=288.2, min_value=150.0, max_value=340.0, step=0.01, format="%.2f", container=c1)
        surface_emissivity = _optional_number("surface emissivity", key="surface_emissivity", default=0.95, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", container=c2)
        co2_ppmv = _optional_number("CO2 (ppmv)", key="co2_ppmv", default=constants.BACKGROUND_VMR_PPMV["CO2"], min_value=0.0, step=1.0, format="%.1f", container=exp)
    return dict(
        atmosphere=atmosphere, surface_type=surface_type, aerosol=aerosol,
        visibility_km=visibility_km, wind_speed_m_s=wind_speed_m_s,
        ground_temp_k=ground_temp_k, ground_rh_pct=ground_rh_pct,
        surface_skin_temp_k=surface_skin_temp_k, surface_emissivity=surface_emissivity,
        co2_ppmv=co2_ppmv, surface_altitude_km=surface_altitude_km,
    )


def _sensor_controls() -> dict:
    st.sidebar.header("Sensor")
    c1, c2 = st.sidebar.columns(2)
    sensor_alt_km = _number_input("Sensor altitude (km)", default=0.0, min_value=0.0, step=0.01, format="%.3f", key="sensor_alt_km", container=c1)
    sensor_zenith_deg = _number_input(
        "Sensor zenith (deg)", default=0.0, min_value=0.0, max_value=180.0, step=0.1, format="%.1f", key="sensor_zenith_deg",
        help="0 = straight up, 90 = horizontal, 180 = straight down.", container=c2,
    )
    c1, c2 = st.sidebar.columns(2)
    sensor_azimuth_deg = _number_input(
        "Sensor azimuth (deg)", default=0.0, min_value=0.0, max_value=360.0, step=0.5, format="%.1f", key="sensor_azimuth_deg",
        help="Clockwise from true north.", container=c1,
    )
    sensor_aperture_diameter_m = _number_input(
        "Aperture diameter (m)", default=0.10, min_value=0.0, max_value=10.0, step=0.01,
        format="%.3f", key="sensor_aperture_diameter_m",
        help="Clear aperture of the receiving optic.", container=c2,
    )
    c1, c2 = st.sidebar.columns(2)
    latitude_deg = _number_input("Latitude (deg)", default=35.0, min_value=-90.0, max_value=90.0, step=0.01, format="%.2f", key="latitude_deg", container=c1)
    longitude_deg = _number_input("Longitude (deg)", default=0.0, min_value=-180.0, max_value=180.0, step=0.01, format="%.2f", key="longitude_deg", container=c2)
    return dict(
        sensor_alt_km=sensor_alt_km, sensor_zenith_deg=sensor_zenith_deg,
        sensor_azimuth_deg=sensor_azimuth_deg, latitude_deg=latitude_deg, longitude_deg=longitude_deg,
        sensor_aperture_diameter_m=sensor_aperture_diameter_m,
    )


def _source_controls() -> dict:
    exp = st.sidebar.expander("Source / target", expanded=False)
    with exp:
        exp.caption("A discrete target along the path (e.g. a hot spot). Leave everything off to integrate to the end of the path (ground or space).")
        c1, c2 = exp.columns(2)
        source_alt_km = _optional_number("source altitude (km)", key="source_alt_km", default=0.0, min_value=0.0, step=0.01, format="%.3f", container=c1)
        source_range_km = _optional_number("source range (km)", key="source_range_km", default=5.0, min_value=0.0, step=0.1, format="%.2f", container=c2)
        c1, c2 = exp.columns(2)
        source_temp_k = _optional_number("source temperature (K)", key="source_temp_k", default=650.0, min_value=0.0, step=1.0, format="%.1f", container=c1)
        source_area_m2 = _optional_number("source area (m^2)", key="source_area_m2", default=1.0, min_value=0.0, step=0.01, format="%.2f", container=c2)
        source_emissivity = _number_input("Source emissivity", default=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="source_emissivity", container=exp)
    return dict(
        source_alt_km=source_alt_km, source_range_km=source_range_km, source_temp_k=source_temp_k,
        source_area_m2=source_area_m2, source_emissivity=source_emissivity,
    )


def _illumination_controls() -> dict:
    exp = st.sidebar.expander("Illumination", expanded=False)
    with exp:
        use_dt = _checkbox("Set local date/time", default=False, key="use_local_datetime", container=exp)
        local_datetime = None
        if use_dt:
            c1, c2 = exp.columns(2)
            if "local_date" in st.session_state:
                d = c1.date_input("Local date", key="local_date")
            else:
                d = c1.date_input("Local date", value=date(2026, 7, 15), key="local_date")

            if "local_time" in st.session_state:
                t = c2.time_input("Local time", key="local_time")
            else:
                t = c2.time_input("Local time", value=dtime(13, 30), key="local_time")
            local_datetime = datetime.combine(d, t)
        c1, c2 = exp.columns(2)
        utc_offset_hours = _optional_number("UTC offset (hours)", key="utc_offset_hours", default=0.0, min_value=-14.0, max_value=14.0, step=0.5, format="%.1f", container=c1)
        moon_phase = _number_input(
            "Moon phase", default=0.5, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="moon_phase",
            help="0 = new, 0.5 = full, 1 = new again.", container=c2,
        )
        c1, c2 = exp.columns(2)
        sunrise_local_hours = _optional_number(
            "sunrise (local clock hours)", key="sunrise_local_hours", default=6.75,
            min_value=0.0, max_value=24.0, step=0.25, format="%.2f", container=c1,
        )
        sunset_local_hours = _optional_number(
            "sunset (local clock hours)", key="sunset_local_hours", default=21.80,
            min_value=0.0, max_value=24.0, step=0.1, format="%.2f", container=c2,
        )
        c1, c2 = exp.columns(2)
        include_solar = _checkbox("Include solar illumination", default=True, key="include_solar", container=c1)
        include_lunar = _checkbox("Include lunar illumination", default=True, key="include_lunar", container=c2)

    turb_exp = st.sidebar.expander("Optical turbulence", expanded=False)
    with turb_exp:
        turb_exp.caption(
            "Diagnostic only — turbulence redistributes a point source in angle "
            "and time, it does not absorb, so no radiance value changes."
        )
        c1, c2 = turb_exp.columns(2)
        turbulence = c1.selectbox(
            "Cn² profile model", options=list(TurbulenceModel),
            format_func=lambda m: m.value, index=0, key="turbulence",
        )
        turbulence_cn2_ground = _optional_number(
            "ground Cn² (m^-2/3)", key="turbulence_cn2_ground", default=1.7e-14,
            min_value=0.0, step=1.0e-15, format="%.3e", container=c2,
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
        spectral_unit="Wavelength (um)",
        wn_min=constants.MWIR_GRID_MIN_CM1, wn_max=constants.MWIR_GRID_MAX_CM1,
        wl_min=round(10000.0 / constants.MWIR_GRID_MAX_CM1, 3),
        wl_max=round(10000.0 / constants.MWIR_GRID_MIN_CM1, 3),
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
        st.session_state["_saved_wn_min"] = p["wn_min"]
        st.session_state["_saved_wn_max"] = p["wn_max"]

        prec = 4 if p["wn_max"] - p["wn_min"] < 10 else 3
        wl_min_val = round(10000.0 / p["wn_max"], prec)
        wl_max_val = round(10000.0 / p["wn_min"], prec)
        st.session_state["wl_min"] = wl_min_val
        st.session_state["wl_max"] = wl_max_val
        st.session_state["_saved_wl_min"] = wl_min_val
        st.session_state["_saved_wl_max"] = wl_max_val

        st.session_state["resolution_cm1"] = p["resolution"]
        st.session_state["_saved_resolution_cm1"] = p["resolution"]
        st.session_state["line_wing_cutoff_cm1"] = p["wing"]
        st.session_state["_saved_line_wing_cutoff_cm1"] = p["wing"]
        st.session_state["line_strength_cutoff"] = p["strength"]
        st.session_state["_saved_line_strength_cutoff"] = p["strength"]
        st.session_state["line_optical_depth_cutoff"] = p["tau"]
        st.session_state["_saved_line_optical_depth_cutoff"] = p["tau"]
    st.session_state["_last_spectral_preset"] = None if preset == "Custom" else preset


def _spectral_and_accuracy_controls() -> dict:
    st.sidebar.header("Spectral range & resolution")
    _apply_spectral_preset()

    unit = st.sidebar.radio(
        "Spectral units", ["Wavelength (um)", "Wavenumber (cm^-1)"],
        index=0, horizontal=True, key="spectral_unit"
    )

    if unit == "Wavelength (um)":
        c1, c2 = st.sidebar.columns(2)
        default_wl_min = round(10000.0 / float(st.session_state.get("wn_max", constants.MWIR_GRID_MAX_CM1)), 3)
        default_wl_max = round(10000.0 / float(st.session_state.get("wn_min", constants.MWIR_GRID_MIN_CM1)), 3)
        wl_min = _number_input(
            "Wavelength min (um)", min_value=0.0, max_value=25.0, step=0.001, format="%.3f",
            default=default_wl_min, key="wl_min", container=c1
        )
        wl_max = _number_input(
            "Wavelength max (um)", min_value=0.0, max_value=25.0, step=0.001, format="%.3f",
            default=default_wl_max, key="wl_max", container=c2
        )
        if wl_max > wl_min and wl_min > 0:
            if abs(wl_min - 2.326) < 1e-4 and abs(wl_max - 5.556) < 1e-4:
                wn_min = constants.MWIR_GRID_MIN_CM1
                wn_max = constants.MWIR_GRID_MAX_CM1
            elif abs(wl_min - 3.000) < 1e-4 and abs(wl_max - 5.000) < 1e-4:
                wn_min = 2000.0
                wn_max = 3333.33
            else:
                wn_min = round(10000.0 / wl_max, 2)
                wn_max = round(10000.0 / wl_min, 2)
            st.session_state["wn_min"] = wn_min
            st.session_state["_saved_wn_min"] = wn_min
            st.session_state["wn_max"] = wn_max
            st.session_state["_saved_wn_max"] = wn_max
            st.sidebar.caption(f"-> {wn_min:.2f} - {wn_max:.2f} cm^-1")
        else:
            wn_min = float(st.session_state.get("wn_min", constants.MWIR_GRID_MIN_CM1))
            wn_max = float(st.session_state.get("wn_max", constants.MWIR_GRID_MAX_CM1))
            if wl_min > 0 and wl_max > 0:
                wn_min = round(10000.0 / wl_max, 2)
                wn_max = round(10000.0 / wl_min, 2)
                st.session_state["wn_min"] = wn_min
                st.session_state["_saved_wn_min"] = wn_min
                st.session_state["wn_max"] = wn_max
                st.session_state["_saved_wn_max"] = wn_max
    else:
        c1, c2 = st.sidebar.columns(2)
        wn_min = _number_input(
            "Wavenumber min (cm^-1)", min_value=400.0, max_value=8000.0, step=0.1, format="%.2f",
            default=constants.MWIR_GRID_MIN_CM1, key="wn_min", container=c1
        )
        wn_max = _number_input(
            "Wavenumber max (cm^-1)", min_value=400.0, max_value=8000.0, step=0.01, format="%.2f",
            default=constants.MWIR_GRID_MAX_CM1, key="wn_max", container=c2
        )
        if wn_max > wn_min and wn_min > 0:
            prec = 4 if wn_max - wn_min < 10 else 3
            wl_min_val = round(10000.0 / wn_max, prec)
            wl_max_val = round(10000.0 / wn_min, prec)
            st.session_state["wl_min"] = wl_min_val
            st.session_state["_saved_wl_min"] = wl_min_val
            st.session_state["wl_max"] = wl_max_val
            st.session_state["_saved_wl_max"] = wl_max_val
            st.sidebar.caption(f"-> {1.0e4 / wn_max:.4f} - {1.0e4 / wn_min:.4f} um")


    c1, c2 = st.sidebar.columns(2)
    resolution_cm1 = _number_input(
        "Resolution (cm^-1)", min_value=0.0, max_value=50.0, step=0.1, format="%.4f", default=4.0, key="resolution_cm1",
        help="Never capped -- set it as fine as your data needs.", container=c1,
    )
    sampling = _enum_select("Spectral sampling", _SAMPLING_LABELS, SpectralSampling.RESOLVED,
                            key="spectral_sampling", container=c2)

    resolved_fields: dict = {}
    if sampling == SpectralSampling.RESOLVED:
        c1, c2 = st.sidebar.columns(2)
        resolved_fields["instrument_function"] = _enum_select(
            "Instrument function", {f: f.value.title() for f in InstrumentFunction},
            InstrumentFunction.BOX, key="instrument_function", container=c1)
        fwhm = _number_input(
            "Instrument FWHM (cm^-1)", min_value=0.0, max_value=100.0, default=0.0,
            step=0.1, format="%.4f", key="instrument_fwhm_cm1", container=c2)
        fine = _number_input(
            "Fine step (cm^-1, 0 = auto)", min_value=0.0, max_value=1.0, default=0.0,
            step=0.001, format="%.4f", key="fine_step_cm1")
        resolved_fields["instrument_fwhm_cm1"] = fwhm if fwhm > 0 else None
        resolved_fields["fine_step_cm1"] = fine if fine > 0 else None
        st.sidebar.caption(
            "The physics runs on a grid fine enough to resolve absorption lines, then is "
            "averaged through the instrument function onto the output grid.")

    st.sidebar.subheader("Accuracy")
    c1, c2 = st.sidebar.columns(2)
    line_optical_depth_cutoff = _number_input(
        "Optical-depth cutoff", min_value=0.0, max_value=1.0, step=1.0e-6, format="%.2e",
        default=3.0e-6, key="line_optical_depth_cutoff", container=c1,
        help="Drop a line from a layer when its peak optical depth there is below this.",
    )
    line_wing_cutoff_cm1 = _number_input(
        "Line wing cutoff (cm^-1)", min_value=0.0, max_value=200.0, step=0.5, format="%.1f", default=25.0,
        key="line_wing_cutoff_cm1", container=c2,
    )
    line_strength_cutoff = _number_input(
        "Legacy strength cutoff", min_value=0.0, max_value=1.0, step=1.0e-5, format="%.2e",
        default=0.0, key="line_strength_cutoff",
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
        streams = [8, 16, 32, 64]
        cur_streams = st.session_state.get("disort_streams", 32)
        idx = streams.index(cur_streams) if cur_streams in streams else 2
        fields["disort_streams"] = st.sidebar.selectbox("DISORT streams", streams, index=idx, key="disort_streams")
        st.sidebar.caption("Solves the full vertical column and repeated surface reflection. "
                           "Plane-parallel geometry replaces spherical paths. Horizontal views need a finite target range. "
                           "Increase streams to check angular convergence; near-horizon curvature is not modeled.")
    if solver == RadianceSolver.SPHERICAL_MC:
        fields["spherical_photons"] = st.sidebar.number_input("Photon histories per spectral point", min_value=0, max_value=1000000, value=4096, step=1024, key="spherical_photons")
        fields["spherical_seed"] = st.sidebar.number_input("Monte Carlo seed", min_value=0, max_value=2**32-1, value=0, step=1, key="spherical_seed")
        st.sidebar.caption("Experimental solver: curved atmospheric shells for every photon path; scalar, no refraction. Sampling noise decreases approximately as 1/sqrt(histories). Start with a narrow spectrum.")
    return fields
