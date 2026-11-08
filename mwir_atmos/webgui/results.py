"""GUI main panel: render a :class:`~mwir_atmos.types.ScenarioResult`.

Widgets here are display-only (axis units, which components to plot, which
bands to tabulate) and never change the scenario, so toggling them reruns the
script but hits the result cache instead of recomputing.
"""

from __future__ import annotations

import io
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from mwir_atmos import constants, report
from mwir_atmos.scenario_io import scenario_to_dict
from mwir_atmos.source_attenuation import (
    SourceIntensityRemaining,
    compute_band_source_remaining,
    compute_source_intensity_remaining,
)
from mwir_atmos.spectral import radiance_to_per_um
from mwir_atmos.types import (
    RADIANCE_COMPONENTS,
    REQUIRED_SPECIES,
    RadianceSolver,
    Scenario,
    ScenarioResult,
    Species,
)
from mwir_atmos.webgui import plots
from mwir_atmos.webgui.sidebar import AEROSOL_LABELS, MODEL_LABELS

__all__ = ["render_result", "setup_download", "download_row"]


def render_result(result: ScenarioResult, run_fn: Optional[Callable[[Scenario], ScenarioResult]] = None) -> None:
    """Every section of the main panel for one computed scenario."""
    attn: Optional[SourceIntensityRemaining] = None
    if result.scenario.source_range_km is not None and result.scenario.source_range_km > 0.0:
        try:
            attn = compute_source_intensity_remaining(result.scenario, result, run_fn=run_fn)
        except Exception as exc:
            st.warning(f"Could not compute near-zero reference comparison: {exc}")

    _scenario_summary(result, attn)
    if attn is not None:
        _source_intensity_card(attn, result.scenario)

    use_wavelength = st.radio(
        "X-axis units", ["Wavelength (um)", "Wavenumber (cm^-1)"], index=0, horizontal=True, key="x_axis_units"
    ) == "Wavelength (um)"
    x, x_label = _x_axis(result.grid, use_wavelength)

    full_csv = io.StringIO()
    report.write_csv(result, full_csv)
    st.download_button(
        "Download full spectral data (CSV)", data=full_csv.getvalue(),
        file_name="mwir_spectrum.csv", mime="text/csv", key="download_full_csv",
    )

    _transmittance_section(result, x, x_label)
    _species_section(result, x, x_label)
    _radiance_section(result, x, x_label)
    _turbulence_section(result)
    _profile_section(result)
    _band_table_section(result, run_fn=run_fn)


def setup_download(scenario: Scenario) -> None:
    with st.expander("Save setup / use from a script"):
        st.caption("Saves every active simulation input. Load this YAML in the sidebar, CLI or Python; override just the fields you want to vary.")
        st.download_button("Download setup (YAML)",
                           data=yaml.safe_dump(scenario_to_dict(scenario), sort_keys=False),
                           file_name="mwir_setup.yaml", mime="application/yaml", key="download_setup")
        st.code('from mwir_atmos.setups import load_setup, with_overrides\n'
                'from mwir_atmos.api import run_scenario\n\n'
                'base = load_setup("mwir_setup.yaml")\n'
                'trial = with_overrides(base, source_range_km=5.0, source_alt_km=None)\n'
                'result = run_scenario(trial)', language="python")
        st.caption("For repeated runs with a custom detectability score, see the distance-sweep example in README.md.")


def download_row(name: str, png_bytes: Optional[bytes]) -> None:
    if png_bytes is not None:
        st.download_button(
            "Download chart as PNG", data=png_bytes, file_name=f"mwir_{name}.png", mime="image/png",
            key=f"download_png_{name}",
        )
    else:  # pragma: no cover - only reached if matplotlib is unavailable
        st.caption(
            "PNG export needs matplotlib (not available in this environment); use the CSV download "
            "above, or a plotly chart's own camera-icon toolbar button, instead."
        )


# =========================================================================
# Sections
# =========================================================================


def _x_axis(grid, use_wavelength: bool) -> tuple[np.ndarray, str]:
    if use_wavelength:
        return grid.wavelength_um, "Wavelength (um)"
    return grid.wavenumber_cm1, "Wavenumber (cm^-1)"


def _axis_name(x_label: str) -> str:
    return "wavelength" if x_label.startswith("Wavelength") else "wavenumber"


def _scenario_summary(result: ScenarioResult, attn: Optional[SourceIntensityRemaining] = None) -> None:
    sc = result.scenario
    path = result.path
    bits = [
        f"**{MODEL_LABELS[sc.atmosphere]}** atmosphere, **{AEROSOL_LABELS[sc.aerosol]}** aerosol "
        f"(visibility {sc.visibility_km:g} km)",
        f"sensor alt {sc.sensor_alt_km:g} km, zenith {sc.sensor_zenith_deg:g} deg, azimuth {sc.sensor_azimuth_deg:g} deg",
        f"path: {path.n_layers} layers, {path.total_length_km:.3f} km, "
        f"hits_ground={path.hits_ground}, reaches_space={path.reaches_space}"
        + (f", tangent_alt={path.tangent_alt_km:.2f} km" if path.tangent_alt_km is not None else ""),
        f"grid: {result.grid.wavenumber_cm1[0]:.2f}-{result.grid.wavenumber_cm1[-1]:.2f} cm^-1 "
        f"({result.grid.wavelength_um[-1]:.3f}-{result.grid.wavelength_um[0]:.3f} um), "
        f"{result.grid.resolution_cm1:g} cm^-1 resolution, {result.grid.n} points"
        + (f", line-resolved at {result.fine_step_cm1:.4g} cm^-1 through a "
           f"{sc.instrument_function.value} instrument function"
           if result.fine_step_cm1 is not None else ", point-sampled"),
    ]
    if sc.source_range_km is not None and attn is not None:
        ref_txt = f"{attn.reference_range_km * 1000:.0f} m" if attn.reference_range_km < 1.0 else f"{attn.reference_range_km:.3f} km"
        bits.append(
            f"source range {sc.source_range_km:g} km: **{attn.percentage_remaining:.3f}%** of source intensity remains "
            f"(reference baseline at {ref_txt}, {attn.reference_transmittance_pct:.3f}% trans)"
        )
    if sc.radiance_solver == RadianceSolver.DISORT:
        bits.insert(0, f"**DISORT multiple scattering**, {sc.disort_streams} streams, **plane-parallel geometry**")
    if sc.radiance_solver == RadianceSolver.SPHERICAL_MC:
        bits.insert(0, f"**Spherical multiple scattering**, {sc.spherical_photons} photon histories per spectral point, seed {sc.spherical_seed}")
    if result.sun is not None:
        bits.append(
            f"sun: zenith {result.sun.zenith_deg:.1f} deg, above_horizon={result.sun.is_above_horizon}"
        )
    if result.moon is not None:
        bits.append(
            f"moon: zenith {result.moon.zenith_deg:.1f} deg, above_horizon={result.moon.is_above_horizon}, "
            f"illuminated={result.moon.illuminated_fraction:.2f}"
        )
    st.markdown("  \n".join(f"- {b}" for b in bits))


def _source_intensity_card(attn: SourceIntensityRemaining, scenario: Scenario) -> None:
    st.subheader("Source Target Attenuation")
    ref_desc = f"{attn.reference_range_km * 1000:.0f} m" if attn.reference_range_km < 1.0 else f"{attn.reference_range_km:.3f} km"

    col1, col2, col3, col4 = st.columns(4)
    if attn.is_thermal:
        col1.metric(
            "Source Radiance Remaining",
            f"{attn.percentage_remaining:.3f}%",
            help="Percentage of in-band emitted target radiance reaching the sensor aperture through atmospheric extinction.",
        )
        col2.metric(
            "Path Transmittance",
            f"{attn.transmittance_at_range * 100:.3f}%",
            delta=f"{attn.transmittance_remaining_pct:.3f}% of ref",
            help=f"Mean atmospheric transmittance along the {attn.source_range_km:g} km path (reference transmittance at {ref_desc}: {attn.reference_transmittance_pct:.3f}%).",
        )
        col3.metric(
            "In-Band Target Radiance",
            f"{attn.intensity_at_range:.2e} W/cm²/sr",
            help=f"Received at sensor vs {attn.intensity_at_reference:.2e} W/cm²/sr at {ref_desc} reference.",
        )
    else:
        col1.metric(
            "Source Intensity Remaining",
            f"{attn.percentage_remaining:.3f}%",
            help="Percentage of source intensity surviving atmospheric attenuation.",
        )
        col2.metric(
            "Path Transmittance",
            f"{attn.transmittance_at_range * 100:.3f}%",
            help=f"Mean atmospheric transmittance along the {attn.source_range_km:g} km path.",
        )
        col3.metric(
            "Near Reference Transmittance",
            f"{attn.reference_transmittance_pct:.3f}%",
            help=f"Transmittance at {ref_desc} where atmospheric attenuation is negligible.",
        )

    col4.metric("Source Slant Range", f"{attn.source_range_km:g} km", delta=f"ref: {ref_desc}")

    st.caption(
        f"Compared against a short-range baseline of **{ref_desc}** along the line of sight "
        f"where atmospheric attenuation is nearly zero ({attn.reference_transmittance_pct:.3f}% transmittance). "
        + (f"Weights Planck emission at T = {scenario.source_temp_k:g} K."
           if attn.is_thermal else "Unweighted path transmittance.")
    )


def _transmittance_section(result: ScenarioResult, x: np.ndarray, x_label: str) -> None:
    st.subheader(f"Transmittance vs {_axis_name(x_label)}")
    t = result.transmittance
    available = {"total": t.total, "molecular": t.molecular, "aerosol": t.aerosol, "rayleigh": t.rayleigh}
    chosen = st.multiselect(
        "Components", list(available), default=["total", "molecular", "aerosol"], key="trans_components"
    )
    series = {name: available[name] for name in chosen}
    plots.line_chart(st, x, series, x_label=x_label, y_label="Transmittance", title="")
    png = plots.line_chart_png(x, series, x_label=x_label, y_label="Transmittance")
    download_row(f"transmittance_{'_'.join(chosen) or 'none'}", png)


def _species_section(result: ScenarioResult, x: np.ndarray, x_label: str) -> None:
    st.subheader("Per-species transmittance")
    by_species = result.transmittance.by_species
    ordered_names = [sp.value for sp in REQUIRED_SPECIES if sp in by_species]
    chosen = st.multiselect("Species", ordered_names, default=ordered_names, key="species_components")
    name_to_species = {sp.value: sp for sp in REQUIRED_SPECIES}
    series = {name: by_species[name_to_species[name]] for name in chosen if name_to_species[name] in by_species}
    plots.line_chart(st, x, series, x_label=x_label, y_label="Transmittance", title="")
    png = plots.line_chart_png(x, series, x_label=x_label, y_label="Transmittance")
    download_row(f"species_{'_'.join(chosen) or 'none'}", png)


def _radiance_section(result: ScenarioResult, x: np.ndarray, x_label: str) -> None:
    st.subheader(f"Radiance vs {_axis_name(x_label)}")
    radiance = result.radiance
    if radiance is None:
        st.info("Radiance was not computed for this run.")
        return
    if radiance.sampling_standard_error:
        error = radiance.sampling_standard_error["total"]
        relative = np.divide(error, radiance.total, out=np.zeros_like(error), where=radiance.total > 0)
        st.caption(f"Monte Carlo sampling uncertainty: maximum relative standard error {100*float(np.max(relative)):.2g}% over this spectrum. One standard error, not a bound on physical-model error; rare unsampled contributions can be missed. Increase histories and change the seed to check convergence.")
    components = {name: getattr(radiance, name) for name in RADIANCE_COMPONENTS if name != "total"}
    if result.scenario.radiance_solver in (RadianceSolver.DISORT, RadianceSolver.SPHERICAL_MC):
        st.caption("Thermal components include subsequent scattering and surface reflection. "
                   "Solar/lunar multiple-scattering components include all diffuse atmospheric and surface contributions. "
                   "Direct discs and target emission are separate.")
    total = radiance.total
    y_label = "Radiance (W cm^-2 sr^-1 / cm^-1)"
    if x_label.startswith("Wavelength"):
        components = {name: radiance_to_per_um(y, result.grid.wavenumber_cm1)
                      for name, y in components.items()}
        total = radiance_to_per_um(total, result.grid.wavenumber_cm1)
        y_label = "Radiance (W m^-2 sr^-1 / um)"
    plots.stacked_area_chart(st, x, components, total=total, x_label=x_label, y_label=y_label, title="")
    png = plots.stacked_area_png(x, components, total=total, x_label=x_label, y_label=y_label)
    download_row("radiance", png)


def _turbulence_section(result: ScenarioResult) -> None:
    """Optical turbulence diagnostics, shown only when a Cn^2 model is selected."""
    t = result.turbulence
    if t is None:
        return

    st.subheader("Optical turbulence")
    st.caption(
        f"Cn² model **{result.scenario.turbulence.value}**, evaluated at "
        f"{t.reference_wavelength_um:.2f} µm. These are diagnostics: turbulence "
        "redistributes a point source in angle and time but absorbs nothing, so "
        "no radiance above is modified."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fried parameter r₀", f"{t.fried_parameter_r0_cm:.1f} cm",
                help="Aperture size over which the wavefront stays roughly flat. "
                     "Scales as λ^(6/5), so MWIR values are ~12x the visible ones.")
    col2.metric("Isoplanatic angle θ₀", f"{t.isoplanatic_angle_urad:.1f} µrad",
                help="Angular field over which the turbulence-induced wavefront error is common.")
    col3.metric("Scintillation index", f"{t.scintillation_index:.4f}",
                help="Normalised intensity variance. Saturates near 1 in strong turbulence.")
    col4.metric("Greenwood freq.", "n/a" if t.greenwood_frequency_hz is None
                else f"{t.greenwood_frequency_hz:.1f} Hz",
                help="Bandwidth an adaptive-optics loop would need to track the turbulence.")

    if t.aperture_diameter_m is None:
        return

    st.markdown("**Point-source response** for a %.0f cm aperture" % (t.aperture_diameter_m * 100))
    regime = "seeing-limited" if t.is_seeing_limited else "diffraction-limited"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diffraction limit λ/D", f"{t.diffraction_limit_urad:.2f} µrad")
    c2.metric("Seeing blur λ/r₀", f"{t.seeing_limit_urad:.2f} µrad")
    c3.metric("Effective spot", f"{t.effective_blur_urad:.2f} µrad")
    c4.metric("Strehl / peak ratio", f"{t.strehl_ratio:.3f}")

    if t.is_seeing_limited:
        st.warning(
            f"Aperture ({t.aperture_diameter_m * 100:.0f} cm) exceeds r₀ "
            f"({t.fried_parameter_r0_cm:.0f} cm) — **{regime}**. A point source is "
            f"blurred to ~{t.effective_blur_urad:.1f} µrad rather than the "
            f"{t.diffraction_limit_urad:.1f} µrad the optics could deliver, and "
            f"on-axis peak irradiance falls to {t.strehl_ratio:.0%} of the "
            "turbulence-free case. Total collected power is unchanged."
        )
    else:
        st.success(
            f"Aperture ({t.aperture_diameter_m * 100:.0f} cm) is smaller than r₀ "
            f"({t.fried_parameter_r0_cm:.0f} cm) — still **{regime}**. Turbulence "
            f"costs only {1 - t.strehl_ratio:.1%} of peak irradiance here."
        )

    with st.expander("Cn² vs altitude", expanded=False):
        st.caption(
            "Log axis: Cn² spans ~30 orders of magnitude between the surface and "
            "the top of the atmosphere, so a linear axis shows nothing."
        )
        altitudes = [0.5 * (l.z_bottom_km + l.z_top_km) for l in result.path.layers]
        chart = dict(x_label="Cn² (m^-2/3)", title="Refractive-index structure constant", log_x=True)
        plots.profile_chart(st, t.cn2_profile_m23, altitudes, **chart)
        download_row("turbulence_cn2", plots.profile_png(t.cn2_profile_m23, altitudes, **chart))


def _profile_section(result: ScenarioResult) -> None:
    st.subheader("Atmospheric profile")
    profile = result.profile
    col1, col2 = st.columns(2)
    with col1:
        plots.profile_chart(st, profile.temperature_k, profile.altitude_km, x_label="Temperature (K)", title="Temperature")
        png = plots.profile_png(profile.temperature_k, profile.altitude_km, x_label="Temperature (K)", title="Temperature")
        download_row("profile_temperature", png)
    with col2:
        h2o = profile.vmr_ppmv.get(Species.H2O)
        if h2o is not None:
            plots.profile_chart(st, h2o, profile.altitude_km, x_label="H2O (ppmv)", title="Water vapor")
            png = plots.profile_png(h2o, profile.altitude_km, x_label="H2O (ppmv)", title="Water vapor")
            download_row("profile_h2o", png)


def _band_table_section(result: ScenarioResult, run_fn: Optional[Callable] = None) -> None:
    st.subheader("Band summary")
    band_names = list(constants.BANDS_CM1)
    chosen = st.multiselect("Bands", band_names, default=band_names, key="band_selection")
    if not chosen:
        st.info("No bands selected.")
        return
    rows = report.band_rows(result, [(name, name) for name in chosen])
    if not rows:
        st.info("None of the selected bands overlap the computed grid.")
        return

    band_remaining: dict[str, float] = {}
    if result.scenario.source_range_km is not None and result.scenario.source_range_km > 0.0:
        try:
            band_remaining = compute_band_source_remaining(
                result.scenario, result, [(name, name) for name in chosen], run_fn=run_fn
            )
        except Exception:
            pass

    trans_data = []
    for r in rows:
        item = {
            "band": r.label,
            "range (um)": f"{r.lo_um:.3f}-{r.hi_um:.3f}",
            **r.transmittance,
            **{f"species: {sp}": v for sp, v in r.species_transmittance.items()},
        }
        if r.label in band_remaining:
            item["source remaining (%)"] = f"{band_remaining[r.label]:.3f}%"
        trans_data.append(item)

    trans_df = pd.DataFrame(trans_data).set_index("band")
    st.caption("Band-averaged transmittance")
    st.dataframe(trans_df, width="stretch")

    if any(r.radiance is not None for r in rows):
        rad_data = []
        for r in rows:
            if r.radiance is None:
                continue
            item = {"band": r.label, **r.radiance}
            if r.label in band_remaining:
                item["source remaining (%)"] = f"{band_remaining[r.label]:.3f}%"
            rad_data.append(item)
        radiance_df = pd.DataFrame(rad_data).set_index("band")
        st.caption("Band-integrated radiance (W cm^-2 sr^-1)")
        st.dataframe(radiance_df, width="stretch")

    csv_buf = io.StringIO()
    trans_df.to_csv(csv_buf)
    st.download_button(
        "Download band table (CSV)", data=csv_buf.getvalue(),
        file_name="mwir_band_summary.csv", mime="text/csv", key="download_band_csv",
    )
