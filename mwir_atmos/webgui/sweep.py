"""GUI "Distance sweep" mode: move the source outward until band transmittance
drops below a threshold (:func:`mwir_atmos.setups.sweep_distance`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from mwir_atmos.setups import sweep_distance, with_overrides
from mwir_atmos.spectral import band_average
from mwir_atmos.types import Scenario

__all__ = ["distance_sweep_section", "sweep_distances"]

#: Largest number of distances one GUI sweep may run.
_MAX_DISTANCES = 200


def sweep_distances(start: float, stop: float, step: float) -> np.ndarray:
    """``start, start + step, ...`` up to and including ``stop``."""
    if start <= 0.0:
        raise ValueError("Start distance must be greater than zero")
    if step <= 0.0:
        raise ValueError("Distance step must be greater than zero")
    if stop < start:
        raise ValueError("Maximum distance must be at least the start distance")
    count = int(np.floor((stop - start) / step)) + 1
    if count > _MAX_DISTANCES:
        raise ValueError(f"Use at most {_MAX_DISTANCES} distances per sweep; increase the distance step")
    distances = start + np.arange(count) * step
    if not np.isclose(distances[-1], stop):
        distances = np.append(distances, stop)
    if len(distances) > _MAX_DISTANCES:
        raise ValueError(f"Use at most {_MAX_DISTANCES} distances per sweep")
    return distances


def _check_band_um(scenario: Scenario, lo_um: float, hi_um: float) -> tuple[float, float]:
    """The band must lie inside the scenario's wavelength window and hold two samples."""
    if lo_um <= 0.0 or hi_um <= 0.0:
        raise ValueError("Band wavelengths must be greater than zero")
    if lo_um >= hi_um:
        raise ValueError("Minimum wavelength must be strictly below maximum wavelength")

    wn_from_hi_wl = 10000.0 / hi_um
    wn_from_lo_wl = 10000.0 / lo_um
    sc_wn_min = float(scenario.wavenumber_min_cm1)
    sc_wn_max = float(scenario.wavenumber_max_cm1)

    if wn_from_hi_wl < sc_wn_min - 0.5 or wn_from_lo_wl > sc_wn_max + 0.5:
        sc_wl_min = 10000.0 / sc_wn_max
        sc_wl_max = 10000.0 / sc_wn_min
        raise ValueError(
            f"Choose a band inside the scenario window ({sc_wl_min:.3f} - {sc_wl_max:.3f} um), "
            "with minimum below maximum"
        )

    lo_cm1 = max(sc_wn_min, wn_from_hi_wl)
    hi_cm1 = min(sc_wn_max, wn_from_lo_wl)

    first_index = np.ceil((lo_cm1 - sc_wn_min) / scenario.resolution_cm1)
    last_index = np.floor((hi_cm1 - sc_wn_min) / scenario.resolution_cm1)
    if last_index - first_index < 1:
        raise ValueError("The band must contain at least two computed spectral samples")
    return lo_cm1, hi_cm1









def distance_sweep_section(scenario: Scenario, cache_key: str, backend: str = "auto") -> None:
    st.subheader("Distance sweep")
    st.caption("Move the source outward along the saved sensor direction, holding other inputs fixed. Source-altitude limits are cleared. Stop at the first sampled band transmittance below the threshold.")
    st.info("Transmittance is atmospheric throughput, not detector detectability. A Python script can supply its own detectability or SNR metric.")
    default_lo_um = round(10000.0 / float(scenario.wavenumber_max_cm1), 3)
    default_hi_um = round(10000.0 / float(scenario.wavenumber_min_cm1), 3)

    if "sweep_start" not in st.session_state:
        st.session_state["sweep_start"] = 0.1
    if "sweep_stop" not in st.session_state:
        st.session_state["sweep_stop"] = 10.0
    if "sweep_step" not in st.session_state:
        st.session_state["sweep_step"] = 1.0
    if "sweep_threshold" not in st.session_state:
        st.session_state["sweep_threshold"] = 0.1

    last_scenario_key = st.session_state.get("_last_sweep_scenario_key")
    if last_scenario_key != cache_key:
        st.session_state["_last_sweep_scenario_key"] = cache_key
        st.session_state["sweep_band_lo_um"] = default_lo_um
        st.session_state["sweep_band_hi_um"] = default_hi_um
    else:
        st.session_state.setdefault("sweep_band_lo_um", default_lo_um)
        st.session_state.setdefault("sweep_band_hi_um", default_hi_um)
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Start distance (km)", min_value=0.0, step=0.1, format="%.3f", key="sweep_start")
    stop = c2.number_input("Maximum distance (km)", min_value=0.0, step=1.0, format="%.1f", key="sweep_stop")
    step = c3.number_input("Distance step (km)", min_value=0.0, step=0.5, format="%.3f", key="sweep_step")
    c1, c2, c3 = st.columns(3)
    lo_um = c1.number_input("Band minimum (um)", min_value=0.0, step=0.001, format="%.3f", key="sweep_band_lo_um")

    hi_um = c2.number_input("Band maximum (um)", min_value=0.0, step=0.001, format="%.3f", key="sweep_band_hi_um")


    threshold = c3.number_input("Stop below transmittance", min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key="sweep_threshold")
    if lo_um > 0.0 and hi_um > lo_um:
        st.caption(f"-> Spectral band: {lo_um:.3f} - {hi_um:.3f} um ({10000.0 / hi_um:.1f} - {10000.0 / lo_um:.1f} cm^-1)")
    submitted = st.button("Run distance sweep", type="primary", key="btn_run_distance_sweep")

    # Results persist across reruns until any input changes.
    # Sync directly from session_state in case of rapid clicks
    start = float(st.session_state.get("sweep_start", start))
    stop = float(st.session_state.get("sweep_stop", stop))
    step = float(st.session_state.get("sweep_step", step))
    lo_um = float(st.session_state.get("sweep_band_lo_um", lo_um))
    hi_um = float(st.session_state.get("sweep_band_hi_um", hi_um))
    threshold = float(st.session_state.get("sweep_threshold", threshold))

    signature = (cache_key, backend, start, stop, step, lo_um, hi_um, threshold)
    if submitted:
        try:
            distances = sweep_distances(start, stop, step)
            with_overrides(scenario)  # Validate inputs before sizing a grid.
            lo_cm1, hi_cm1 = _check_band_um(scenario, lo_um, hi_um)
            sweep = _run_with_progress(scenario, distances, lo_cm1, hi_cm1, threshold, backend)
            st.session_state["distance_sweep_result"] = (signature, sweep)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Distance sweep stopped: {exc}")
            return
    stored = st.session_state.get("distance_sweep_result")
    if stored is None or stored[0] != signature:
        return
    _show_sweep(stored[1], threshold)


def _run_with_progress(scenario: Scenario, distances: np.ndarray, lo: float, hi: float,
                       threshold: float, backend: str = "auto"):
    circle_placeholder = st.empty()
    circle_placeholder.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="border: 3px solid rgba(37, 99, 235, 0.2); border-top: 3px solid #2563eb; border-radius: 50%; width: 22px; height: 22px; animation: mwir-spin 0.75s linear infinite; flex-shrink: 0;"></div>
            <span style="font-weight: 600; color: #1e3a8a;">Running distance sweep...</span>
        </div>
        <style>
        @keyframes mwir-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)
    progress = st.progress(0.0, text="Starting distance sweep")
    completed = []

    def update(sample):
        completed.append(sample)
        progress.progress(len(completed) / len(distances),
                          text=f"Computed {sample.distance_km:g} km: transmittance {sample.score:.4f}")

    try:
        sweep = sweep_distance(scenario, distances, threshold=threshold,
                           metric=lambda r: band_average(r.grid, r.transmittance.total, (lo, hi)),
                           include_radiance=False, on_sample=update, backend=backend)
    finally:
        circle_placeholder.empty()
        progress.empty()
    return sweep


def _show_sweep(sweep, threshold: float) -> None:
    if sweep.crossed_threshold:
        if sweep.last_passing_distance_km is None:
            st.warning(f"Already below threshold at the first sample ({sweep.first_failing_distance_km:g} km). Try a shorter start distance.")
        else:
            st.success(f"First sampled crossing: {sweep.last_passing_distance_km:g} to {sweep.first_failing_distance_km:g} km. Reduce the step to refine this interval.")
    else:
        st.info("Threshold not crossed within the sampled distances. This does not establish a maximum range.")
    frame = pd.DataFrame({"distance_km": [v.distance_km for v in sweep.samples],
                          "band_transmittance": [v.score for v in sweep.samples]})
    st.line_chart(frame.set_index("distance_km").assign(threshold=threshold))
    st.dataframe(frame, hide_index=True)
    st.download_button("Download distance sweep (CSV)", frame.to_csv(index=False),
                       file_name="mwir_distance_sweep.csv", mime="text/csv", key="download_sweep")
