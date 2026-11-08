"""Atmospheric attenuation and source intensity retention against a near-zero reference.

When a discrete source/target is placed at an explicit slant range
(``scenario.source_range_km``), atmospheric gases and aerosols attenuate its
intensity before reaching the sensor.

To quantify how much source intensity survives, this module compares the
received signal against a near reference range (100 m = 0.1 km) along the
same sightline (unattenuated source baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from . import api
from .report import BandRow, BandSpec, band_rows
from .setups import with_overrides
from .types import Scenario, ScenarioResult

__all__ = [
    "SourceIntensityRemaining",
    "choose_reference_range_km",
    "compute_band_source_remaining",
    "compute_source_intensity_remaining",
]

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


@dataclass(frozen=True)
class SourceIntensityRemaining:
    """Atmospheric attenuation metrics for a target at ``source_range_km``."""

    percentage_remaining: float
    """Percentage of source intensity (0-100%) surviving atmospheric attenuation to the sensor."""

    source_range_km: float
    """Target slant range from sensor, km."""

    reference_range_km: float
    """Ultra-short reference range, km (where atmospheric extinction is nearly zero)."""

    intensity_at_range: float
    """Integrated source radiance (W cm^-2 sr^-1) or mean transmittance at ``source_range_km``."""

    intensity_at_reference: float
    """Integrated source radiance (W cm^-2 sr^-1) or mean transmittance at ``reference_range_km``."""

    transmittance_at_range: float
    """Mean total path transmittance at ``source_range_km`` (0.0 - 1.0)."""

    transmittance_at_reference: float
    """Mean total path transmittance at ``reference_range_km`` (~1.0)."""

    transmittance_remaining_pct: float
    """Percentage of path transmittance remaining relative to the near reference."""

    is_thermal: bool
    """True if evaluating discrete thermal source emission (weighted by Planck function)."""

    reference_transmittance_pct: float
    """Percentage transmittance at the reference distance (~100%)."""


def choose_reference_range_km(source_range_km: float) -> float:
    """Select a reference distance for baseline comparison (standard: 100 m = 0.1 km).

    Bounds:
        - Standard reference distance is 100 m (0.1 km).
        - For target ranges <= 100 m, scales down to 1/10th of the range,
          floored at 1 cm (1e-5 km).
    """
    range_val = float(source_range_km)
    if range_val <= 0.1:
        return max(1.0e-5, range_val * 0.1)
    return 0.1


def compute_source_intensity_remaining(
    scenario: Scenario,
    result: Optional[ScenarioResult] = None,
    *,
    reference_range_km: Optional[float] = None,
    run_fn: Optional[Callable[[Scenario], ScenarioResult]] = None,
) -> Optional[SourceIntensityRemaining]:
    """Calculate the percentage of source intensity that remains after atmospheric attenuation.

    Compares the scenario's received signal against a near-zero reference path
    along the same sightline where atmospheric attenuation has not yet degraded
    the signal.

    Parameters:
        scenario: The active scenario with ``source_range_km`` set.
        result: Pre-computed ``ScenarioResult`` for ``scenario`` (computed if None).
        reference_range_km: Custom reference distance, km (default: 100 m = 0.1 km).
        run_fn: Optional runner function (e.g. cached GUI executor), defaults to ``api.run_scenario``.

    Returns:
        :class:`SourceIntensityRemaining`, or None if ``source_range_km`` is not set.
    """
    if scenario.source_range_km is None or scenario.source_range_km <= 0.0:
        return None

    runner = run_fn if run_fn is not None else api.run_scenario

    if result is None:
        result = runner(scenario)

    if reference_range_km is None:
        ref_km = choose_reference_range_km(scenario.source_range_km)
    else:
        ref_km = float(reference_range_km)

    sc_ref = with_overrides(scenario, source_range_km=ref_km, source_alt_km=None)
    result_ref = runner(sc_ref)

    # Path transmittance comparison
    trans_range = float(np.mean(result.transmittance.total))
    trans_ref = float(np.mean(result_ref.transmittance.total))
    trans_pct = (trans_range / trans_ref * 100.0) if trans_ref > 0.0 else 0.0
    trans_pct = min(100.0, max(0.0, trans_pct))

    # Thermal emission vs pure transmission
    is_thermal = (
        scenario.source_temp_k is not None
        and result.radiance is not None
        and result_ref.radiance is not None
    )

    if is_thermal:
        grid_wn = result.grid.wavenumber_cm1
        int_range = float(_trapz(result.radiance.source_emitted, grid_wn))
        int_ref = float(_trapz(result_ref.radiance.source_emitted, grid_wn))
        pct = (int_range / int_ref * 100.0) if int_ref > 0.0 else 0.0
        pct = min(100.0, max(0.0, pct))
    else:
        int_range = trans_range
        int_ref = trans_ref
        pct = trans_pct

    return SourceIntensityRemaining(
        percentage_remaining=pct,
        source_range_km=float(scenario.source_range_km),
        reference_range_km=ref_km,
        intensity_at_range=int_range,
        intensity_at_reference=int_ref,
        transmittance_at_range=trans_range,
        transmittance_at_reference=trans_ref,
        transmittance_remaining_pct=trans_pct,
        is_thermal=is_thermal,
        reference_transmittance_pct=trans_ref * 100.0,
    )


def compute_band_source_remaining(
    scenario: Scenario,
    result: ScenarioResult,
    band_specs: Sequence[tuple[str, BandSpec]],
    *,
    reference_range_km: Optional[float] = None,
    run_fn: Optional[Callable[[Scenario], ScenarioResult]] = None,
) -> dict[str, float]:
    """Calculate the source intensity percentage remaining for each band."""
    if scenario.source_range_km is None or scenario.source_range_km <= 0.0:
        return {}

    runner = run_fn if run_fn is not None else api.run_scenario

    if reference_range_km is None:
        ref_km = choose_reference_range_km(scenario.source_range_km)
    else:
        ref_km = float(reference_range_km)

    sc_ref = with_overrides(scenario, source_range_km=ref_km, source_alt_km=None)
    result_ref = runner(sc_ref)

    rows = band_rows(result, band_specs)
    rows_ref = band_rows(result_ref, band_specs)
    row_ref_by_label = {r.label: r for r in rows_ref}

    is_thermal = scenario.source_temp_k is not None and result.radiance is not None

    remaining_by_band: dict[str, float] = {}
    for r in rows:
        ref_r = row_ref_by_label.get(r.label)
        if ref_r is None:
            continue
        if is_thermal and r.radiance is not None and ref_r.radiance is not None:
            src_range = r.radiance.get("source_emitted", 0.0)
            src_ref = ref_r.radiance.get("source_emitted", 0.0)
            val = (src_range / src_ref * 100.0) if src_ref > 0.0 else 0.0
        else:
            trans_range = r.transmittance.get("total", 0.0)
            trans_ref = ref_r.transmittance.get("total", 0.0)
            val = (trans_range / trans_ref * 100.0) if trans_ref > 0.0 else 0.0
        remaining_by_band[r.label] = min(100.0, max(0.0, val))

    return remaining_by_band
