"""Tests for source attenuation and near-zero baseline comparison."""

import pytest
from mwir_atmos.source_attenuation import (
    choose_reference_range_km,
    compute_band_source_remaining,
    compute_source_intensity_remaining,
)
from mwir_atmos.types import Scenario, SpectralSampling


def _sc(**kwargs):
    defaults = dict(
        wavenumber_min_cm1=2200.0,
        wavenumber_max_cm1=2400.0,
        resolution_cm1=2.0,
        spectral_sampling=SpectralSampling.POINT,
    )
    defaults.update(kwargs)
    return Scenario(**defaults)


def test_reference_distance_selection():
    # Long distance: standard reference is 100 m (0.1 km)
    assert choose_reference_range_km(10.0) == pytest.approx(0.1)
    assert choose_reference_range_km(5.0) == pytest.approx(0.1)

    # Moderate distance (0.05 km = 50 m): 1/10th = 5 m (0.005 km)
    assert choose_reference_range_km(0.05) == pytest.approx(0.005)

    # Very short distance (0.00005 km = 5 cm): floored at 1 cm (0.00001 km)
    assert choose_reference_range_km(0.00005) == pytest.approx(0.00001)


def test_source_intensity_remaining_none_when_no_source_range():
    sc = _sc(source_range_km=None)
    assert compute_source_intensity_remaining(sc) is None

    sc_zero = _sc(source_range_km=0.0)
    assert compute_source_intensity_remaining(sc_zero) is None


def test_source_intensity_remaining_thermal_source():
    sc = _sc(source_range_km=5.0, source_temp_k=650.0)
    attn = compute_source_intensity_remaining(sc)

    assert attn is not None
    assert attn.is_thermal is True
    assert attn.source_range_km == 5.0
    assert attn.reference_range_km == pytest.approx(0.1)
    assert 0.0 <= attn.percentage_remaining <= 100.0
    assert 0.0 <= attn.transmittance_remaining_pct <= 100.0

    # Physical monotonicity: received intensity at 5 km is strictly less than at 100 m
    assert attn.intensity_at_range < attn.intensity_at_reference
    assert attn.transmittance_at_range < attn.transmittance_at_reference
    assert attn.reference_transmittance_pct > 30.0


def test_source_intensity_remaining_pure_transmission_no_thermal():
    sc = _sc(source_range_km=3.0, source_temp_k=None)
    attn = compute_source_intensity_remaining(sc)

    assert attn is not None
    assert attn.is_thermal is False
    assert attn.source_range_km == 3.0
    assert 0.0 <= attn.percentage_remaining <= 100.0
    # In pure transmission mode, percentage_remaining matches transmittance_remaining_pct
    assert attn.percentage_remaining == pytest.approx(attn.transmittance_remaining_pct)


def test_source_intensity_monotonic_with_distance():
    sc_short = _sc(source_range_km=1.0, source_temp_k=650.0)
    sc_long = _sc(source_range_km=5.0, source_temp_k=650.0)

    attn_short = compute_source_intensity_remaining(sc_short)
    attn_long = compute_source_intensity_remaining(sc_long)

    assert attn_short is not None and attn_long is not None
    # Shorter distance must preserve more source intensity than longer distance
    assert attn_short.percentage_remaining > attn_long.percentage_remaining
    assert attn_short.transmittance_remaining_pct > attn_long.transmittance_remaining_pct


def test_compute_band_source_remaining():
    sc = _sc(source_range_km=5.0, source_temp_k=650.0, wavenumber_min_cm1=2000.0, wavenumber_max_cm1=2600.0)
    from mwir_atmos.api import run_scenario

    result = run_scenario(sc)

    bands = [
        ("MWIR_window_1", "MWIR_window_1"),
        ("CO2_4_3um", "CO2_4_3um"),
    ]
    remaining = compute_band_source_remaining(sc, result, bands)

    assert "MWIR_window_1" in remaining
    assert "CO2_4_3um" in remaining
    # Atmospheric window must transmit significantly more than the opaque CO2 band
    assert remaining["MWIR_window_1"] > remaining["CO2_4_3um"]
