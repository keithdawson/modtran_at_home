"""Grid construction, Planck, band integration and resolution conversions.

Every test here is pure arithmetic -- no line-by-line calls -- so the whole file
runs in well under a second. That matters: the physics suites are minutes long,
so the cheap invariants belong somewhere they can be run constantly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mwir_atmos.constants import C2_RAD, STEFAN_BOLTZMANN
from mwir_atmos.spectral import (
    band_average,
    band_integrate,
    band_slice,
    convolve_gaussian,
    dplanck_dt,
    grid_from_wavelength_um,
    make_grid,
    planck,
    radiance_to_per_um,
    resolution_cm1_for_wavelength_step,
    resolution_cm1_to_um,
    resolution_um_to_cm1,
)

# --------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------


def test_grid_spans_the_requested_range_inclusively():
    g = make_grid(2000.0, 2010.0, 1.0)
    assert g.n == 11
    assert g.wavenumber_cm1[0] == pytest.approx(2000.0)
    assert g.wavenumber_cm1[-1] == pytest.approx(2010.0)
    assert np.allclose(np.diff(g.wavenumber_cm1), 1.0)


def test_grid_is_strictly_increasing():
    g = make_grid(1800.0, 4300.0, 1.0)
    assert np.all(np.diff(g.wavenumber_cm1) > 0)


def test_default_grid_covers_the_full_mwir_band():
    """docs/build_contract.md invariant: the default must contain 3-5 um with margin."""
    g = make_grid()
    assert g.wavenumber_cm1[0] <= 2000.0        # 5 um
    assert g.wavenumber_cm1[-1] >= 3333.34      # 3 um


def test_wavelength_and_wavenumber_are_reciprocal():
    g = make_grid(2000.0, 3000.0, 10.0)
    np.testing.assert_allclose(g.wavelength_um, 1.0e4 / g.wavenumber_cm1, rtol=1e-15)


def test_grid_from_wavelength_matches_a_hand_conversion():
    g = grid_from_wavelength_um(3.0, 5.0, 1.0)
    assert g.wavenumber_cm1[0] == pytest.approx(2000.0)
    assert g.wavenumber_cm1[-1] <= 3333.34


@pytest.mark.parametrize(
    "lo, hi, res",
    [(2000.0, 2000.0, 1.0), (3000.0, 2000.0, 1.0), (2000.0, 3000.0, 0.0), (2000.0, 3000.0, -1.0)],
)
def test_degenerate_grids_are_rejected(lo, hi, res):
    with pytest.raises(ValueError):
        make_grid(lo, hi, res)


def test_grid_from_wavelength_rejects_nonsense():
    with pytest.raises(ValueError):
        grid_from_wavelength_um(0.0, 5.0)
    with pytest.raises(ValueError):
        grid_from_wavelength_um(5.0, 3.0)


# --------------------------------------------------------------------------
# Planck
# --------------------------------------------------------------------------


def test_planck_scalar_temperature_returns_one_row():
    g = make_grid(2000.0, 3000.0, 10.0)
    assert planck(g.wavenumber_cm1, 300.0).shape == (g.n,)


def test_planck_vector_temperature_returns_one_row_per_temperature():
    g = make_grid(2000.0, 3000.0, 10.0)
    out = planck(g.wavenumber_cm1, np.array([250.0, 300.0, 350.0]))
    assert out.shape == (3, g.n)
    # Hotter is brighter everywhere in the MWIR (well short of the crossover).
    assert np.all(out[2] > out[1]) and np.all(out[1] > out[0])


@pytest.mark.parametrize("bad_t", [0.0, -1.0, -300.0])
def test_non_positive_temperature_gives_zeros_not_nan(bad_t):
    """A 'no surface' / 'no source' case must be representable without branching."""
    g = make_grid(2000.0, 3000.0, 50.0)
    out = planck(g.wavenumber_cm1, bad_t)
    assert np.all(out == 0.0)
    assert np.all(np.isfinite(out))


def test_planck_peak_sits_where_wien_says():
    """For B_nu the peak is at x = hc*nu/kT ~ 2.821, i.e. nu ~ 1.9607*T cm^-1."""
    for temperature in (200.0, 300.0, 600.0, 1200.0):
        expected = 2.821439 * temperature / C2_RAD
        g = make_grid(max(1.0, 0.3 * expected), 3.0 * expected, expected / 400.0)
        found = g.wavenumber_cm1[int(np.argmax(planck(g.wavenumber_cm1, temperature)))]
        assert found == pytest.approx(expected, rel=0.01)


def test_planck_integrates_to_stefan_boltzmann():
    """Integrating B_nu over all wavenumbers must give sigma*T^4/pi."""
    for temperature in (250.0, 300.0, 400.0):
        g = make_grid(1.0, 40.0 * temperature, 2.0)
        total = band_integrate(g, planck(g.wavenumber_cm1, temperature),
                               (g.wavenumber_cm1[0], g.wavenumber_cm1[-1]))
        expected = STEFAN_BOLTZMANN * temperature**4 / math.pi * 1.0e-4   # W cm^-2 sr^-1
        assert total == pytest.approx(expected, rel=0.01)


def test_mwir_is_a_small_slice_of_a_room_temperature_blackbody():
    """A well-known sanity figure: 3-5 um holds ~1-2% of a 300 K emitter."""
    g = make_grid(1800.0, 4300.0, 1.0)
    in_band = band_integrate(g, planck(g.wavenumber_cm1, 300.0), "MWIR_3_5um")
    total = STEFAN_BOLTZMANN * 300.0**4 / math.pi * 1.0e-4
    assert 0.005 < in_band / total < 0.03


def test_dplanck_dt_matches_a_finite_difference():
    g = make_grid(2000.0, 3300.0, 50.0)
    t, h = 300.0, 0.01
    analytic = dplanck_dt(g.wavenumber_cm1, t)
    numeric = (planck(g.wavenumber_cm1, t + h) - planck(g.wavenumber_cm1, t - h)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


def test_dplanck_dt_is_positive_and_zero_for_dead_sources():
    g = make_grid(2000.0, 3300.0, 100.0)
    assert np.all(dplanck_dt(g.wavenumber_cm1, 300.0) > 0.0)
    assert np.all(dplanck_dt(g.wavenumber_cm1, 0.0) == 0.0)


def test_radiance_per_um_conversion_uses_the_right_jacobian():
    """|dnu/dlambda| = nu^2/1e4 per um, and 1 W cm^-2 = 1e4 W m^-2."""
    g = make_grid(2000.0, 3000.0, 100.0)
    b = planck(g.wavenumber_cm1, 300.0)
    np.testing.assert_allclose(
        radiance_to_per_um(b, g.wavenumber_cm1), b * g.wavenumber_cm1**2, rtol=1e-15
    )


def test_per_um_conversion_preserves_the_band_integral():
    """Integrating in either variable must give the same total radiance."""
    g = make_grid(2000.0, 3333.0, 1.0)
    b_nu = planck(g.wavenumber_cm1, 300.0)
    in_wavenumber = np.trapezoid(b_nu, x=g.wavenumber_cm1)             # W cm^-2 sr^-1
    per_um = radiance_to_per_um(b_nu, g.wavenumber_cm1)                # W m^-2 sr^-1 um^-1
    lam = g.wavelength_um
    order = np.argsort(lam)
    in_wavelength = np.trapezoid(per_um[order], x=lam[order]) * 1.0e-4  # -> W cm^-2 sr^-1
    assert in_wavelength == pytest.approx(in_wavenumber, rel=1e-6)


# --------------------------------------------------------------------------
# Band integration
# --------------------------------------------------------------------------


def test_band_slice_selects_only_in_band_points():
    g = make_grid(2000.0, 3000.0, 1.0)
    sl = band_slice(g, (2400.0, 2600.0))
    selected = g.wavenumber_cm1[sl]
    assert selected[0] >= 2400.0 and selected[-1] <= 2600.0
    assert selected.size == 201


def test_named_bands_resolve():
    g = make_grid(1800.0, 4300.0, 1.0)
    assert band_slice(g, "MWIR_3_5um").stop > band_slice(g, "MWIR_3_5um").start


def test_band_outside_the_grid_raises():
    g = make_grid(2000.0, 2100.0, 1.0)
    with pytest.raises(ValueError):
        band_slice(g, (3000.0, 3100.0))


def test_integrating_a_constant_recovers_width_times_constant():
    g = make_grid(2000.0, 2100.0, 1.0)
    ones = np.full(g.n, 3.0)
    assert band_integrate(g, ones, (2000.0, 2100.0)) == pytest.approx(3.0 * 100.0)


def test_band_average_of_a_constant_is_that_constant():
    g = make_grid(2000.0, 2100.0, 1.0)
    assert band_average(g, np.full(g.n, 0.42), (2000.0, 2100.0)) == pytest.approx(0.42)


def test_band_integrate_handles_extra_leading_axes():
    g = make_grid(2000.0, 2100.0, 1.0)
    stack = np.stack([np.full(g.n, 1.0), np.full(g.n, 2.0)])
    out = band_integrate(g, stack, (2000.0, 2100.0))
    assert out.shape == (2,)
    assert out[1] == pytest.approx(2.0 * out[0])


# --------------------------------------------------------------------------
# Instrument line shape
# --------------------------------------------------------------------------


def test_convolution_below_grid_spacing_is_a_no_op():
    g = make_grid(2000.0, 2100.0, 1.0)
    y = np.random.default_rng(0).random(g.n)
    np.testing.assert_array_equal(convolve_gaussian(g, y, 0.5), y)


def test_convolution_conserves_area():
    g = make_grid(2000.0, 2200.0, 1.0)
    y = np.zeros(g.n)
    y[100] = 1.0
    out = convolve_gaussian(g, y, 10.0)
    assert out.sum() == pytest.approx(y.sum(), rel=1e-6)


def test_convolving_a_spike_produces_the_requested_fwhm():
    g = make_grid(2000.0, 2400.0, 0.5)
    y = np.zeros(g.n)
    y[g.n // 2] = 1.0
    fwhm_request = 12.0
    out = convolve_gaussian(g, y, fwhm_request)

    half = out.max() / 2.0
    above = np.nonzero(out >= half)[0]
    measured = (above[-1] - above[0]) * g.resolution_cm1
    assert measured == pytest.approx(fwhm_request, rel=0.1)


def test_convolution_never_introduces_negative_values():
    g = make_grid(2000.0, 2200.0, 1.0)
    y = np.abs(np.random.default_rng(1).normal(size=g.n))
    assert np.all(convolve_gaussian(g, y, 8.0) >= 0.0)


def test_convolution_preserves_shape_for_stacked_input():
    g = make_grid(2000.0, 2200.0, 1.0)
    stack = np.random.default_rng(2).random((3, g.n))
    assert convolve_gaussian(g, stack, 5.0).shape == stack.shape


# --------------------------------------------------------------------------
# Resolution conversions -- wavenumber vs wavelength
# --------------------------------------------------------------------------


def test_resolution_conversions_are_exact_inverses():
    for wavelength in (3.0, 4.0, 5.0):
        for delta_um in (0.001, 0.01, 0.15):
            cm1 = resolution_um_to_cm1(delta_um, wavelength)
            assert resolution_cm1_to_um(cm1, wavelength) == pytest.approx(delta_um, rel=1e-12)


@pytest.mark.parametrize(
    "delta_um, wavelength_um, expected_cm1",
    [(0.15, 5.0, 60.0), (0.15, 4.0, 93.75), (0.01, 4.0, 6.25), (0.001, 5.0, 0.4)],
)
def test_documented_conversion_values(delta_um, wavelength_um, expected_cm1):
    """These are the figures quoted in the README and CLI help."""
    assert resolution_um_to_cm1(delta_um, wavelength_um) == pytest.approx(expected_cm1, rel=1e-9)


def test_a_fixed_wavenumber_step_widens_with_wavelength():
    """The whole reason the two units are not interchangeable across a band."""
    steps = [resolution_cm1_to_um(4.0, lam) for lam in (3.0, 4.0, 5.0)]
    assert steps == sorted(steps)
    assert steps[0] == pytest.approx(0.0036, rel=1e-6)
    assert steps[-1] == pytest.approx(0.0100, rel=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_resolution_conversions_reject_non_positive_input(bad):
    with pytest.raises(ValueError):
        resolution_um_to_cm1(bad, 4.0)
    with pytest.raises(ValueError):
        resolution_um_to_cm1(0.01, bad)
    with pytest.raises(ValueError):
        resolution_cm1_to_um(bad, 4.0)


def test_wavelength_step_guarantee_holds_across_the_whole_band():
    """The binding constraint is the LONGEST wavelength, so sizing there must
    keep the actual step under the request everywhere -- including the red end."""
    lo, hi = 2000.0, 3333.34
    for target_um in (0.15, 0.01, 0.002):
        res_cm1 = resolution_cm1_for_wavelength_step(target_um, lo, hi)
        g = make_grid(lo, hi, res_cm1)
        worst_step_um = float(np.max(np.abs(np.diff(np.sort(g.wavelength_um)))))
        assert worst_step_um <= target_um * (1.0 + 1e-9)
        # And it must not be needlessly conservative -- within a factor of two.
        assert worst_step_um > target_um * 0.5


def test_wavelength_step_sizing_rejects_a_non_positive_lower_bound():
    with pytest.raises(ValueError):
        resolution_cm1_for_wavelength_step(0.01, 0.0, 3000.0)
