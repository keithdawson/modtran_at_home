"""Original Fortran reference checks and conservation/integration tests."""
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from mwir_atmos.optics.lowtran_aerosol import optical_properties, extinction_profile, TABLE_MODELS
from mwir_atmos.optics.aerosol import aerosol_optical_depth
from mwir_atmos.types import AerosolModel, PathLayer, SpectralGrid, Scenario
from mwir_atmos.setups import save_setup, load_setup
from mwir_atmos.validation import validate_scenario

REF = Path(__file__).parent / "reference_data"


def layer(a=0., b=0., ds=1., rh=75.):
    return PathLayer(0, a, b, ds, 1013.25, 288.15, rh, 2.5e24, {}, {}, {})


def test_original_fortran_spectral_reference():
    with np.load(REF / "lowtran_spectra.npz") as data:
        rows = data['data']
    actual = []
    for r in rows:
        p = {1: 'RUR', 5: 'URB', 4: 'OCN', 6: 'TRO'}[int(r['code'])] if r['region'] == 1 else {2: 'TRO', 3: 'BST', 4: 'DME'}[int(r['region'])]
        actual.append(np.array(optical_properties(p, 70. if r['region'] == 2 else r['rh_pct'],
                      np.array([np.clip(r['wavelength_um'], .2, 40.)]))).ravel())
    expected = np.column_stack([rows[n] for n in ('extinction', 'absorption', 'g')])
    # Original REAL*4 arithmetic and wavelength round-trip limit agreement.
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-9)


def test_original_fortran_altitude_profiles():
    with np.load(REF / "lowtran_profiles.npz") as data:
        rows = data['data']
    actual = [extinction_profile(r['visibility_km'], winter=r['season'] == 2)[1][int(r['node']) - 1] for r in rows]
    np.testing.assert_allclose(actual, rows['extinction_per_km'], rtol=3e-7)


@pytest.mark.parametrize('model', list(TABLE_MODELS))
def test_layer_split_conserves_absorption_scattering_and_phase_moment(model):
    grid = SpectralGrid(np.linspace(2000., 3333., 30), 1333./29)
    whole = [layer(0., 40., 80.)]
    split = [layer(a, b, 2 * (b-a)) for a, b in zip([0., 1.3, 2., 7., 10., 30.], [1.3, 2., 7., 10., 30., 40.])]
    a, s, g = aerosol_optical_depth(whole, grid, model)
    aa, ss, gg = aerosol_optical_depth(split, grid, model)
    np.testing.assert_allclose(a[0], aa.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(s[0], ss.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose((s*g)[0], (ss*gg).sum(axis=0), rtol=1e-12)
    assert whole[0].aerosol_od_550nm == pytest.approx(sum(x.aerosol_od_550nm for x in split))
    assert np.all(a >= 0) and np.all(s >= 0) and np.all((g >= 0) & (g <= 1))


def test_horizontal_visibility_loading_and_no_artificial_scattering_cap():
    grid = SpectralGrid(np.array([1e4/3.75, 1e4/.55]), 1e4/.55-1e4/3.75)
    dry, wet = layer(rh=0), layer(rh=99)
    a, s, _ = aerosol_optical_depth([dry, wet], grid, AerosolModel.LOWTRAN_MARITIME)
    np.testing.assert_allclose(a[:, 1] + s[:, 1], .158)
    assert dry.aerosol_od_550nm == pytest.approx(wet.aerosol_od_550nm)
    assert np.any(s/(a+s) > .5)
    assert not np.isclose((a+s)[0, 0], (a+s)[1, 0])


def test_reverse_path_and_upper_boundary():
    grid = SpectralGrid(np.array([2500., 2600.]), 100.)
    a = aerosol_optical_depth([layer(1., 110., 109.)], grid, AerosolModel.LOWTRAN_RURAL)
    b = aerosol_optical_depth([layer(110., 1., 109.)], grid, AerosolModel.LOWTRAN_RURAL)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)
    assert all(np.all(x == 0) for x in aerosol_optical_depth([layer(101., 110., 9.)], grid, AerosolModel.LOWTRAN_RURAL))


def test_table_setup_roundtrip_and_validation(tmp_path):
    scenario = Scenario(aerosol=AerosolModel.LOWTRAN_MARITIME)
    save_setup(scenario, tmp_path / 'setup.yaml')
    assert load_setup(tmp_path / 'setup.yaml').aerosol == scenario.aerosol
    for visibility in [1., 51.]:
        with pytest.raises(ValueError, match='between 2 and 50'):
            validate_scenario(replace(scenario, visibility_km=visibility))
    with pytest.raises(ValueError, match='0.2 to 40'):
        optical_properties('RUR', 70., np.array([41.]))
    for rh, capped in [(-1., 0.), (100., 99.)]:
        np.testing.assert_allclose(optical_properties('RUR', rh, np.array([4.])), optical_properties('RUR', capped, np.array([4.])))
