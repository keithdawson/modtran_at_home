"""Independent numerical references; no network or Fortran needed at test time."""
from pathlib import Path
import numpy as np
import pytest
from mwir_atmos.optics.continuum import _mt_ckd_coefficients, h2o_continuum_optical_depth
from mwir_atmos.optics.rayleigh import _rayleigh_cross_section_cm2
from mwir_atmos.types import SpectralGrid, PathLayer, Species

DATA = Path(__file__).parent / 'reference_data'

@pytest.mark.parametrize('case', [1, 2, 3, 4, 5])
def test_continuum_against_unmodified_aer_fortran(case):
    with np.load(DATA / 'mt_ckd_4_3_fortran.npz') as data:
        d = {name: data[name][data['case'] == case] for name in data.files}
    grid = SpectralGrid(d['wavenumber_cm1'], float(np.diff(d['wavenumber_cm1'])[0]))
    p, t, vmr = (d[n][0] for n in ['pressure_mbar','temperature_k','h2o_vmr'])
    actual = _mt_ckd_coefficients(grid, p, t, vmr)
    for result, key in zip(actual, ['self_cm2','foreign_cm2']):
        # Absolute tolerance must be zero: these physical values are << 1e-8.
        np.testing.assert_allclose(result, d[key], rtol=2e-12, atol=0)
    air_column = 2.5e24
    layer = PathLayer(index=0, z_bottom_km=0, z_top_km=0, ds_km=1,
                      p_mbar=p, t_k=t, rh_pct=0, air_column_cm2=air_column,
                      column_cm2={Species.H2O: air_column*vmr},
                      p_eff_mbar={Species.H2O:p}, t_eff_k={Species.H2O:t})
    expected_tau = (d['self_cm2']+d['foreign_cm2'])*air_column*vmr
    np.testing.assert_allclose(h2o_continuum_optical_depth([layer], grid)[0],
                               expected_tau, rtol=2e-12, atol=0)


def test_continuum_against_aer_distributed_example():
    with np.load(DATA / 'mt_ckd_4_3_aer_example.npz') as d:
        grid = SpectralGrid(d['wavenumbers'], 1.0)
        actual = _mt_ckd_coefficients(grid, 1013.0, 300.0, 0.00990098)
        for result, key in zip(actual, ['self_absorption', 'frgn_absorption']):
            np.testing.assert_allclose(result, d[key], rtol=2e-12, atol=0)


def test_continuum_rejects_extrapolation():
    with pytest.raises(ValueError, match='interpolation range'):
        _mt_ckd_coefficients(SpectralGrid(np.array([21000.0]), 1.0), 1013., 296., .01)


# Tomasi et al. (2005), Applied Optics 44, Table 4, US Standard surface.
# P=1013 mbar, T=288.20 K, water partial pressure=7.85075 mbar, CO2=385 ppm.
# The model uses dry air and Peck-Reeder dispersion; allow 0.3% for that
# documented difference. This is a comparison, not a fit to these values.
@pytest.mark.parametrize('wavelength, published', [
    (.50,6.6451e-27), (.55,4.4995e-27), (1.0,4.0030e-28),
    (1.5,7.8549e-29), (2.0,2.4796e-29), (2.2,1.6927e-29),
    (2.4,1.1947e-29), (2.6,8.6714e-30), (2.8,6.4453e-30),
    (3.0,4.8900e-30), (3.5,2.6386e-30), (4.0,1.5463e-30),
])
def test_rayleigh_against_published_cross_sections(wavelength, published):
    actual = _rayleigh_cross_section_cm2(np.array([wavelength]))[0]
    assert actual == pytest.approx(published, rel=.003, abs=0)
