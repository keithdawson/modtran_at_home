"""DISORT reference spectra, energy balance, and the application's geometry seam."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

from mwir_atmos._vendor.pythonic_disort import pydisort, subroutines
from mwir_atmos.radiometry.multiple_scattering import solve_column, phase_moments, horizontal_source
from mwir_atmos.types import SpectralSampling
from mwir_atmos.types import Scenario, RadianceSolver, AerosolModel, SurfaceType
from mwir_atmos.api import build_profile, run_scenario
from mwir_atmos.geometry.plane_parallel import geometry
from mwir_atmos.setups import save_setup, load_setup
from mwir_atmos.cli import RADIANCE_COMPONENTS


def toy_radiance(*, sensor=1., angle=180., distance=1., target=None, solar=True):
    from mwir_atmos.types import SpectralGrid, PathLayer, PathGeometry, LayerOptics, SkyPosition, AtmosphereProfile, ModelAtmosphere
    from mwir_atmos.optics.transmittance import path_transmittance
    from mwir_atmos.radiometry.multiple_scattering import compute_multiple_scattering
    grid = SpectralGrid(np.array([2500.,2501.]),1.)
    layer = PathLayer(0,0.,1.,1.,1013.,300.,50.,1e24,{}, {}, {})
    column = PathGeometry([layer],0.,0.,0.,None,1.,False,True)
    end = sensor + np.cos(np.deg2rad(angle))*distance
    sightline = replace(layer, z_bottom_km=min(sensor,end), z_top_km=max(sensor,end), ds_km=distance)
    path = PathGeometry([sightline],sensor,angle,90.,None,distance,abs(end)<1e-8,abs(end-1)<1e-8)
    optical = LayerOptics(np.full((1,2),.5),np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,2)),grid,
                          aerosol_asymmetry=np.zeros((1,2)))
    sightopt = replace(optical,tau_molecular=optical.tau_molecular*distance)
    scenario = Scenario(spectral_sampling=SpectralSampling.POINT, radiance_solver=RadianceSolver.DISORT, sensor_alt_km=sensor,
                        sensor_zenith_deg=angle,sensor_azimuth_deg=90.,source_range_km=distance,
                        source_temp_k=target,surface_emissivity=.7, surface_skin_temp_k=300.,include_solar=solar,include_lunar=False)
    profile = AtmosphereProfile(np.array([0.,1.]),np.array([1013.,900.]),np.array([300.,300.]),{},ModelAtmosphere.US_STANDARD)
    sun=SkyPosition(np.rad2deg(np.arccos(.6)),0.,90.,True)
    moon=SkyPosition(180.,0.,180.,False)
    return compute_multiple_scattering(scenario=scenario,grid=grid,profile=profile,path=path,optics=sightopt,
                                      column=column,column_optics=optical,transmittance=path_transmittance(sightopt),sun=sun,moon=moon)


def test_adapter_lambertian_beam_reflection_has_correct_units_and_angles():
    from mwir_atmos.radiometry.radiance import solar_irradiance
    result=toy_radiance()
    expected=.3/np.pi*solar_irradiance(result.grid)*.6*np.exp(-.5/.6)*np.exp(-.5)
    np.testing.assert_allclose(result.solar_multiple_scatter,expected,rtol=1e-5)
    assert np.all(result.direct_solar == 0)


def test_finite_target_replaces_background_without_duplicate_surface_emission():
    from mwir_atmos.spectral import planck
    result=toy_radiance(target=400.,solar=False)
    expected=planck(result.grid.wavenumber_cm1,300.)*-np.expm1(-.5)+planck(result.grid.wavenumber_cm1,400.)*np.exp(-.5)
    np.testing.assert_allclose(result.total,expected,rtol=1e-5)

@pytest.mark.parametrize('case', ['1a','1d','2a','2d','7a','8a','9c'])
def test_original_stamnes_disort_4099(case):
    with np.load(Path(__file__).parent/'reference_data/disort'/f'{case}.npz') as data:
        args = {k.removeprefix('input_'): data[k] for k in data.files if k.startswith('input_')}
        expected = {k.removeprefix('expected_'): data[k] for k in data.files if k.startswith('expected_')}
    for key in ['NQuad','NLeg','NFourier']:
        if key in args:
            args[key] = int(args[key])
    if 'surface_albedo' in args:
        args['BDRF_Fourier_modes'] = [float(args.pop('surface_albedo'))]
    mu, up, down, _, intensity = pydisort(**args)
    tau, phi = expected['tau_test_arr'], expected['phi_arr']
    calculated = {'flup':up(tau), 'rfldn':down(tau)[0], 'rfldir':down(tau)[1],
                  'uu': intensity(tau, phi)[np.argsort(mu)]}
    for name, actual in calculated.items():
        np.testing.assert_allclose(actual, expected[name], rtol=2e-3,
                                   atol=max(float(np.max(np.abs(expected[name])))*2e-6, 1e-6),
                                   err_msg=f'{case} {name}')


def test_conservative_scattering_conserves_flux():
    moments = phase_moments(np.array([1.]), np.array([0.]), np.array([.6]), 32)
    _, up, down, _, _ = solve_column([3.], [1.], moments, streams=32, beam=1., mu0=.6, surface_albedo=.7)
    tau = np.linspace(0.,3.,15)
    net = down(tau)[0] + down(tau)[1] - up(tau)
    np.testing.assert_allclose(net, net[0], rtol=2e-7, atol=2e-9)
    assert float(up(0)) > 0


def test_isothermal_absorbing_layer_matches_formal_solution():
    moments = phase_moments(np.zeros(1), np.zeros(1), np.zeros(1), 16)
    mu, _, _, _, intensity = solve_column([2.], [0.], moments, planck_layers=[3.], surface_emission=5.)
    tau = .7
    expected = np.where(mu > 0, 3.+2.*np.exp(-(2.-tau)/np.abs(mu)), 3.*-np.expm1(-tau/np.abs(mu)))
    np.testing.assert_allclose(intensity(tau, 0.), expected, rtol=2e-12, atol=1e-12)


def test_horizontal_source_has_correct_thermal_limit_and_multiple_scattering():
    moments = phase_moments(np.array([1.]), np.array([0.]), np.array([0.]), 16)
    solved = solve_column([2.], [0.], moments, planck_layers=[3.])
    assert horizontal_source(solved, 1., 0., moments[0], planck_value=3.) == pytest.approx(3.)
    solved = solve_column([2.], [.9], moments, beam=1., mu0=.6)
    source = horizontal_source(solved, 1., .9, moments[0], mu0=.6, beam=1.)
    first_order = .9/(4*np.pi)*np.exp(-1/.6)
    assert source > 2*first_order


def test_rayleigh_hg_moment_mixture():
    m = phase_moments(np.array([3.]), np.array([1.]), np.array([.8]), 16)[0]
    assert m[0] == 1.
    assert m[1] == pytest.approx(.75*.8)
    assert m[2] == pytest.approx(.75*.8**2+.25*.1)


def test_formal_sightline_matches_disort_at_quadrature_angle():
    from mwir_atmos.radiometry.multiple_scattering import integrate_sightline
    from mwir_atmos.types import PathLayer, PathGeometry
    moments=phase_moments(np.array([1.]),np.array([0.]),np.array([.7]),32)
    solved=solve_column([3.],[.95],moments,streams=32,beam=1.,mu0=.6,
                        planck_layers=[2.],surface_emission=.5,surface_albedo=.2)
    index=9
    mu=solved[0][index]
    layer=PathLayer(0,0.,1.,1/mu,1013.,300.,50.,1e24,{}, {}, {})
    path=PathGeometry([layer],1.,float(np.rad2deg(np.arccos(-mu))),0.,None,1/mu,True,False)
    result=integrate_sightline(solved,path=path,optical_depth=np.array([3/mu]),heights=np.array([0.,1.]),
                              tau_edges=np.array([0.,3.]),omega=np.array([.95]),moments=moments,
                              mu_view=mu,phi=.8,planck_layers=np.array([2.]),mu0=.6,beam=1.,
                              surface_emission=.5,surface_albedo=.2,include_surface=True)
    expected=solved[-1](0.,.8)[index]
    assert result == pytest.approx(expected,rel=2e-6)


@pytest.mark.parametrize('angle,distance', [(0.,None),(60.,2.),(90.,3.),(150.,.05)])
def test_plane_parallel_geometry_shares_column_and_preserves_columns(angle, distance):
    s=Scenario(spectral_sampling=SpectralSampling.POINT, radiance_solver=RadianceSolver.DISORT, sensor_alt_km=.1,
               sensor_zenith_deg=angle, source_range_km=distance)
    column, path, indices, factors=geometry(s,build_profile(s))
    assert column.layers[0].z_bottom_km == pytest.approx(0.)
    assert column.layers[-1].z_top_km == pytest.approx(100.)
    for layer, i, factor in zip(path.layers, indices, factors):
        assert layer.air_column_cm2 == pytest.approx(column.layers[i].air_column_cm2*factor)
    if distance:
        assert path.total_length_km == pytest.approx(distance)


def test_horizontal_requires_finite_range():
    s=Scenario(spectral_sampling=SpectralSampling.POINT, radiance_solver=RadianceSolver.DISORT, sensor_zenith_deg=90.)
    with pytest.raises(ValueError, match='positive source_range'):
        geometry(s,build_profile(s))


def test_disort_setup_and_real_scenario(tmp_path):
    s=Scenario(spectral_sampling=SpectralSampling.POINT, radiance_solver=RadianceSolver.DISORT, aerosol=AerosolModel.LOWTRAN_MARITIME,
               sensor_alt_km=.1, sensor_zenith_deg=90., source_range_km=1.,
               wavenumber_min_cm1=2500., wavenumber_max_cm1=2502., resolution_cm1=1.,
               include_solar=False, include_lunar=False)
    save_setup(s,tmp_path/'disort.yaml')
    loaded=load_setup(tmp_path/'disort.yaml')
    assert loaded.radiance_solver == RadianceSolver.DISORT
    r=run_scenario(loaded)
    assert np.isfinite(r.radiance.total).all() and np.all(r.radiance.total > 0)
    np.testing.assert_allclose(r.radiance.total, sum(getattr(r.radiance,k) for k in RADIANCE_COMPONENTS if k != 'total'))
    assert np.all(r.radiance.solar_single_scatter == 0)
    # Constant horizontal medium: all atmospheric origins share 1-exp(-tau).
    longer=run_scenario(replace(s,source_range_km=2.))
    expected=r.radiance.total*(1+r.transmittance.total)
    np.testing.assert_allclose(longer.radiance.total,expected,rtol=1e-10)
