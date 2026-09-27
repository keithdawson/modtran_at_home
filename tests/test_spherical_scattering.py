"""Analytic spherical limits, angular sampling, and independent DISORT checks."""
import io
import json
from dataclasses import replace
import numpy as np
import pytest
from mwir_atmos.geometry.spherical_shells import segments, direction
from mwir_atmos.radiometry.spherical_scattering import solve_sphere, beam_transmission, sample_phase
from mwir_atmos.radiometry.multiple_scattering import solve_column, phase_moments
from mwir_atmos._vendor.pythonic_disort import subroutines


def test_shell_chords_and_ground_interception():
    r = np.array([6371.,6372.,6471.])
    p = np.array([[0.,0.,6371.],[0.,0.,6372.]])
    d = np.array([[1.,0.,0.],[0.,0.,-1.]])
    edges, ids, end, ground = segments(p,d,r)
    assert end[0] == pytest.approx(np.sqrt(6471.**2-6371.**2))
    assert end[1] == pytest.approx(1.)
    assert ground.tolist() == [False,True]
    assert np.sum(np.diff(edges[0])[ids[0]==0]) == pytest.approx(np.sqrt(6372.**2-6371.**2))


def test_twilight_shadow_depends_on_scattering_altitude():
    r = np.array([6371.,6471.])
    p = np.array([[0.,0.,6371.],[0.,0.,6421.]])
    t = beam_transmission(p,direction(94.,0.),r,np.array([0.]))
    np.testing.assert_array_equal(t,[0.,1.])


@pytest.mark.parametrize('angle',[0.,90.,180.])
def test_absorption_only_spherical_formal_solution(angle):
    r = np.array([6371.,6471.]); p=np.array([0.,0.,6372.]); d=direction(angle,0.)
    _,_,end,ground=segments(p,d,r)
    mean,error,total_error=solve_sphere(r,[.01],[0.],[0.],[0.],position=p,look=d,
        planck_layers=[3.],surface_emission=5.,photons=128)
    expected=3.*-np.expm1(-.01*end[0])+ground[0]*5.*np.exp(-.01*end[0])
    assert mean.sum() == pytest.approx(expected,rel=1e-12)
    assert total_error < 1e-13


def test_lambertian_solar_boundary_exact_units():
    mean,_,_=solve_sphere([6371.,6372.],[0.],[0.],[0.],[0.],position=[0,0,6372.],look=[0,0,-1],
                         surface_albedo=.4,beams=[(direction(60.,0.),2.)],photons=128)
    assert mean[2] == pytest.approx(.4/np.pi,rel=1e-12)


@pytest.mark.parametrize('fraction,g,expected1,expected2',[(0.,0.,0.,.1),(1.,.7,.7,.49),(.4,.6,.24,.4*.36+.6*.1)])
def test_phase_sampling_has_correct_legendre_moments(fraction,g,expected1,expected2):
    n=100000; axis=np.tile([0.,0.,1.],(n,1)); rng=np.random.default_rng(48)
    out=sample_phase(axis,np.full(n,fraction),np.full(n,g),rng)
    np.testing.assert_allclose(np.linalg.norm(out,axis=1),1.,atol=1e-12)
    c=out[:,2]
    assert abs(c.mean()-expected1)<.007
    assert abs(((3*c*c-1)/2).mean()-expected2)<.007


@pytest.mark.parametrize('aerosol,g',[(0.,0.),(1.,.6)])
def test_large_radius_limit_matches_independent_disort(aerosol,g):
    # Radius 1e6 km keeps geometric corrections small while avoiding cancellation.
    moment=phase_moments(np.array([aerosol]),np.array([1-aerosol]),np.array([g]),32)
    solved=solve_column([.7],[.8],moment,streams=32,beam=1.,mu0=.6,surface_albedo=.3)
    interpolated=subroutines.interpolate(solved[4])
    reference=float(interpolated(-.5,.7,1.2))
    mean,error,_=solve_sphere([1e6,1e6+1],[.7],[.8],[aerosol],[g],
        position=[0,0,1e6],look=direction(60.,np.rad2deg(1.2)),surface_albedo=.3,
        beams=[(direction(np.rad2deg(np.arccos(.6)),0.),1.)],photons=32768,seed=17)
    assert abs(mean[2]-reference) < 5*error[2]+.002*reference


def test_event_limit_raises_instead_of_truncating():
    with pytest.raises(RuntimeError,match='event limit'):
        solve_sphere([1.,2.],[100.],[1.],[0.],[0.],position=[0,0,1.5],look=[0,0,1],photons=128,max_events=1)


def test_scattered_thermal_and_ground_sources_match_disort_limit():
    moments=phase_moments(np.array([1.]),np.array([0.]),np.array([.6]),32)
    solved=solve_column([.7],[.8],moments,streams=32,planck_layers=[3.],surface_emission=5.,surface_albedo=.3)
    reference=float(subroutines.interpolate(solved[4])(-.5,.7,0.))
    mean,_,error=solve_sphere([1e6,1e6+1],[.7],[.8],[1.],[.6],position=[0,0,1e6],look=direction(60.,0.),
        planck_layers=[3.],surface_emission=5.,surface_albedo=.3,photons=32768,seed=72)
    assert mean[0]>0 and mean[1]>0
    assert abs(mean.sum()-reference)<5*error+.002*reference


def test_real_scenario_setup_finite_target_and_exports(tmp_path):
    from mwir_atmos.types import Scenario,RadianceSolver,AerosolModel,SpectralSampling
    from mwir_atmos.api import run_scenario
    from mwir_atmos.setups import save_setup,load_setup
    from mwir_atmos.cli import _write_json,_write_csv,_write_npz,RADIANCE_COMPONENTS
    from mwir_atmos.spectral import planck
    sc=Scenario(spectral_sampling=SpectralSampling.POINT, radiance_solver=RadianceSolver.SPHERICAL_MC,spherical_photons=128,
                sensor_alt_km=.1,sensor_zenith_deg=90.,source_range_km=1.,source_temp_k=400.,
                include_solar=False,include_lunar=False,wavenumber_min_cm1=2500.,wavenumber_max_cm1=2504.,resolution_cm1=4.)
    setup=tmp_path/'sphere.yaml'; save_setup(sc,setup)
    loaded=load_setup(setup); assert loaded==sc
    result=run_scenario(loaded)
    assert np.all(result.radiance.total>0)
    np.testing.assert_allclose(result.radiance.source_emitted,planck(result.grid.wavenumber_cm1,400.)*result.transmittance.total)
    np.testing.assert_allclose(result.radiance.total,sum(getattr(result.radiance,n) for n in RADIANCE_COMPONENTS if n!='total'))
    json_out=io.StringIO(); _write_json(result,[],json_out)
    assert 'total' in json.loads(json_out.getvalue())['radiance_sampling_standard_error']
    csv_out=io.StringIO(); _write_csv(result,csv_out)
    assert 'radiance_stderr_total' in csv_out.getvalue()
    _write_npz(result,[],str(tmp_path/'out.npz'))
    with np.load(tmp_path/'out.npz') as f: assert 'radiance_stderr_total' in f
    second=run_scenario(loaded)
    np.testing.assert_array_equal(second.radiance.total,result.radiance.total)

