"""Saved inputs, selective overrides, and bounded threshold sweeps."""
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pytest
from mwir_atmos import api, setups
from mwir_atmos.types import Scenario, AerosolModel
from mwir_atmos.spectral import band_average


def test_save_load_overrides_preserve_original(tmp_path):
    original = Scenario(local_datetime=datetime(2026,9,8,12,30), aerosol=AerosolModel.DESERT,
                        source_alt_km=2., source_range_km=3., visibility_km=15.)
    path = tmp_path/'project'/'setup.yaml'
    setups.save_setup(original,path)
    loaded = setups.load_setup(path,source_alt_km=None,source_range_km=5.)
    assert loaded.local_datetime == original.local_datetime
    assert loaded.aerosol == original.aerosol
    assert loaded.visibility_km == 15.
    assert loaded.source_alt_km is None and loaded.source_range_km == 5.
    assert setups.load_setup(path) == original
    assert original.source_range_km == 3.
    with pytest.raises(ValueError,match='unknown scenario field'):
        setups.load_setup(path,visiblity_km=10.)
    with pytest.raises(ValueError,match='visibility_km'):
        setups.with_overrides(original,visibility_km=-1.)


def fake_runner(monkeypatch, scores):
    calls=[]
    def run(sc,**kwargs):
        calls.append(sc)
        assert kwargs['include_turbulence'] is False
        return SimpleNamespace(score=next(scores))
    monkeypatch.setattr(api,'run_scenario',run)
    return calls


def test_threshold_crossing_equality_and_no_extra_runs(monkeypatch):
    calls=fake_runner(monkeypatch,iter([.5,.2,.1]))
    original=Scenario(source_alt_km=10.,source_range_km=9.)
    emitted=[]
    result=setups.sweep_distance(original,[1,2,3,4],metric=lambda r:r.score,threshold=.2,on_sample=emitted.append)
    assert result.crossed_threshold
    assert result.last_passing_distance_km == 2.
    assert result.first_failing_distance_km == 3.
    assert len(calls) == len(emitted) == 3
    assert all(sc.source_alt_km is None for sc in calls)
    assert [sc.source_range_km for sc in calls] == [1,2,3]
    assert original.source_alt_km == 10. and original.source_range_km == 9.


@pytest.mark.parametrize('scores,passed,failed', [([.1],None,1.),([.5,.4],2.,None)])
def test_first_failure_and_no_crossing(monkeypatch,scores,passed,failed):
    fake_runner(monkeypatch,iter(scores))
    result=setups.sweep_distance(Scenario(),range(1,len(scores)+1),metric=lambda r:r.score,threshold=.2)
    assert result.last_passing_distance_km == passed
    assert result.first_failing_distance_km == failed
    assert result.crossed_threshold == (failed is not None)


@pytest.mark.parametrize('distances', [[],[0],[float('nan')],[float('inf')],[2,1],[1,1]])
def test_reject_invalid_distances_before_running(monkeypatch,distances):
    calls=fake_runner(monkeypatch,iter([]))
    with pytest.raises(ValueError,match='distances'):
        setups.sweep_distance(Scenario(),distances,metric=lambda r:r.score,threshold=.2)
    assert not calls


def test_reject_clipped_path_before_spectrum(monkeypatch):
    calls=fake_runner(monkeypatch,iter([]))
    with pytest.raises(ValueError,match='available path'):
        setups.sweep_distance(Scenario(sensor_alt_km=1.,sensor_zenith_deg=180.),[2.],
                              metric=lambda r:r.score,threshold=.2)
    assert not calls


def test_reject_nonfinite_metric(monkeypatch):
    fake_runner(monkeypatch,iter([float('nan')]))
    with pytest.raises(ValueError,match='non-finite'):
        setups.sweep_distance(Scenario(),[1.],metric=lambda r:r.score,threshold=.2)


def test_saved_setup_and_sweep_run_real_physics(tmp_path):
    setup=Scenario(sensor_alt_km=.1,sensor_zenith_deg=90.,wavenumber_min_cm1=2500.,
                   wavenumber_max_cm1=2510.,resolution_cm1=1.,source_range_km=.1)
    path=tmp_path/'real.yaml'; setups.save_setup(setup,path)
    direct=api.run_scenario(setups.with_overrides(setup,source_range_km=.2),include_radiance=False,include_turbulence=False)
    saved=setups.run_setup(path,overrides={'source_range_km':.2},include_radiance=False,include_turbulence=False)
    np.testing.assert_array_equal(direct.transmittance.total,saved.transmittance.total)
    result=setups.sweep_distance(path,[.1,.2,.3],metric=lambda r:band_average(r.grid,r.transmittance.total,(2500.,2510.)),threshold=0.,include_radiance=False)
    assert len(result.samples)==3 and not result.crossed_threshold
    assert result.samples[0].score > result.samples[1].score > result.samples[2].score
    assert result.samples[1].score == pytest.approx(band_average(saved.grid,saved.transmittance.total,(2500.,2510.)))
