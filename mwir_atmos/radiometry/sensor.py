"""Sensor-adapter boundary: convert calibrated inputs, never guess their meaning.

Future agent: START HERE and read docs/sensor_adapter.md before importing files.
The RT solver produces scene radiance, not detector counts. Keep file decoding,
optical response, and electronic/noise models separate and independently tested.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .polarization import rotation


@dataclass
class SensorResponse:
    """Canonical v1 intensity analyzer, sampled in vacuum wavenumber cm^-1.

    analyzer_row: [N,4] dimensionless first row of the optical Mueller chain.
    roll_deg: detector +Q axis relative to the RT reference basis, positive
    toward e2 about PHYSICAL propagation k (not the outward viewing direction).
    responsivity_e_per_joule: optional [N] *after* analyzer transmission.
    etendue_m2_sr: optional area times accepted solid angle; not aperture area
    alone. exposure_s: integration time. Both are needed to predict electrons.
    """
    wavenumber_cm1: np.ndarray
    analyzer_row: np.ndarray
    roll_deg: float = 0.
    responsivity_e_per_joule: np.ndarray | None = None
    etendue_m2_sr: float | None = None
    exposure_s: float | None = None


def load_sensor_response(path):
    """Reference JSON importer. Add proprietary/CSV/Excel importers beside this.

    Do NOT loosen validation to make an unknown vendor file load. Establish its
    units, spectral density convention, coordinate frame, and calibration stage
    first, then translate to SensorResponse. Calibrated radiance may already have
    throughput removed; applying it a second time corrupts the result.
    """
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    allowed={'version','wavenumber_cm1','analyzer_row','roll_deg','responsivity_e_per_joule','etendue_m2_sr','exposure_s'}
    if data.get('version')!=1 or set(data)-allowed:
        raise ValueError('Unknown sensor response schema: require version 1 and documented keys')
    return SensorResponse(**{k:np.asarray(v,float) if k in ('wavenumber_cm1','analyzer_row','responsivity_e_per_joule') and v is not None else v for k,v in data.items() if k!='version'})


def apply_sensor_response(grid,stokes,response):
    """Apply a passive intensity analyzer and, optionally, predict photoelectrons.

    All I/Q/U/V values are spectral densities in W cm^-2 sr^-1 per cm^-1.
    Dimensionless response resampling does NOT require a spectral Jacobian;
    conversion of radiance density from per-um or per-nm DOES. Keep that conversion
    in the input adapter and validate by preserving an integrated flat spectrum.

    This is a uniform-radiance, one-ray model. Resolved images, nonuniform FOVs,
    PSFs, vignetting and point targets require angular/spatial integration before
    multiplying by etendue. Do not substitute source area for sensor etendue.
    """
    wn=np.asarray(response.wavenumber_cm1,float); row=np.asarray(response.analyzer_row,float)
    if wn.ndim!=1 or len(wn)<2 or row.shape!=(len(wn),4) or not np.isfinite(wn).all() or not np.isfinite(row).all() or np.any(np.diff(wn)<=0):
        raise ValueError('Sensor grid must increase and analyzer_row must be finite [N,4]')
    strength=np.linalg.norm(row[:,1:],axis=1)
    if np.any(row[:,0]<strength-1e-12) or np.any(row[:,0]+strength>1+1e-12):
        raise ValueError('Analyzer row must be passive and map every physical Stokes vector to nonnegative intensity')
    target=grid.wavenumber_cm1
    if target[0]<wn[0] or target[-1]>wn[-1] or not np.isfinite(response.roll_deg):
        raise ValueError('Sensor calibration must cover the whole spectral grid; roll must be finite')
    analyzed=np.sum(np.array([np.interp(target,wn,row[:,j]) for j in range(4)]).T*(np.asarray(stokes)@rotation(np.deg2rad(response.roll_deg)).T),axis=1)
    output={'analyzed_radiance':analyzed.tolist(),'photoelectrons':None,'noise_model':None}
    if response.responsivity_e_per_joule is not None:
        efficiency=np.asarray(response.responsivity_e_per_joule,float)
        if efficiency.shape!=wn.shape or not np.isfinite(efficiency).all() or np.any(efficiency<0):
            raise ValueError('Responsivity must be nonnegative finite electrons per joule on the response grid')
        if response.etendue_m2_sr is None or response.exposure_s is None or not np.isfinite([response.etendue_m2_sr,response.exposure_s]).all() or min(response.etendue_m2_sr,response.exposure_s)<=0:
            raise ValueError('Electron prediction requires positive etendue_m2_sr and exposure_s')
        spectral=analyzed*1e4*np.interp(target,wn,efficiency)
        output['photoelectrons']=float(np.trapezoid(spectral,target)*response.etendue_m2_sr*response.exposure_s)
    # FUTURE NOISE HOOK: shot noise uses TOTAL photons including backgrounds,
    # dark current adds electrons, read noise adds variance per read, and gain
    # converts electrons to DN only after these stages. Include well saturation,
    # clipping, nonlinearity, quantization, bad pixels, and calibration covariance.
    # RT Monte Carlo error is numerical error, NOT physical sensor noise. Q/U/I
    # estimates are correlated: marginal standard errors cannot be propagated
    # through an analyzer without covariance or per-history projected estimates.
    return output
