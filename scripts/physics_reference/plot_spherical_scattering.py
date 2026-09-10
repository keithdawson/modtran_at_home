"""Reproducible spherical/plane-parallel comparison and real twilight spectrum."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from datetime import datetime
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mwir_atmos.radiometry.spherical_scattering import solve_sphere
from mwir_atmos.radiometry.multiple_scattering import solve_column,phase_moments
from mwir_atmos.geometry.spherical_shells import direction
from mwir_atmos._vendor.pythonic_disort import subroutines
from mwir_atmos.types import Scenario,RadianceSolver,AerosolModel
from mwir_atmos.api import run_scenario
from mwir_atmos.spectral import radiance_to_per_um

root=Path(__file__).resolve().parents[2]
angles=np.array([0.,30.,60.,75.,80.,85.,88.,89.,90.,91.,92.])
means=[]; errors=[]; flat=[]
moments=phase_moments(np.array([0.]),np.array([1.]),np.array([0.]),32)
disort=solve_column([.2],[1.],moments,streams=32,mu0=.5,beam=1.,surface_albedo=.1)
interpolate=subroutines.interpolate(disort[4])
for i,angle in enumerate(angles):
    mean,error,_=solve_sphere([6371.,6381.],[.02],[1.],[0.],[0.],position=[0.,0.,6371.1],
        look=direction(angle,90.),beams=[(direction(60.,0.),1.)],surface_albedo=.1,
        photons=32768,seed=300+i)
    means.append(mean[2]); errors.append(error[2])
    flat.append(float(interpolate(-np.cos(np.deg2rad(angle)),.198,np.pi/2)) if angle<90 else np.nan)
sc=Scenario(radiance_solver=RadianceSolver.SPHERICAL_MC,spherical_photons=8192,
    sensor_alt_km=10.,sensor_zenith_deg=90.,sensor_azimuth_deg=90.,
    aerosol=AerosolModel.LOWTRAN_MARITIME,visibility_km=23.,latitude_deg=0.,longitude_deg=0.,
    local_datetime=datetime(2026,3,20,18,20),utc_offset_hours=0.,include_lunar=False,
    wavenumber_min_cm1=2800.,wavenumber_max_cm1=2840.,resolution_cm1=4.)
result=run_scenario(sc)
rad=result.radiance; x=result.grid.wavelength_um
convert=lambda a:radiance_to_per_um(a,result.grid.wavenumber_cm1)
fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
ax[0].errorbar(angles,means,yerr=2*np.array(errors),fmt='o-',label='Spherical MC (±2 standard errors)')
ax[0].plot(angles,flat,'--',label='Plane-parallel DISORT')
ax[0].set(xlabel='Sensor zenith (degrees)',ylabel='Radiance / normal solar irradiance (sr⁻¹)',
          title='Rayleigh shell: optical depth 0.2, Sun zenith 60°')
ax[0].legend(fontsize=8)
ax[1].errorbar(x,convert(rad.total),yerr=2*convert(rad.sampling_standard_error['total']),fmt='o-',label='Total (±2 standard errors)')
ax[1].plot(x,convert(rad.solar_multiple_scatter),label='Diffuse solar origin')
ax[1].plot(x,convert(rad.path_thermal),label='Atmospheric thermal origin')
ax[1].set(xlabel='Wavelength (µm)',ylabel='Radiance (W m⁻² sr⁻¹ µm⁻¹)',
          title=f'Spherical horizontal view at 10 km, twilight\nSun zenith {result.sun.zenith_deg:.1f}°, LOWTRAN maritime')
ax[1].legend(fontsize=8)
for a in ax: a.grid(alpha=.25)
fig.savefig(root/'docs/spherical_scattering_checks.png',dpi=140)
payload={'angles_deg':angles.tolist(),'spherical':means,'standard_error':errors,
         'flat': [None if not np.isfinite(x) else x for x in flat],
         'toy_photons':32768,'real_photons':sc.spherical_photons,
         'sun_zenith_deg':result.sun.zenith_deg,'real_wavenumber_cm1':result.grid.wavenumber_cm1.tolist(),
         'real_total':rad.total.tolist(),'real_total_standard_error':rad.sampling_standard_error['total'].tolist(),
         'real_solar':rad.solar_multiple_scatter.tolist()}
(root/'docs/spherical_scattering_checks.json').write_text(json.dumps(payload,indent=2)+'\n')
print('Wrote spherical comparison plot and numerical results')
