"""Run multiple-scattering and angular-convergence comparisons, save a figure."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from datetime import datetime
from dataclasses import replace
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mwir_atmos.api import run_scenario
from mwir_atmos.types import Scenario,RadianceSolver,AerosolModel
from mwir_atmos.radiometry.multiple_scattering import solve_column,phase_moments
from mwir_atmos._vendor.pythonic_disort import subroutines
from mwir_atmos.spectral import radiance_to_per_um

base=Scenario(radiance_solver=RadianceSolver.DISORT,aerosol=AerosolModel.LOWTRAN_MARITIME,
              visibility_km=2.,sensor_alt_km=.1,sensor_zenith_deg=60.,sensor_azimuth_deg=90.,
              latitude_deg=0.,local_datetime=datetime(2026,3,20,12),utc_offset_hours=0.,
              include_lunar=False,wavenumber_min_cm1=2200.,wavenumber_max_cm1=2900.,resolution_cm1=10.)
results={}
times={}
for streams in [16,32,64]:
    started=time.perf_counter()
    results[streams]=run_scenario(replace(base,disort_streams=streams))
    times[streams]=time.perf_counter()-started
    print(f'{streams} streams: {times[streams]:.2f} seconds',flush=True)

fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
mu=np.linspace(.1,1.,100)
moments=phase_moments(np.array([1.]),np.array([0.]),np.array([0.]),64)
solved=solve_column([2.],[.95],moments,streams=64,mu0=.6,beam=1.)
multiple=np.asarray(subroutines.interpolate(solved[-1])(mu,0.,0.)).ravel()
single=.95/(4*np.pi)*.6/(mu+.6)*-np.expm1(-2*(1/mu+1/.6))
axes[0,0].plot(np.rad2deg(np.arccos(mu)),single,label='Exactly one scattering')
axes[0,0].plot(np.rad2deg(np.arccos(mu)),multiple,label='All scattering orders')
axes[0,0].set(title='Isotropic slab: optical depth 2, albedo 0.95',
              xlabel='Viewing angle from upward normal (degrees)',ylabel='Reflected radiance / incident irradiance (sr$^{-1}$)')
reference=results[64]
lam=reference.grid.wavelength_um
for streams,result in results.items():
    y=radiance_to_per_um(result.radiance.total,result.grid.wavenumber_cm1)
    axes[0,1].plot(lam,y,label=f'{streams} streams',linestyle='--' if streams==16 else '-')
    if streams != 64:
        relative=100*np.abs(result.radiance.total/reference.radiance.total-1.)
        axes[1,0].semilogy(lam,np.maximum(relative,1e-8),label=f'{streams} vs 64 streams')
for name,label in [('path_thermal','Atmospheric thermal origin'),('surface_emitted','Surface thermal origin'),
                    ('solar_multiple_scatter','Solar diffuse (all orders)')]:
    axes[1,1].plot(lam,radiance_to_per_um(getattr(results[32].radiance,name),reference.grid.wavenumber_cm1),label=label)
axes[0,1].set(title='Dense maritime atmosphere: total radiance',ylabel='Radiance (W m$^{-2}$ sr$^{-1}$ µm$^{-1}$)')
axes[1,0].set(title='Angular convergence against 64 streams',ylabel='Absolute relative difference (%)')
axes[1,1].set(title='32-stream radiance by original source',ylabel='Radiance (W m$^{-2}$ sr$^{-1}$ µm$^{-1}$)')
for ax in [axes[0,1],axes[1,0],axes[1,1]]:
    ax.set_xlabel('Wavelength (µm)')
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('Multiple scattering and surface feedback — numerical checks',fontsize=15)
root=Path(__file__).resolve().parents[2]
fig.savefig(root/'docs/multiple_scattering_checks.png',dpi=150)
report={'scenario':'100 m sensor, zenith 60°, maritime visibility 2 km, equinox noon at equator; 2200–2900 cm^-1 at 10 cm^-1',
        'runtime_seconds':times,'max_relative_radiance_difference':{str(s):float(np.max(np.abs(results[s].radiance.total/reference.radiance.total-1))) for s in [16,32]}}
(root/'docs/multiple_scattering_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(report,flush=True)
