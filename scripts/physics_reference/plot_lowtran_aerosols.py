"""Run representative horizontal scenarios and plot aerosol model comparisons."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mwir_atmos.api import run_scenario
from mwir_atmos.types import Scenario, AerosolModel
from mwir_atmos.optics.lowtran_aerosol import optical_properties, extinction_profile

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for model, label, style in [(AerosolModel.RURAL, 'Rural legacy', '--'),
                            (AerosolModel.LOWTRAN_RURAL, 'Rural LOWTRAN', '-'),
                            (AerosolModel.NAVY_MARITIME, 'Navy maritime legacy', '--'),
                            (AerosolModel.LOWTRAN_MARITIME, 'Maritime LOWTRAN', '-')]:
    scenario = Scenario(aerosol=model, sensor_alt_km=.1, sensor_zenith_deg=90.,
                        source_range_km=5., visibility_km=23., ground_rh_pct=75.,
                        resolution_cm1=2.)
    result = run_scenario(scenario)
    lam = result.grid.wavelength_um
    assert np.isfinite(result.transmittance.total).all()
    assert result.radiance is not None
    axes[0, 0].plot(lam, result.transmittance.aerosol, style, label=label)
    axes[1, 0].plot(lam, result.transmittance.total, style, alpha=.8, label=label)

lam = np.linspace(2.4, 5.5, 500)
for prefix, label in [('RUR', 'Rural'), ('OCN', 'Maritime')]:
    for rh, style in [(0., '--'), (99., '-')]:
        ext, ab, _ = optical_properties(prefix, rh, lam)
        axes[0, 1].plot(lam, 1-ab/ext, style, label=f'{label}, RH {rh:g}%')
for vis in [5., 23., 50.]:
    z, sigma = extinction_profile(vis)
    axes[1, 1].semilogx(sigma, z, label=f'Visibility {vis:g} km')
axes[1, 1].set(xlabel='Aerosol extinction at 550 nm (km$^{-1}$)', ylabel='Altitude MSL (km)', ylim=(0, 12))
for ax in axes[:, 0]:
    ax.set(xlabel='Wavelength (µm)', ylabel='Transmittance', xlim=(2.4, 5.5), ylim=(0, 1.02))
axes[0, 1].set(xlabel='Wavelength (µm)', ylabel='Single-scattering albedo', ylim=(0, 1))
for ax, title in zip(axes.flat, ['Aerosol-only transmission: 5 km path at 100 m',
                                'LOWTRAN optical tables: no imposed albedo ceiling',
                                'Full atmosphere: gas + aerosol + Rayleigh',
                                'Original background profiles: spring/summer']):
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('LOWTRAN 7 aerosols compared with the legacy approximation', fontsize=15)
dest = Path(__file__).resolve().parents[2] / 'docs/lowtran_aerosol_comparison.png'
fig.savefig(dest, dpi=160)
print(dest)
