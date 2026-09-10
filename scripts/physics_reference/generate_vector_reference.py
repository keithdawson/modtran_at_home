"""Dev-only reference: pip install sasktran2==2026.8.2 miepython==3.0.2.

Run with --deps artifacts/validationdeps when packages were installed with
pip --target. Neither reference package is a production/runtime dependency.
"""
import sys
from pathlib import Path
if '--deps' in sys.argv: sys.path.insert(0,sys.argv[sys.argv.index('--deps')+1])
import json
import importlib.metadata
import numpy as np
import sasktran2 as sk
import miepython

assert importlib.metadata.version('sasktran2')=='2026.8.2'
assert importlib.metadata.version('miepython')=='3.0.2'
cases=[]
for depth,az in [(.2,0.),(.7,1.2),(2.,0.)]:
    config=sk.Config();config.num_stokes=3;config.num_streams=32
    config.num_singlescatter_moments=32
    config.multiple_scatter_source=sk.MultipleScatterSource.DiscreteOrdinates
    geometry=sk.Geometry1D(cos_sza=.6,solar_azimuth=0,earth_radius_m=1e9,
        altitude_grid_m=np.linspace(0.,1000.,101),interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.PlaneParallel)
    viewing=sk.ViewingGeometry()
    # SASKTRAN's relative solar azimuth is Sun relative to viewing plane;
    # this project's absolute LOS azimuth is view relative to Sun at azimuth 0.
    viewing.add_ray(sk.SolarAnglesObserverLocation(.6,-az,-.5,1000.))
    at=sk.Atmosphere(geometry,config,wavelengths_nm=np.array([550.]),calculate_derivatives=False)
    at.temperature_k=np.full(101,288.15);at.pressure_pa=np.full(101,101325.)
    density=101325/(1.380649e-23*288.15)
    at['rayleigh']=sk.constituent.Rayleigh(method='manual',wavelengths_nm=np.array([550.]),
        xs=np.array([depth/1000/density]),king_factor=np.array([1.]))
    at['surface']=sk.constituent.LambertianSurface(.3)
    stokes=sk.Engine(config,geometry,viewing).calculate_radiance(at).radiance.values.ravel()
    cases.append({'tau':depth,'view_azimuth_rad':az,'view_mu':-.5,'sun_mu':.6,'albedo':.3,'stokes':stokes.tolist()})
mie=[]
for x,n,k in [(.01,1.33,0.),(1.,1.5,.01),(10.,1.5,.1)]:
    qext,qsca,qback,g=miepython.efficiencies_mx(complex(n,k),x)
    mu=np.array([-.8,0.,.3,.9]);matrix=miepython.phase_matrix(complex(n,k),x,mu,norm='4pi')
    # miepython uses the opposite complex-amplitude/V convention. Store its
    # native matrix, then compare the invariant I/Q block and |U,V coupling|.
    mie.append({'x':x,'n':n,'k':k,'qext':float(qext),'qsca':float(qsca),'g':float(g),
                'mu':mu.tolist(),'matrix_native':matrix.tolist()})
out={'sources':{'sasktran2':'2026.8.2, https://sasktran2.readthedocs.io/en/latest/examples/coulson.html',
                'miepython':'3.0.2, https://miepython.readthedocs.io/'},
     'description':'Independent vector discrete-ordinate Rayleigh solutions and Lorenz-Mie sphere coefficients',
     'rayleigh':cases,'mie':mie}
root=Path(__file__).resolve().parents[2]
(root/'tests/reference_data/vector_reference.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(cases,indent=2))
