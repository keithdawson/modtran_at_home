"""Spherical refractive characteristics using the invariant b = n r sin(zenith).

Between levels we interpolate n*r, not n. This has a useful exact path-length
primitive and avoids fake total reflections introduced by constant-index shells.
Nonmonotonic n*r (ducts/multiple images) is deliberately rejected.
"""
from pathlib import Path
import json
import numpy as np
import yaml
from ..types import RayTrace
from .spherical_shells import segments

_GX, _GW = np.polynomial.legendre.leggauss(12)


def reference_index(wavelength_um):
    """Mathar 2007 dry reference air, 288.15 K, 101325 Pa, 370 ppm CO2.

    HOOK: Do not extend these tables across absorption gaps by extrapolation.
    A future dispersive ray solver must select n(lambda) per spectral point,
    rebuild trajectories AND solar shooting, and preserve L/n^2 transport.
    """
    for file in sorted((Path(__file__).parent/'refractivity_data').glob('*.yml')):
        data=yaml.safe_load(file.read_text(encoding='utf-8'))
        table=np.array([list(map(float,line.split())) for line in data['DATA'][0]['data'].splitlines() if line.strip()])
        if table[0,0] <= wavelength_um <= table[-1,0]:
            return float(np.interp(wavelength_um,table[:,0],table[:,1]))
    raise ValueError('Refraction reference wavelength must lie in a Mathar table: 1.3–2.5, 2.8–4.2 or 4.35–5.2 um')


def profile_index(scenario, profile, heights):
    """Geometry-index hook. Supplied n(z) overrides the dry density approximation.

    File contract v1: {version:1, altitude_km:[...], refractive_index:[...]}, JSON.
    Altitudes are ASL, not above ground; indices are absolute (NOT refractivity
    N=1e6*(n-1)). Require full coverage. No silent extrapolation, unit guessing,
    humidity formula extrapolation, or replacement of missing values by unity.
    Sensor calibration files must not be loaded here; this is atmospheric data.
    """
    if scenario.refractive_index_file:
        data=json.loads(Path(scenario.refractive_index_file).read_text(encoding='utf-8'))
        z=np.asarray(data['altitude_km'],float); n=np.asarray(data['refractive_index'],float)
        if data.get('version')!=1 or z.ndim!=1 or z.shape!=n.shape or len(z)<2 or not np.isfinite(z).all() or not np.isfinite(n).all() or np.any(np.diff(z)<=0) or np.any((n<1)|(n>1.01)) or z[0]>min(heights) or z[-1]<max(heights):
            raise ValueError('Invalid or incomplete refractive-index profile v1')
        return np.interp(heights,z,n)
    # Dilute-gas density scaling, not the full humid Mathar temperature fit.
    # Standard profiles extend outside the fit's laboratory P/T validity range.
    # Keep this approximation named/documented; real inversions and water-vapor
    # dispersion belong in a measured/modelled index profile supplied above.
    p=np.interp(heights,profile.altitude_km,profile.pressure_mbar)
    t=np.interp(heights,profile.altitude_km,profile.temperature_k)
    return 1+(reference_index(scenario.refraction_reference_um)-1)*(p/1013.25)*(288.15/t)


def index_at(p,radii,index):
    r=np.linalg.norm(p,axis=-1)
    return np.interp(r,radii,radii*index)/r


def advance(p,d,distance,layer,radii,index):
    """Advance within one shell exactly in arc length, quadrature in polar angle."""
    r=np.linalg.norm(p,axis=1); radial=p/r[:,None]
    cosine=np.clip(np.sum(radial*d,axis=1),-1,1)
    tangential=d-cosine[:,None]*radial
    sine=np.linalg.norm(tangential,axis=1)
    tangent=np.divide(tangential,sine[:,None],out=np.zeros_like(d),where=sine[:,None]>1e-14)
    nr=radii*index; a=(nr[layer+1]-nr[layer])/(radii[layer+1]-radii[layer])
    intercept=nr[layer]-a*radii[layer]
    nr0=a*r+intercept; b=nr0*sine
    u0=nr0*cosine; u1=u0+a*distance
    nr1=np.sqrt(u1*u1+b*b); r1=(nr1-intercept)/a
    u=(u0[:,None]+u1[:,None])/2+(u1-u0)[:,None]/2*_GX
    nrq=np.sqrt(u*u+b[:,None]**2)
    angle=(u1-u0)/2*np.sum(_GW*b[:,None]/((nrq-intercept[:,None])*nrq),axis=1)
    radial1=np.cos(angle)[:,None]*radial+np.sin(angle)[:,None]*tangent
    tangent1=-np.sin(angle)[:,None]*radial+np.cos(angle)[:,None]*tangent
    d1=(u1/nr1)[:,None]*radial1+(b/nr1)[:,None]*tangent1
    return r1[:,None]*radial1,d1


def trace(p,d,radii,index=None,limit=np.inf):
    """Trace all rays to ground/space/range; homogeneous optics per curved segment."""
    p=np.atleast_2d(p).astype(float); d=np.broadcast_to(d,p.shape).astype(float)
    if index is None or np.all(np.asarray(index)==1):
        edges,ids,end,ground=segments(p,d,radii,limit)
        return RayTrace(np.diff(edges),ids,p[:,None,:]+edges[:,:-1,None]*d[:,None,:],
                        np.broadcast_to(d[:,None,:],(*ids.shape,3)).copy(),p+end[:,None]*d,d.copy(),ground,~ground & (end<np.asarray(limit)+1e-8))
    index=np.asarray(index,float); nr=index*radii
    if index.shape!=radii.shape or not np.isfinite(index).all() or np.any(index<1) or np.any(np.diff(nr)<=0):
        raise ValueError('Refraction requires finite n>=1 and increasing n*r: ducting/multiple images are unsupported')
    n=len(p); q=p.copy(); v=d.copy(); remaining=np.broadcast_to(limit,(n,)).astype(float).copy()
    r=np.linalg.norm(q,axis=1); inward=np.sum(q*v,axis=1)<0
    layer=np.clip(np.searchsorted(radii,r,side='right')-1,0,len(radii)-2)
    active=np.ones(n,bool); ground=np.zeros(n,bool); space=np.zeros(n,bool)
    lengths=[]; layers=[]; starts=[]; directions=[]
    for _ in range(2*len(radii)+4):
        r=np.linalg.norm(q,axis=1); cosine=np.sum(q*v,axis=1)/r
        a=np.diff(nr)[layer]/np.diff(radii)[layer]; c=nr[layer]-a*radii[layer]
        nr0=a*r+c; b=nr0*np.sqrt(np.maximum(1-cosine*cosine,0)); u0=nr0*cosine
        inward=(cosine< -1e-12)
        can_inner=inward & (b<nr[layer])
        # A tangent within the shell needs no special split: u passes smoothly
        # through zero and the exact arc-length primitive covers both halves.
        boundary=np.where(can_inner,nr[layer],nr[layer+1])
        uend=np.sqrt(np.maximum(boundary**2-b*b,0))*np.where(can_inner,-1.,1.)
        ds=np.maximum((uend-u0)/a,0.)
        ds=np.where(active,np.minimum(ds,remaining),0.)
        lengths.append(ds.copy()); layers.append(layer.copy()); starts.append(q.copy()); directions.append(v.copy())
        q1,v1=advance(q,v,ds,layer,radii,index)
        q[active]=q1[active]; v[active]=v1[active]; remaining-=ds
        reached=active & (remaining>1e-8)
        lower=reached & can_inner; upper=reached & ~can_inner
        ground |= lower & (layer==0); space |= upper & (layer==len(radii)-2)
        active &= ~(ground|space|(remaining<=1e-8))
        layer=np.clip(layer+np.where(lower,-1,1),0,len(radii)-2)
        if not np.any(active):
            return RayTrace(np.array(lengths).T,np.array(layers).T,np.swapaxes(starts,0,1),np.swapaxes(directions,0,1),q,v,ground,space)
    raise RuntimeError('Refractive ray did not leave the atmosphere; check the profile for ducting')


def connect_source(p,body,radii,index,extinction):
    """Shoot toward a distant source, returning apparent direction and beam factor.

    Factor includes extinction AND refractive angular focusing. It multiplies
    vacuum normal irradiance, not horizontal irradiance. Neglecting the angular
    Jacobian while bending rays would violate etendue/flux conservation.
    Only the unique non-ducting branch is supported. No astronomical pointing
    correction belongs in the sensor adapter: it must use this apparent ray.
    """
    p=np.atleast_2d(p); r=np.linalg.norm(p,axis=1); normal=p/r[:,None]
    cosine=np.clip(normal@body,-1.,1.); target=np.arccos(cosine)
    transverse=body-cosine[:,None]*normal
    norm=np.linalg.norm(transverse,axis=1)
    fallback=np.cross(normal,np.broadcast_to([0.,1.,0.],normal.shape))
    transverse=np.divide(transverse,norm[:,None],out=fallback.copy(),where=norm[:,None]>1e-12)
    local_n=index_at(p,radii,index)
    critical=np.arcsin(np.clip(radii[0]*index[0]/(r*local_n),0.,1.))
    hi=np.pi-critical-1e-8; lo=np.zeros(len(p))
    def evaluate(angle):
        direction=np.cos(angle)[:,None]*normal+np.sin(angle)[:,None]*transverse
        rays=trace(p,direction,radii,index)
        true=np.arccos(np.clip(np.sum(rays.exit_direction*normal,axis=1),-1,1))
        return true,rays,direction
    horizon,_,_=evaluate(hi)
    visible=target<=horizon
    for _ in range(24):
        mid=(lo+hi)/2; actual,_,_=evaluate(mid)
        lo=np.where(actual<target,mid,lo); hi=np.where(actual>=target,mid,hi)
    angle=(lo+hi)/2; actual,rays,direction=evaluate(angle)
    h=1e-5; plus,_,_=evaluate(np.minimum(angle+h,np.pi-critical-1e-9))
    minus,_,_=evaluate(np.maximum(angle-h,0.))
    span=np.minimum(angle+h,np.pi-critical-1e-9)-np.maximum(angle-h,0.)
    derivative=(plus-minus)/span
    jacobian=np.divide(np.sin(angle),np.sin(target)*derivative,out=np.ones(len(p)),where=np.sin(target)>1e-7)
    jacobian=np.where(target<1e-7,1/derivative**2,jacobian)
    tau=np.sum(rays.lengths*extinction[rays.layer],axis=1)
    factor=np.where(visible & rays.reaches_space,local_n**2*jacobian*np.exp(-tau),0.)
    if np.any(~np.isfinite(factor)) or np.any(factor<0):
        raise RuntimeError('Refractive solar shooting produced an invalid focusing factor')
    return direction,factor
