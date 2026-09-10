"""Scalar backward Monte Carlo transfer through concentric spherical shells.

Method family: Emde & Mayer (2007), ACP 7, 2259-2270,
https://doi.org/10.5194/acp-7-2259-2007. Independent implementation, not MYSTIC.
Thermal first-flight integrals are analytic; solar/lunar sources use local
estimates. Implicit absorption and unbiased Russian roulette retain all orders.
"""
import numpy as np
from ..geometry.spherical_shells import segments, direction
from ..constants import EARTH_RADIUS_KM, SOLAR_SOLID_ANGLE_SR, LUNAR_SOLID_ANGLE_SR, SOLAR_ANGULAR_RADIUS_RAD
from ..spectral import planck
from ..types import RadianceResult
from .surface import surface_emissivity
from .radiance import solar_irradiance, lunar_irradiance


def phase(cosine, fraction, g):
    """Normalized scalar HG/Rayleigh mixture, integral over 4pi = 4pi."""
    return fraction*(1-g*g)/(1+g*g-2*g*cosine)**1.5 + (1-fraction)*.75*(1+cosine*cosine)


def rotate(axis, cosine, azimuth):
    axis = np.asarray(axis,float)
    helper = np.zeros_like(axis); helper[:, 2] = 1.
    helper[np.abs(axis[:,2]) > .9] = [1.,0.,0.]
    u = np.cross(axis, helper); u /= np.linalg.norm(u, axis=1)[:,None]
    v = np.cross(axis, u)
    return cosine[:,None]*axis + np.sqrt(np.maximum(1-cosine*cosine, 0.))[:,None]*(np.cos(azimuth)[:,None]*u+np.sin(azimuth)[:,None]*v)


def sample_phase(axis, fraction, g, rng):
    n = len(axis); aerosol = rng.random(n) < fraction
    u = rng.random(n); c = 2*u-1
    hg = aerosol & (np.abs(g) > 1e-7)
    c[hg] = (1+g[hg]**2-((1-g[hg]**2)/(1-g[hg]+2*g[hg]*u[hg]))**2)/(2*g[hg])
    pending = np.flatnonzero(~aerosol)
    while len(pending):
        candidate = rng.uniform(-1,1,len(pending))
        accepted = rng.random(len(pending)) < (1+candidate**2)/2
        c[pending[accepted]] = candidate[accepted]
        pending = pending[~accepted]
    return rotate(axis, np.clip(c,-1,1), rng.uniform(0,2*np.pi,n))


def beam_transmission(p, body, radii, extinction):
    d = np.broadcast_to(body, p.shape)
    edges, ids, _, ground = segments(p,d,radii)
    tau = np.sum(np.diff(edges)*extinction[ids],axis=1)
    return np.where(ground, 0., np.exp(-tau))


def _solve_batch(radii, extinction, omega, fraction, g, *, position, look,
                 planck_layers=None, surface_emission=0., surface_albedo=0.,
                 beams=(), photons=4096, seed=0, max_events=256,
                 distance=np.inf, finite=False):
    """Return means and standard errors: atmospheric, ground, then beam origins.

    Coefficients are per km, bottom-up. Each beam is (unit direction toward
    source, normal irradiance). Statistics use independent complete histories,
    including covariance between their source components in the total error.
    """
    radii, extinction, omega, fraction, g = [np.asarray(x,float) for x in (radii,extinction,omega,fraction,g)]
    if len(radii) != len(extinction)+1 or np.any(np.diff(radii)<=0):
        raise ValueError('Shell radii must increase, with one coefficient per shell')
    if not all(np.isfinite(x).all() for x in (radii,extinction,omega,fraction,g)) or np.any(extinction<0) or np.any((omega<0)|(omega>1)) or np.any((fraction<0)|(fraction>1)) or np.any(np.abs(g)>=1):
        raise ValueError('Invalid spherical optical coefficients')
    if not 0 <= surface_albedo <= 1 or photons < 2:
        raise ValueError('Require surface albedo in [0,1] and at least two photons')
    blackbody = np.zeros_like(extinction) if planck_layers is None else np.asarray(planck_layers,float)
    position, look = np.asarray(position,float), np.asarray(look,float)
    if not np.isfinite(position).all() or not np.isclose(np.linalg.norm(look),1.) or not radii[0]-1e-7 <= np.linalg.norm(position) <= radii[-1]+1e-7:
        raise ValueError("Require a position within the shells and a unit look direction")
    rng = np.random.default_rng(seed)
    values = np.zeros((photons,2+len(beams)))
    active = np.arange(photons); weight = np.ones(photons)
    p = np.broadcast_to(position,(photons,3)).copy(); d = np.broadcast_to(look,p.shape).copy()
    for event in range(max_events):
        edges, ids, end, ground = segments(p,d,radii,distance if event==0 else np.inf)
        if event==0 and finite:
            ground[:] = False
        ds = np.diff(edges); tau = extinction[ids]*ds
        cumulative = np.cumsum(tau,axis=1)
        transmission = np.exp(-cumulative[:,-1])
        start_t = np.exp(-np.column_stack([np.zeros(len(p)), cumulative[:,:-1]]))
        values[active,0] += weight*np.sum(start_t*-np.expm1(-tau)*(1-omega[ids])*blackbody[ids],axis=1)
        values[active,1] += weight*transmission*ground*surface_emission
        endpoint = p+end[:,None]*d
        for b,(body,beam) in enumerate(beams):
            ix = np.flatnonzero(ground)
            if len(ix) and surface_albedo > 0:
                normal = endpoint[ix]/np.linalg.norm(endpoint[ix],axis=1)[:,None]
                mu = np.maximum(normal@body,0.)
                direct = beam_transmission(endpoint[ix],body,radii,extinction)
                values[active[ix],2+b] += weight[ix]*transmission[ix]*surface_albedo/np.pi*beam*mu*direct
        optical_sample = rng.exponential(size=len(p))
        collision = optical_sample < cumulative[:,-1]
        scatter = np.flatnonzero(collision)
        if len(scatter):
            interval = np.sum(cumulative[scatter] <= optical_sample[scatter,None],axis=1)
            layer = ids[scatter,interval]
            before = np.where(interval>0,cumulative[scatter,np.maximum(interval-1,0)],0.)
            s = edges[scatter,interval]+(optical_sample[scatter]-before)/extinction[layer]
            p[scatter] += s[:,None]*d[scatter]
            weight[scatter] *= omega[layer]
            for b,(body,beam) in enumerate(beams):
                direct = beam_transmission(p[scatter],body,radii,extinction)
                angular = phase(d[scatter]@body,fraction[layer],g[layer])
                values[active[scatter],2+b] += weight[scatter]*beam/(4*np.pi)*angular*direct
            d[scatter] = sample_phase(d[scatter],fraction[layer],g[layer],rng)
        bounce = np.flatnonzero(~collision & ground)
        if len(bounce):
            p[bounce] = endpoint[bounce]
            normal = p[bounce]/np.linalg.norm(p[bounce],axis=1)[:,None]
            d[bounce] = rotate(normal,np.sqrt(rng.random(len(bounce))),rng.uniform(0,2*np.pi,len(bounce)))
            weight[bounce] *= surface_albedo
        keep = (collision | ground) & (weight > 0)
        # Survival probability proportional to small weight; no fixed-order cutoff.
        small = keep & (weight < .001)
        survive = rng.random(len(p)) < weight/.001
        keep[small] &= survive[small]
        weight[small & keep] = .001
        active, p, d, weight = active[keep],p[keep],d[keep],weight[keep]
        if not len(active):
            return values.mean(axis=0), values.std(axis=0,ddof=1)/np.sqrt(photons), values.sum(axis=1).std(ddof=1)/np.sqrt(photons)
    raise RuntimeError('Spherical MC event limit reached; increase spherical_max_events (no truncated result returned)')



def solve_sphere(radii, extinction, omega, fraction, g, **kwargs):
    """Bounded-memory batches; combine complete-history means and variances."""
    count = kwargs.pop('photons',4096)
    seed = kwargs.pop('seed',0)
    if type(count) is not int or count < 2:
        raise ValueError('photons must be an integer of at least two')
    batches = (count+4095)//4096
    sizes = [count//batches+(i<count%batches) for i in range(batches)]
    seeds = np.random.SeedSequence(seed).spawn(batches)
    mean = None; m2 = None; total_m2 = 0.; done = 0
    for n, child in zip(sizes,seeds):
        value, error, total_error = _solve_batch(radii,extinction,omega,fraction,g,photons=n,seed=child,**kwargs)
        if mean is None:
            mean = value; m2 = error**2*n*(n-1); total_m2 = total_error**2*n*(n-1)
        else:
            delta = value-mean
            m2 += error**2*n*(n-1)+delta**2*done*n/(done+n)
            total_m2 += total_error**2*n*(n-1)+delta.sum()**2*done*n/(done+n)
            mean += delta*n/(done+n)
        done += n
    return mean, np.sqrt(m2/(count*(count-1))), np.sqrt(total_m2/(count*(count-1)))


def compute_spherical_scattering(*, scenario, grid, profile, path, optics, column, column_optics, transmittance, sun, moon):
    names = ('path_thermal','surface_emitted','surface_reflected_solar','surface_reflected_lunar',
             'solar_single_scatter','lunar_single_scatter','direct_solar','direct_lunar','source_emitted',
             'solar_multiple_scatter','lunar_multiple_scatter')
    components = {name:np.zeros(grid.n) for name in names}
    errors = {name:np.zeros(grid.n) for name in ('path_thermal','surface_emitted','solar_multiple_scatter','lunar_multiple_scatter','total')}
    thickness = np.array([l.ds_km for l in column.layers])
    radii = EARTH_RADIUS_KM+np.r_[column.layers[0].z_bottom_km,[l.z_top_km for l in column.layers]]
    ext = column_optics.tau_total/thickness[:,None]
    scat = column_optics.tau_aerosol_sca+column_optics.tau_rayleigh
    omega = np.divide(scat,column_optics.tau_total,out=np.zeros_like(scat),where=column_optics.tau_total>0)
    fraction = np.divide(column_optics.tau_aerosol_sca,scat,out=np.zeros_like(scat),where=scat>0)
    blackbody = np.array([planck(grid.wavenumber_cm1,l.t_k) for l in column.layers])
    eps = surface_emissivity(scenario.surface_type,incidence_deg=0.,override=scenario.surface_emissivity)
    skin = scenario.surface_skin_temp_k or scenario.ground_temp_k or float(profile.temperature_k[0])
    surface_b = eps*planck(grid.wavenumber_cm1,skin)
    position = np.array([0.,0.,EARTH_RADIUS_KM+scenario.sensor_alt_km])
    look = direction(scenario.sensor_zenith_deg,scenario.sensor_azimuth_deg)
    # With no date the ephemeris returns placeholders, not a physical night sky.
    beams = []
    if scenario.include_solar and scenario.local_datetime is not None:
        beams.append(('solar_multiple_scatter',sun,solar_irradiance(grid,sun.distance_factor),'direct_solar',SOLAR_SOLID_ANGLE_SR,np.rad2deg(SOLAR_ANGULAR_RADIUS_RAD)))
    if scenario.include_lunar and scenario.local_datetime is not None:
        thermal, reflected = lunar_irradiance(grid,moon)
        beams.append(('lunar_multiple_scatter',moon,thermal+reflected,'direct_lunar',LUNAR_SOLID_ANGLE_SR,.2586))
    finite = not (path.hits_ground or path.reaches_space) or scenario.source_temp_k is not None
    for k in range(grid.n):
        sources = [(direction(body.zenith_deg,body.azimuth_deg),irr[k]) for _,body,irr,*_ in beams]
        mean, stderr, total_error = solve_sphere(radii,ext[:,k],omega[:,k],fraction[:,k],column_optics.aerosol_asymmetry[:,k],
            position=position,look=look,planck_layers=blackbody[:,k],surface_emission=surface_b[k],surface_albedo=1-eps,
            beams=sources,photons=scenario.spherical_photons,seed=scenario.spherical_seed+k,
            max_events=scenario.spherical_max_events,distance=path.total_length_km,finite=finite)
        for j,name in enumerate(['path_thermal','surface_emitted']+[b[0] for b in beams]):
            components[name][k],errors[name][k] = mean[j],stderr[j]
        errors['total'][k] = total_error
        if path.reaches_space and not finite:
            for _,body,irr,direct,solid_angle,radius in beams:
                if body.scattering_angle_deg <= radius:
                    components[direct][k] = irr[k]/solid_angle*beam_transmission(position[None,:],direction(body.zenith_deg,body.azimuth_deg),radii,ext[:,k])[0]
    if scenario.source_temp_k is not None:
        components['source_emitted'] = scenario.source_emissivity*planck(grid.wavenumber_cm1,scenario.source_temp_k)*transmittance.total
    return RadianceResult(grid=grid,total=sum(components.values()),sampling_standard_error=errors,**components)
