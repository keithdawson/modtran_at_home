"""Scalar backward Monte Carlo transfer through concentric spherical shells.

Method family: Emde & Mayer (2007), ACP 7, 2259-2270,
https://doi.org/10.5194/acp-7-2259-2007. Independent implementation, not MYSTIC.
Thermal first-flight integrals are analytic; solar/lunar sources use local
estimates. Implicit absorption and unbiased Russian roulette retain all orders.
"""
import numpy as np
from ..geometry.spherical_shells import segments, direction
from ..geometry.refraction import trace, advance, index_at, connect_source
from .polarization import scattering_matrix, transport, MiePhase, surface_matrix
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
                 distance=np.inf, finite=False, refractive_index=None, polarized=False, mie=None, surface_index=None, beam_solid_angles=None):
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
    values = np.zeros((photons,2+len(beams),4) if polarized else (photons,2+len(beams)))
    response = np.broadcast_to(np.eye(4),(photons,4,4)).copy()
    def add(rows,component,amplitude,mapping):
        if polarized:
            values[active[rows],component] += amplitude[:,None]*mapping[:,:,0]
        else:
            values[active[rows],component] += amplitude
    def source_path(points,body):
        if refractive_index is None:
            return np.broadcast_to(body,points.shape),beam_transmission(points,body,radii,extinction)
        return connect_source(points,body,radii,refractive_index,extinction)
    active = np.arange(photons); weight = np.ones(photons)
    p = np.broadcast_to(position,(photons,3)).copy(); d = np.broadcast_to(look,p.shape).copy()
    for event in range(max_events):
        rays=trace(p,d,radii,refractive_index,distance if event==0 and finite else np.inf)
        ds,ids,ground=rays.lengths,rays.layer,rays.hits_ground.copy()
        if event==0 and finite:
            ground[:] = False
        tau=extinction[ids]*ds; cumulative=np.cumsum(tau,axis=1)
        transmission=np.exp(-cumulative[:,-1])
        start_t=np.exp(-np.column_stack([np.zeros(len(p)),cumulative[:,:-1]]))
        nstart=np.ones(len(p)) if refractive_index is None else index_at(p,radii,refractive_index)
        # Reduced-radiance invariant L/n^2. Thermal B is specified as local
        # radiance; within each optical shell use n at its segment start.
        nsegment=np.ones_like(ds) if refractive_index is None else index_at(rays.starts,radii,refractive_index)
        local_thermal=start_t*-np.expm1(-tau)*(1-omega[ids])*blackbody[ids]*(nstart[:,None]/nsegment)**2
        add(np.arange(len(p)),0,weight*np.sum(local_thermal,axis=1),response)
        endpoint=rays.endpoint
        end_response=response@transport(d,rays.exit_direction) if polarized else response
        nend=np.ones(len(p)) if refractive_index is None else index_at(endpoint,radii,refractive_index)
        if surface_index is None:
            add(np.arange(len(p)),1,weight*transmission*ground*surface_emission*(nstart/nend)**2,end_response)
            for b,(body,beam) in enumerate(beams):
                ix=np.flatnonzero(ground)
                if len(ix) and surface_albedo>0:
                    normal=endpoint[ix]/np.linalg.norm(endpoint[ix],axis=1)[:,None]
                    apparent,direct=source_path(endpoint[ix],body)
                    mu=np.maximum(np.sum(normal*apparent,axis=1),0.)
                    add(ix,2+b,weight[ix]*transmission[ix]*(nstart[ix]/nend[ix])**2*surface_albedo/np.pi*beam*mu*direct,end_response[ix])
        else:
            if not polarized: raise ValueError('Fresnel surface requires polarized transfer')
            ix=np.flatnonzero(ground)
            if len(ix):
                normal=endpoint[ix]/np.linalg.norm(endpoint[ix],axis=1)[:,None]
                reflected,matrix,emission=surface_matrix(rays.exit_direction[ix],normal,surface_index)
                scale=weight[ix]*transmission[ix]*(nstart[ix]/nend[ix])**2
                values[active[ix],1]+=scale[:,None]*surface_emission*np.einsum('nij,nj->ni',end_response[ix],emission)
                # A smooth mirror has a delta BRDF: a diffuse E*cos/pi estimate
                # is WRONG. Evaluate the finite source disc along the reflected
                # characteristic, retaining its solid-angle/radiance conversion.
                specular=trace(endpoint[ix],reflected,radii,refractive_index)
                attenuation=np.exp(-np.sum(specular.lengths*extinction[specular.layer],axis=1))
                tmap=transport(reflected,specular.exit_direction)
                mapping=end_response[ix]@matrix@tmap
                for b,(body,beam) in enumerate(beams):
                    solid=6.8e-5 if beam_solid_angles is None else beam_solid_angles[b]
                    radius=np.arccos(1-solid/(2*np.pi))
                    visible=(specular.exit_direction@body>=np.cos(radius)) & specular.reaches_space
                    if refractive_index is not None:
                        attenuation_local=attenuation*(nend[ix]/index_at(specular.endpoint,radii,refractive_index))**2
                    else: attenuation_local=attenuation
                    add(ix,2+b,scale*beam/solid*attenuation_local*visible,mapping)
        optical_sample = rng.exponential(size=len(p))
        collision = optical_sample < cumulative[:,-1]
        scatter = np.flatnonzero(collision)
        if len(scatter):
            interval = np.sum(cumulative[scatter] <= optical_sample[scatter,None],axis=1)
            layer = ids[scatter,interval]
            before = np.where(interval>0,cumulative[scatter,np.maximum(interval-1,0)],0.)
            local_distance=(optical_sample[scatter]-before)/extinction[layer]
            old=d[scatter].copy()
            starts=rays.starts[scatter,interval]; local_dir=rays.directions[scatter,interval]
            if refractive_index is None:
                p[scatter]=starts+local_distance[:,None]*local_dir
                d[scatter]=local_dir
            else:
                p[scatter],d[scatter]=advance(starts,local_dir,local_distance,layer,radii,refractive_index)
                weight[scatter]*=(nstart[scatter]/index_at(p[scatter],radii,refractive_index))**2
            if polarized:
                response[scatter]=response[scatter]@transport(old,d[scatter])
            weight[scatter] *= omega[layer]
            for b,(body,beam) in enumerate(beams):
                apparent,direct=source_path(p[scatter],body)
                if polarized:
                    matrix=scattering_matrix(d[scatter],apparent,fraction[layer],g[layer],mie)
                    add(scatter,2+b,weight[scatter]*beam/(4*np.pi)*direct,response[scatter]@matrix)
                else:
                    angular=phase(np.sum(d[scatter]*apparent,axis=1),fraction[layer],g[layer])
                    add(scatter,2+b,weight[scatter]*beam/(4*np.pi)*angular*direct,response[scatter])
            if polarized:
                # Preconditioned backward sampling: proposal P11/4pi, accept
                # (H M)00/(2 H00 P11). H00 remains one after normalization.
                # The ratio is in [0,2] for a physical phase matrix. Sampling
                # P11 alone and merely multiplying signed Q/U weights can have
                # catastrophic variance in thick Rayleigh media (Garcia Munoz &
                # Mills 2015, doi:10.1051/0004-6361/201424042).
                pending=scatter.copy(); pending_layers=layer.copy()
                for attempt in range(100):
                    if not len(pending): break
                    candidate=sample_phase(d[pending],fraction[pending_layers],g[pending_layers],rng)
                    if mie is not None:
                        use=rng.random(len(pending))<fraction[pending_layers]
                        candidate[use]=rotate(d[pending[use]],mie.sample(rng,int(use.sum())),rng.uniform(0,2*np.pi,int(use.sum())))
                        candidate[~use]=sample_phase(d[pending[~use]],np.zeros((~use).sum()),g[pending_layers[~use]],rng)
                    matrix=scattering_matrix(d[pending],candidate,fraction[pending_layers],g[pending_layers],mie)
                    product=response[pending]@matrix
                    ratio=product[:,0,0]/matrix[:,0,0]
                    if np.any(ratio < -1e-9) or np.any(ratio>2+1e-9):
                        raise RuntimeError('Nonphysical polarization importance weights')
                    accepted=rng.random(len(pending))<np.clip(ratio/2,0,1)
                    j=pending[accepted]
                    response[j]=product[accepted]/product[accepted,0,0,None,None]
                    d[j]=candidate[accepted]
                    pending,pending_layers=pending[~accepted],pending_layers[~accepted]
                if len(pending): raise RuntimeError('Polarized angular sampler failed to converge')
            else:
                d[scatter]=sample_phase(d[scatter],fraction[layer],g[layer],rng)
        bounce=np.flatnonzero(~collision & ground)
        if len(bounce):
            p[bounce]=endpoint[bounce]
            normal=p[bounce]/np.linalg.norm(p[bounce],axis=1)[:,None]
            weight[bounce]*=(nstart[bounce]/nend[bounce])**2
            if surface_index is None:
                d[bounce]=rotate(normal,np.sqrt(rng.random(len(bounce))),rng.uniform(0,2*np.pi,len(bounce)))
                weight[bounce]*=surface_albedo
                if polarized:
                    response[bounce]=0.
                    response[bounce,:,0]=end_response[bounce,:,0]
            else:
                d[bounce],matrix,_=surface_matrix(rays.exit_direction[bounce],normal,surface_index)
                product=end_response[bounce]@matrix
                intensity=product[:,0,0]
                weight[bounce]*=intensity
                response[bounce]=np.divide(product,intensity[:,None,None],out=np.zeros_like(product),where=intensity[:,None,None]>0)
        keep = (collision | ground) & (weight > 0)
        # Survival probability proportional to small weight; no fixed-order cutoff.
        small = keep & (weight < .001)
        survive = rng.random(len(p)) < weight/.001
        keep[small] &= survive[small]
        weight[small & keep] = .001
        response=response[keep]
        active, p, d, weight = active[keep],p[keep],d[keep],weight[keep]
        if not len(active):
            return values.mean(axis=0), values.std(axis=0,ddof=1)/np.sqrt(photons), values.sum(axis=1).std(axis=0,ddof=1)/np.sqrt(photons)
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
            total_m2 += total_error**2*n*(n-1)+delta.sum(axis=0)**2*done*n/(done+n)
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
    refractive_index=None
    if scenario.refraction:
        from ..geometry.refraction import profile_index
        refractive_index=profile_index(scenario,profile,radii-EARTH_RADIUS_KM)
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
    vector={name:np.zeros((grid.n,4)) for name in names} if scenario.polarization else {}
    vector_errors={name:np.zeros((grid.n,4)) for name in errors} if scenario.polarization else {}
    surface_index=complex(scenario.surface_index_real,scenario.surface_index_imag) if scenario.surface_polarization=='fresnel' else None
    if surface_index is not None: surface_b=planck(grid.wavenumber_cm1,skin)
    for k in range(grid.n):
        sources = [(direction(body.zenith_deg,body.azimuth_deg),irr[k]) for _,body,irr,*_ in beams]
        mie=MiePhase(2*np.pi*scenario.mie_radius_um/grid.wavelength_um[k],complex(scenario.mie_index_real,scenario.mie_index_imag)) if scenario.aerosol_polarization=='mie_sphere' else None
        mean, stderr, total_error = solve_sphere(radii,ext[:,k],omega[:,k],fraction[:,k],column_optics.aerosol_asymmetry[:,k],
            position=position,look=look,planck_layers=blackbody[:,k],surface_emission=surface_b[k],surface_albedo=1-eps,
            beams=sources,photons=scenario.spherical_photons,seed=scenario.spherical_seed+k,
            max_events=scenario.spherical_max_events,distance=path.total_length_km,finite=finite,
            refractive_index=refractive_index,polarized=scenario.polarization,mie=mie,
            surface_index=surface_index,beam_solid_angles=[b[4] for b in beams])
        for j,name in enumerate(['path_thermal','surface_emitted']+[b[0] for b in beams]):
            if scenario.polarization:
                vector[name][k],vector_errors[name][k]=mean[j],stderr[j]
                components[name][k],errors[name][k]=mean[j,0],stderr[j,0]
            else: components[name][k],errors[name][k] = mean[j],stderr[j]
        errors['total'][k] = total_error[0] if scenario.polarization else total_error
        if scenario.polarization: vector_errors['total'][k]=total_error
        if path.reaches_space and not finite:
            for _,body,irr,direct,solid_angle,radius in beams:
                body_direction=direction(body.zenith_deg,body.azimuth_deg)
                if refractive_index is None:
                    visible=body.scattering_angle_deg<=radius
                    attenuation=beam_transmission(position[None,:],body_direction,radii,ext[:,k])[0]
                else:
                    ray=trace(position,look,radii,refractive_index)
                    visible=(ray.exit_direction[0]@body_direction>=np.cos(np.deg2rad(radius))) and ray.reaches_space[0]
                    attenuation=np.exp(-np.sum(ray.lengths*ext[ray.layer,k]))*(index_at(position[None,:],radii,refractive_index)[0]/index_at(ray.endpoint,radii,refractive_index)[0])**2
                if visible: components[direct][k]=irr[k]/solid_angle*attenuation
    if scenario.source_temp_k is not None:
        components['source_emitted'] = scenario.source_emissivity*planck(grid.wavenumber_cm1,scenario.source_temp_k)*transmittance.total
    if scenario.polarization:
        for name in ('direct_solar','direct_lunar','source_emitted'): vector[name][:,0]=components[name]
        vector['total']=sum(vector.values())
    result=RadianceResult(grid=grid,total=sum(components.values()),sampling_standard_error=errors,
        stokes=vector,stokes_standard_error=vector_errors,**components)
    if scenario.sensor_response_file:
        from .sensor import load_sensor_response,apply_sensor_response
        result.sensor_signal=apply_sensor_response(grid,vector['total'],load_sensor_response(scenario.sensor_response_file))
    return result


def apply_mie_optics(scenario,column,optics,grid):
    """Match column aerosol extinction at 550 nm, then use consistent Mie optics.

    This preserves the configured aerosol vertical loading/visibility but replaces
    its spectral absorption, scattering and phase. A homogeneous sphere with a
    constant user-specified relative material index is an idealization, not a
    reconstruction of the LOWTRAN particle mixture. Radius is not diameter.
    """
    from .polarization import mie_coefficients
    from dataclasses import replace
    radius=scenario.mie_radius_um; material=complex(scenario.mie_index_real,scenario.mie_index_imag)
    _,_,_,reference=mie_coefficients(2*np.pi*radius/.55,material)
    if reference<=0: raise ValueError('Mie reference extinction must be positive')
    loading=np.array([l.aerosol_od_550nm for l in column.layers])
    scattering=np.zeros_like(optics.tau_aerosol_sca); absorption=scattering.copy(); asymmetry=scattering.copy()
    for k,wavelength in enumerate(grid.wavelength_um):
        x=2*np.pi*radius/wavelength
        _,_,qsca,qext=mie_coefficients(x,material)
        scattering[:,k]=loading*qsca/reference
        absorption[:,k]=loading*max(qext-qsca,0.)/reference
        asymmetry[:,k]=MiePhase(x,material).g
    return replace(optics,tau_aerosol_abs=absorption,tau_aerosol_sca=scattering,aerosol_asymmetry=asymmetry)

