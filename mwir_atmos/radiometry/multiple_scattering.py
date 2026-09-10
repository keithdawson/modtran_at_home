"""DISORT radiance in a complete plane-parallel atmosphere.

Numerical kernel: Ho (2024), JOSS 9(103), 6442, doi:10.21105/joss.06442.
The pinned MIT-licensed PythonicDISORT source is vendored for offline use.
All scattering orders are solved simultaneously by discrete ordinates. This
adapter uses unscaled HG/Rayleigh Legendre moments, with explicit stream control.
It does not add delta-M scaling, polarization, refraction or spherical corrections.
"""
import math
import warnings
import numpy as np

from .._vendor.pythonic_disort import pydisort, subroutines
from ..spectral import planck
from ..types import RadianceResult
from ..constants import SOLAR_SOLID_ANGLE_SR, SOLAR_ANGULAR_RADIUS_RAD, LUNAR_SOLID_ANGLE_SR
from .surface import surface_emissivity
from .radiance import solar_irradiance, lunar_irradiance


def phase_moments(aerosol_scattering, rayleigh_scattering, asymmetry, streams):
    """Unweighted moments: HG beta_l=g**l; Rayleigh beta_2=0.1.

    Weight by scattering optical depth, not extinction or particle abundance.
    """
    aer, ray, g = np.broadcast_arrays(aerosol_scattering, rayleigh_scattering, asymmetry)
    denom = aer + ray
    fraction = np.divide(aer, denom, out=np.zeros_like(denom), where=denom > 0)
    result = fraction[..., None] * g[..., None] ** np.arange(streams)
    result[..., 0] = 1.
    result[..., 2] += (1. - fraction) * .1
    return result


def solve_column(dtau, omega, moments, *, streams=16, mu0=.5, beam=0., phi0=0.,
                 planck_layers=None, surface_emission=0., surface_albedo=0.):
    """Solve one spectral point; optical layers are ordered from top to bottom.

    Sources and boundary radiances use the caller's consistent spectral units.
    The kernel applies (1-omega) to Planck sources itself. Its eigenproblem
    requires omega<1: conservative layers use 1-1e-10, documented and tested.
    """
    dtau, omega = np.asarray(dtau, float), np.asarray(omega, float)
    if np.any(dtau <= 0) or not np.isfinite(dtau).all():
        raise ValueError('DISORT layer extinction must be finite and positive')
    if np.any((omega < 0) | (omega > 1)) or not np.isfinite(omega).all():
        raise ValueError('DISORT single-scattering albedo must lie in [0, 1]')
    if beam > 0 and not 0 < mu0 <= 1:
        raise ValueError('DISORT illumination must be above the plane-parallel horizon')
    sources = np.array([[]]) if planck_layers is None else np.asarray(planck_layers)[:, None]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*close to 1.*')
        return pydisort(np.cumsum(dtau), np.minimum(omega, 1.-1e-10), streams,
                       np.asarray(moments), mu0, beam, phi0,
                       NFourier=streams if beam > 0 else 1,
                       b_pos=surface_emission, BDRF_Fourier_modes=[surface_albedo],
                       s_poly_coeffs=sources, cache_asso_leg='no_mu0')


def horizontal_source(solution, tau, omega, moments, *, phi=0., mu0=.5, beam=0.,
                      phi0=0., planck_value=0.):
    """RTE source at mu=0, evaluated by the solver's angular quadrature.

    This avoids subtracting nearly equal endpoint intensities at the horizon.
    In a homogeneous horizontal slab S is constant and I_path=S*(1-exp(-tau_LOS)).
    """
    mu, _, _, _, intensity = solution
    nphi = 2 * len(mu)
    azimuth = np.arange(nphi) * (2*math.pi/nphi)
    weights = np.tile(subroutines.Gauss_Legendre_quad(len(mu)//2)[1], 2)
    cosine = np.sqrt(1-mu[:, None]**2) * np.cos(phi-azimuth[None, :])
    phase = np.polynomial.legendre.legval(cosine, moments*(2*np.arange(len(moments))+1))
    diffuse = np.asarray(intensity(tau, azimuth)).reshape(len(mu), nphi)
    source = omega * np.sum(phase*diffuse*weights[:, None]) / (2*nphi)
    if beam > 0:
        cosine_beam = np.sqrt(1-mu0**2)*np.cos(phi-phi0)
        phase_beam = np.polynomial.legendre.legval(cosine_beam, moments*(2*np.arange(len(moments))+1))
        source += omega * beam/(4*math.pi)*phase_beam*np.exp(-tau/mu0)
    return float(source + (1-omega)*planck_value)


def integrate_sightline(solution, *, path, optical_depth, heights, tau_edges,
                        omega, moments, mu_view, phi, planck_layers=None,
                        mu0=.5, beam=0., phi0=0., surface_emission=0.,
                        surface_albedo=0., include_surface=False):
    """Formal solution with exact thermal/beam terms and quadrature of diffuse light.

    Sampling the angular field, rather than subtracting endpoint interpolants,
    preserves finite-target and zero-scattering limits. Diffuse integrals split
    into <=0.5 optical-depth intervals with 16-point Gaussian quadrature. Beyond
    32 optical depths from the sensor diffuse attenuation is below 1.3e-14.
    """
    nodes, weights = np.polynomial.legendre.leggauss(16)
    mu, flux_up, flux_down, u0, intensity = solution
    angular_weights = np.tile(subroutines.Gauss_Legendre_quad(len(mu)//2)[1], 2)
    coefficients = moments*(2*np.arange(moments.shape[1])+1)
    if beam > 0:
        nphi = 2*len(mu)
        azimuth = np.arange(nphi)*(2*math.pi/nphi)
        cosine = mu_view*mu[:,None]+np.sqrt(1-mu_view**2)*np.sqrt(1-mu[:,None]**2)*np.cos(phi-azimuth)
        basis = np.polynomial.legendre.legvander(cosine, moments.shape[1]-1)
        phase = np.einsum('lk,ijk->lij', coefficients, basis)
        cosine_beam = -mu_view*mu0 + np.sqrt(1-mu_view**2)*np.sqrt(1-mu0**2)*np.cos(phi-phi0)
        beam_phase = coefficients @ np.polynomial.legendre.legvander(cosine_beam, moments.shape[1]-1).ravel()
    else:
        # Azimuthal average of P_l(cos Theta) is P_l(mu)*P_l(mu').
        phase = (coefficients*np.polynomial.legendre.legvander(mu_view, moments.shape[1]-1)) @ np.polynomial.legendre.legvander(mu, moments.shape[1]-1).T
    before = 0.
    value = 0.
    queries, quadrature_weights, layer_numbers = [], [], []
    for layer, dt in zip(path.layers, optical_depth):
        middle = (layer.z_bottom_km+layer.z_top_km)/2
        bottom_index = min(int(np.searchsorted(heights[1:], middle, side='right')), len(omega)-1)
        number = len(omega)-1-bottom_index
        near = layer.z_bottom_km if mu_view < 0 else layer.z_top_km
        tau_near = float(np.interp(near, heights, tau_edges[::-1]))
        transmission = np.exp(-before)
        if planck_layers is not None:
            value += transmission * -np.expm1(-dt) * (1-omega[number])*planck_layers[number]
        if beam > 0:
            alpha = 1+mu_view/mu0
            if abs(alpha) < 1e-8:
                integral = np.exp(-before-tau_near/mu0)*dt
            elif alpha > 0:
                integral = np.exp(-before-tau_near/mu0)*-np.expm1(-alpha*dt)/alpha
            else:
                tau_far = max(0., tau_near+mu_view*dt)
                integral = np.exp(-before-dt-tau_far/mu0)*-np.expm1(alpha*dt)/(-alpha)
            value += omega[number]*beam*beam_phase[number]/(4*math.pi)*integral
        end = min(float(dt), 32.-before)
        if end > 0 and omega[number] > 0:
            edges = np.linspace(0., end, max(2, int(np.ceil(end/.5))+1))
            x = ((edges[1:]+edges[:-1])[:,None]/2 + np.diff(edges)[:,None]/2*nodes).ravel()
            weights_x = (np.diff(edges)[:,None]/2*weights).ravel()*np.exp(-before-x)*omega[number]
            queries.extend(np.clip(tau_near+mu_view*x, 0., tau_edges[-1]))
            quadrature_weights.extend(weights_x)
            layer_numbers.extend([number]*len(x))
        before += dt
    if queries:
        q = np.asarray(queries)
        numbers = np.asarray(layer_numbers)
        qweights = np.asarray(quadrature_weights)
        if beam > 0:
            field = np.asarray(intensity(q, azimuth)).reshape(len(mu), len(q), nphi)
            # Process each slab without allocating a phase tensor for every query.
            for number in np.unique(numbers):
                select = numbers == number
                source = np.einsum('ip,iqp,i->q', phase[number], field[:,select,:], angular_weights)/(2*nphi)
                value += np.dot(qweights[select], source)
        else:
            field = np.asarray(u0(q)).reshape(len(mu),len(q))
            source = np.sum(phase[numbers].T*field*angular_weights[:,None],axis=0)/2
            value += np.dot(qweights,source)
    if include_surface:
        down_diffuse, down_direct = flux_down(tau_edges[-1])
        boundary = surface_emission+surface_albedo*(down_diffuse+down_direct)/math.pi
        value += np.exp(-before)*float(boundary)
    return float(value)


def compute_multiple_scattering(*, scenario, grid, profile, path, optics, column,
                                column_optics, transmittance, sun, moon):
    """Spectral sensor radiance separated by original source, including feedback.

    Atmospheric and surface thermal fields include all subsequent scatterings
    and ground reflections. Solar/lunar diffuse fields include both atmospheric
    scattering and surface reflection. Direct celestial discs remain separate.
    A finite endpoint supplies no incident background unless source_temp_k is
    specified; its local target emission does not illuminate the global column.
    """
    n = grid.n
    components = {name: np.zeros(n) for name in (
        'path_thermal','surface_emitted','surface_reflected_solar','surface_reflected_lunar',
        'solar_single_scatter','lunar_single_scatter','direct_solar','direct_lunar',
        'source_emitted','solar_multiple_scatter','lunar_multiple_scatter')}
    streams = scenario.disort_streams
    tau_ext = column_optics.tau_total[::-1]
    tau_sca = column_optics.tau_scattering[::-1]
    # Transparent numerical slabs contribute at most 1e-12 optical depth each.
    tau_ext = np.maximum(tau_ext, 1e-12)
    omega = np.clip(tau_sca / tau_ext, 0., 1.)
    temperature = np.array([l.t_k for l in column.layers[::-1]])
    blackbody = planck(grid.wavenumber_cm1, temperature)
    # A Lambertian boundary must use a single hemispherical emissivity, not
    # an observer-angle-dependent Fresnel number for all illumination angles.
    eps = surface_emissivity(scenario.surface_type, incidence_deg=0., override=scenario.surface_emissivity)
    skin = scenario.surface_skin_temp_k
    if skin is None:
        skin = scenario.ground_temp_k
    if skin is None:
        skin = float(profile.temperature_k[0])
    surface_b = eps * planck(grid.wavenumber_cm1, skin)
    heights = np.r_[column.layers[0].z_bottom_km, [l.z_top_km for l in column.layers]]
    view_cos = float(np.cos(np.deg2rad(scenario.sensor_zenith_deg)))
    horizontal = abs(view_cos) < 1e-10
    mu_view = -view_cos  # photon propagation is opposite to the sensor look direction
    phi = np.deg2rad(scenario.sensor_azimuth_deg)
    z_end = scenario.sensor_alt_km + (0. if horizontal else view_cos*path.total_length_km)
    finite = not (path.hits_ground or path.reaches_space) or scenario.source_temp_k is not None
    beams = []
    if scenario.include_solar and sun.is_above_horizon:
        beams.append(('solar_multiple_scatter', sun, solar_irradiance(grid, sun.distance_factor),
                      'direct_solar', SOLAR_SOLID_ANGLE_SR, math.degrees(SOLAR_ANGULAR_RADIUS_RAD)))
    if scenario.include_lunar and moon.is_above_horizon:
        thermal, reflected = lunar_irradiance(grid, moon)
        beams.append(('lunar_multiple_scatter', moon, thermal+reflected,
                      'direct_lunar', LUNAR_SOLID_ANGLE_SR, .2586))
    for k in range(n):
        moments = phase_moments(column_optics.tau_aerosol_sca[::-1, k],
                                column_optics.tau_rayleigh[::-1, k],
                                column_optics.aerosol_asymmetry[::-1, k], streams)
        dt = tau_ext[:, k]
        tau_edges = np.r_[0., np.cumsum(dt)]
        tau_sensor, tau_end = np.interp([scenario.sensor_alt_km, z_end], heights, tau_edges[::-1])
        local = len(dt)-1-min(int(np.searchsorted(heights[1:], scenario.sensor_alt_km, side='right')), len(dt)-1)
        sources = [('path_thermal', blackbody[:, k], 0., None, 0.),
                   ('surface_emitted', None, surface_b[k], None, 0.)]
        sources += [(name, None, 0., body, irradiance[k]) for name, body, irradiance, *_ in beams]
        for name, thermal, boundary, body, beam in sources:
            if boundary == 0 and thermal is None and beam == 0:
                continue
            mu0 = np.cos(np.deg2rad(body.zenith_deg)) if body is not None else .5
            phi0 = np.deg2rad(body.azimuth_deg) if body is not None else 0.
            solved = solve_column(dt, omega[:, k], moments, streams=streams,
                                  mu0=mu0, beam=beam, phi0=phi0, planck_layers=thermal,
                                  surface_emission=boundary, surface_albedo=1.-eps)
            if horizontal:
                source = horizontal_source(solved, tau_sensor, omega[local, k], moments[local],
                                           phi=phi, mu0=mu0, beam=beam, phi0=phi0,
                                           planck_value=0. if thermal is None else thermal[local])
                value = source * -np.expm1(-optics.tau_total[:, k].sum())
            else:
                value = integrate_sightline(solved, path=path, optical_depth=optics.tau_total[:,k],
                                            heights=heights, tau_edges=tau_edges,
                                            omega=omega[:,k], moments=moments, mu_view=mu_view,
                                            phi=phi, planck_layers=thermal, mu0=mu0, beam=beam,
                                            phi0=phi0, surface_emission=boundary, surface_albedo=1.-eps,
                                            include_surface=path.hits_ground and not finite)
            if not np.isfinite(value):
                raise RuntimeError(f'DISORT returned nonfinite {name} at {grid.wavenumber_cm1[k]:g} cm^-1')
            scale = max(float(np.max(blackbody[:, k])), float(surface_b[k]), beam, 1e-30)
            if value < -1e-6*scale:
                raise RuntimeError('DISORT angular truncation produced negative radiance; increase disort_streams')
            components[name][k] = max(value, 0.)
        if path.reaches_space and not finite:
            for _, body, irradiance, direct_name, solid_angle, radius in beams:
                if body.scattering_angle_deg <= radius:
                    components[direct_name][k] = irradiance[k]/solid_angle*np.exp(-tau_sensor/np.cos(np.deg2rad(body.zenith_deg)))
    if scenario.source_temp_k is not None:
        components['source_emitted'] = scenario.source_emissivity*planck(grid.wavenumber_cm1, scenario.source_temp_k)*transmittance.total
    total = sum(components.values())
    return RadianceResult(grid=grid, total=total, **components)
