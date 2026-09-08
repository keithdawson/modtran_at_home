"""Optical turbulence metrics integrated along a slant :class:`PathGeometry`.

Entry point: :func:`compute_turbulence`, called lazily and by exact keyword
signature from ``mwir_atmos.api._attach_turbulence``.

Path-integral convention (read this before any individual function)
---------------------------------------------------------------------
Every classic formula used here (Fried's r0, Fried's theta0, the plane-wave
Rytov variance) is normally written as an integral over *vertical altitude*
``h`` with an explicit ``sec(zeta)`` airmass factor pulled out front, on the
assumption of a plane-parallel atmosphere at a single, path-wide zenith angle
``zeta``. CONTRACT.md instead asks these to be integrated along the actual
resolved slant path, using ``PathLayer.ds_km`` for path length. This module
therefore re-expresses each formula in terms of the physical slant-path
coordinate directly, which has two consequences worth being explicit about:

1. **No separate secant factor is ever applied.** ``PathLayer.ds_km`` is
   already the true physical path length through each layer -- for a simple
   zenith-angle-``zeta`` look this equals ``dh * sec(zeta)`` exactly in the
   plane-parallel limit, so summing ``Cn2 * ds`` *is* ``sec(zeta) * integral
   Cn2 dh``, with no double-counting. Unlike a bare ``1/cos(zeta)`` factor,
   this never diverges at the horizon (``PathGeometry`` already resolves a
   near-90-degree look into a large but finite path length via the actual
   spherical geometry -- see ``geometry.path``'s own docstring), so these
   metrics stay finite for horizontal and limb paths where a textbook
   ``sec(zeta)`` formula would not.
2. **The isoplanatic-angle and Rytov-variance integrands are weighted by
   slant distance from the sensor, not by altitude.** Fried's theta0 formula
   (``2.914 k^2 sec(zeta)^(8/3) integral Cn2(h) h^(5/3) dh``) and the
   inhomogeneous-path Rytov variance both derive from a physical argument
   about the transverse ray separation / residual propagation distance *from
   the receiving aperture*, which grows with physical distance travelled,
   not with vertical height. Substituting the plane-parallel relation
   ``s = h*sec(zeta)`` (``s`` = slant distance from the sensor) into Fried's
   own formula gives ``theta0 = [2.914 k^2 integral Cn2(h(s)) s^(5/3)
   ds]^(-3/5)`` -- algebraically identical to the textbook form for a
   constant-zeta path (verified in tests/test_turbulence.py's vertical-path
   anchor case, where s == h exactly and sec(zeta) == 1), but now expressed
   as a path integral with *no* separate secant term, and correct for any
   path shape ``PathGeometry`` can produce, including one where the sensor
   is not at the low/dense-turbulence end of the path (e.g. a downward-
   looking airborne or space sensor). This restatement is not a local
   invention: Lawson, J.K. & Carrano, C.J. (2006), SPIE 6304 / LLNL
   UCRL-CONF-222540, Eq. 8, cite Fried (1982) directly for exactly this
   path-coordinate form of theta0. The same substitution, with the
   Tatarski/Rytov plane-wave weighting ``(L-z)^(5/6)`` (z measured from the
   far/source end) re-expressed as ``d^(5/6)`` (d = distance *from the
   sensor*), reduces to the homogeneous-path textbook constant: integrating
   a pure ``d^(5/6)`` power law over ``[0, L]`` gives ``(6/11) L^(11/6)``,
   and ``2.25 * 6/11 = 1.227 ~= 1.23``, recovering the task's homogeneous
   formula ``sigma_R^2 = 1.23 Cn2 k^(7/6) L^(11/6)`` exactly (also checked
   numerically in tests/test_turbulence.py).

Neither formula here carries a ``(h/H)`` or ``(1 - h/H)``-style weighting
(a ratio normalised to the *total* path length ``H``): that form belongs to
the spherical-wave uplink/downlink formulas for a source at a specific known
finite range (e.g. Andrews & Phillips (2005) Ch. 12's satellite up/downlink
pair), not the plane-wave, distant/unbounded-source convention the task
specifies for r0, theta0 and the Rytov variance here -- which is what is
implemented throughout. This is documented per-function below as well.

All internal arithmetic is SI (metres, radians, m^-2/3, 1/s); results are
converted to the dataclass's documented units (cm, urad, Hz) only in
:func:`compute_turbulence`.

Wavelength: turbulence is conventionally quoted at 0.5 um, but r0 and theta0
scale as lambda^(6/5) (steeper than most people expect), so a MWIR system
sees dramatically better numbers than the visible-light figures usually
quoted for a given Cn2 profile. ``compute_turbulence`` reports everything at
:data:`MWIR_REFERENCE_WAVELENGTH_UM` (4.0 um, the centre of
``constants.BANDS_CM1["MWIR_3_5um"]``, i.e. 3-5 um) by default, per
CONTRACT.md. Every metric function below is wavelength-parameterised so the
0.5 um figures (needed to validate against literature, e.g. HV5/7's own "5
cm/7 urad" namesake) can be reproduced with the same code -- see
tests/test_turbulence.py, which computes and asserts both.

Sources for the individual constants used below:
  * r0: Fried, D.L. (1966), "Optical Resolution Through a Randomly
    Inhomogeneous Medium for Very Long and Very Short Exposures," J. Opt.
    Soc. Am. 56, 1372-1379.
  * theta0: Fried, D.L. (1982), "Anisoplanatism in adaptive optics," J. Opt.
    Soc. Am. 72, 52-61.
  * Rytov variance / scintillation index (weak-to-strong, saturating):
    Andrews, L.C. & Phillips, R.L. (2005), "Laser Beam Propagation through
    Random Media," 2nd ed., SPIE Press -- the homogeneous plane-wave
    constant 1.23 is the classical Tatarski (1961) result.
  * Greenwood frequency: Greenwood, D.P. (1977), "Bandwidth specification
    for adaptive optics systems," J. Opt. Soc. Am. 67, 390-393.
  * Strehl ratio (Marechal approximation): Mahajan, V.N. (1983), "Strehl
    ratio for primary aberrations in terms of their aberration variance,"
    J. Opt. Soc. Am. 73, 860-861.
"""

from __future__ import annotations

import numpy as np

from ..types import AtmosphereProfile, PathGeometry, Scenario, TurbulenceResult
from .cn2 import bufton_wind_speed, cn2_profile

__all__ = ["compute_turbulence", "MWIR_REFERENCE_WAVELENGTH_UM", "VISIBLE_REFERENCE_WAVELENGTH_UM"]

#: Default reporting wavelength: centre of constants.BANDS_CM1["MWIR_3_5um"]
#: (2000-3333.33 cm^-1 == 3-5 um), per CONTRACT.md's turbulence spec.
MWIR_REFERENCE_WAVELENGTH_UM = 4.0

#: The conventional wavelength turbulence numbers are quoted at (used only to
#: validate against literature figures such as HV5/7's own "5 cm / 7 urad"
#: namesake; not the value ``compute_turbulence`` reports by default).
VISIBLE_REFERENCE_WAVELENGTH_UM = 0.5

#: Assumed sensor aperture diameter for :func:`strehl_ratio_marechal`, since
#: ``Scenario`` has no aperture-diameter field (out of this module's scope to
#: add -- see the dev-alpha report). 10 cm is a representative entrance-pupil
#: size for a MWIR imaging sensor; callers needing a different aperture
#: should call :func:`strehl_ratio_marechal` directly rather than relying on
#: this default.
ASSUMED_APERTURE_DIAMETER_M = 0.10

#: Numerical floor so a zero-turbulence path (or a single layer with exactly
#: zero Cn2) yields a very large-but-finite r0 / near-zero-but-finite
#: variance instead of a literal division by zero. Physically negligible at
#: any real Cn2 magnitude (Cn2 ~ 1e-19 to 1e-13 in the models here).
_EPS = 1.0e-30


def _wavenumber_rad_per_m(wavelength_um: float) -> float:
    """k = 2*pi/lambda, rad/m, from a wavelength given in micrometres."""
    return 2.0 * np.pi / (float(wavelength_um) * 1.0e-6)


def layer_geometry_m(path: PathGeometry, profile: AtmosphereProfile):
    """Per-layer height-above-ground, path length, and sensor distance (m).

    Returns ``(height_agl_m, ds_m, sensor_distance_m)``, each shape
    ``(path.n_layers,)``. ``height_agl_m`` is each layer's mid-point altitude
    minus ``profile.surface_altitude_km`` (both are in the same absolute
    altitude datum -- see ``geometry.path.build_path``), floored at 0, which
    is the height coordinate every Cn2(h) model in :mod:`.cn2` expects.
    ``sensor_distance_m`` is the cumulative slant distance from the sensor
    (``PathLayer`` index 0, per ``types.PathLayer``'s "layer 0 is adjacent to
    the sensor" convention) to each layer's mid-point -- the coordinate the
    isoplanatic-angle and Rytov-variance integrals are weighted by (see the
    module docstring).
    """
    ds_km = np.array([layer.ds_km for layer in path.layers], dtype=np.float64)
    z_mid_km = np.array(
        [0.5 * (layer.z_bottom_km + layer.z_top_km) for layer in path.layers],
        dtype=np.float64,
    )
    ds_m = ds_km * 1000.0
    height_agl_m = np.maximum((z_mid_km - profile.surface_altitude_km) * 1000.0, 0.0)
    cumulative_start_m = np.concatenate(([0.0], np.cumsum(ds_m)[:-1]))
    sensor_distance_m = cumulative_start_m + 0.5 * ds_m
    return height_agl_m, ds_m, sensor_distance_m


def fried_parameter_r0_m(cn2_m23: np.ndarray, ds_m: np.ndarray, wavelength_um: float) -> float:
    """Fried parameter r0, metres (plane wave; see module docstring).

    ``r0 = [0.423 k^2 integral Cn2 ds]^(-3/5)``, ``ds`` the physical slant
    path length -- see the module docstring for why this needs no separate
    ``sec(zeta)`` term.
    """
    k = _wavenumber_rad_per_m(wavelength_um)
    integral = float(np.sum(cn2_m23 * ds_m))
    return (0.423 * k**2 * max(integral, _EPS)) ** (-3.0 / 5.0)


def isoplanatic_angle_rad(
    cn2_m23: np.ndarray, sensor_distance_m: np.ndarray, ds_m: np.ndarray, wavelength_um: float
) -> float:
    """Isoplanatic angle theta0, radians (plane wave; see module docstring).

    ``theta0 = [2.914 k^2 integral Cn2(h(s)) s^(5/3) ds]^(-3/5)``, ``s`` the
    slant distance *from the sensor* (not altitude -- see module docstring).
    """
    k = _wavenumber_rad_per_m(wavelength_um)
    integral = float(np.sum(cn2_m23 * sensor_distance_m ** (5.0 / 3.0) * ds_m))
    return (2.914 * k**2 * max(integral, _EPS)) ** (-3.0 / 5.0)


def rytov_variance_plane_wave(
    cn2_m23: np.ndarray, sensor_distance_m: np.ndarray, ds_m: np.ndarray, wavelength_um: float
) -> float:
    """Rytov variance sigma_R^2, dimensionless (plane wave; unbounded/distant
    source, i.e. no ``h/H`` normalisation -- see module docstring).

    ``sigma_R^2 = 2.25 k^(7/6) integral Cn2(h(s)) s^(5/6) ds``, ``s`` the
    slant distance from the sensor. Reduces to the textbook homogeneous-path
    ``1.23 Cn2 k^(7/6) L^(11/6)`` for constant Cn2 over path length L (module
    docstring; checked numerically in tests/test_turbulence.py). This is the
    weak-fluctuation building block for :func:`scintillation_index`, not
    itself saturating -- it can exceed 1 for a long/strong path.
    """
    k = _wavenumber_rad_per_m(wavelength_um)
    integral = float(np.sum(cn2_m23 * sensor_distance_m ** (5.0 / 6.0) * ds_m))
    return 2.25 * k ** (7.0 / 6.0) * integral


def scintillation_index(rytov_variance: float) -> float:
    """Scintillation index sigma_I^2 from the Rytov variance, saturating.

    Weak-fluctuation theory alone (``sigma_I^2 = sigma_R^2``) grows without
    bound and is well known to overshoot measured scintillation once
    ``sigma_R^2`` approaches order unity ("strong scintillation"). This uses
    the Andrews & Phillips (2005) moderate-to-strong-fluctuation model,
    which splits the log-irradiance variance into large-scale (``sigma_lnX^2``,
    refractive) and small-scale (``sigma_lnY^2``, diffractive) components
    that each roll over as ``sigma_R^2`` grows, so the exponentiated result
    approaches a bounded saturation value (~1) instead of diverging:

        sigma_lnX^2 = 0.49 sigma_R^2 / (1 + 1.11 sigma_R^(12/5))^(7/6)
        sigma_lnY^2 = 0.51 sigma_R^2 / (1 + 0.69 sigma_R^(12/5))^(5/6)
        sigma_I^2   = exp(sigma_lnX^2 + sigma_lnY^2) - 1

    For small sigma_R^2 this reduces to sigma_I^2 ~= sigma_R^2 (weak-
    fluctuation limit); as sigma_R^2 -> infinity, sigma_lnX^2 -> 0 and
    sigma_lnY^2 -> 0.51/0.69^(5/6) ~= 0.695, so sigma_I^2 saturates to
    ~1.0 (verified in tests/test_turbulence.py over sigma_R^2 in [1e-3, 1e6]).
    """
    x = max(float(rytov_variance), 0.0)
    sigma_lnx2 = 0.49 * x / (1.0 + 1.11 * x**1.2) ** (7.0 / 6.0)
    sigma_lny2 = 0.51 * x / (1.0 + 0.69 * x**1.2) ** (5.0 / 6.0)
    return float(np.exp(sigma_lnx2 + sigma_lny2) - 1.0)


def greenwood_frequency_hz(
    cn2_m23: np.ndarray,
    ds_m: np.ndarray,
    height_agl_m: np.ndarray,
    r0_m: float,
    *,
    ground_wind_m_s: float = 5.0,
) -> float:
    """Greenwood frequency f_G, Hz: ``f_G = 0.427 * vbar / r0``.

    ``vbar`` is the Cn2-weighted 5/3-power-mean wind speed,
    ``vbar = [integral Cn2 V(h)^(5/3) ds / integral Cn2 ds]^(3/5)``, using
    the Bufton (1973) wind profile ``V(h)`` (:func:`cn2.bufton_wind_speed`)
    -- the same wind model HV5/7's "pseudowind = 21 m/s" constant is itself
    built from (its rms over 5-20 km altitude comes out close to 21 m/s; see
    :mod:`.cn2`'s module docstring), so this uses one consistent wind
    profile throughout rather than mixing wind assumptions between the Cn2
    model and the Greenwood-frequency wind weighting. Both the numerator and
    denominator use the same ``ds`` weighting, so (unlike r0/theta0) this
    ratio needs no secant or distance-from-sensor correction -- any such
    factor would cancel.
    """
    wind = bufton_wind_speed(height_agl_m, ground_wind_m_s=ground_wind_m_s)
    numerator = float(np.sum(cn2_m23 * wind ** (5.0 / 3.0) * ds_m))
    denominator = float(np.sum(cn2_m23 * ds_m))
    v_bar = (numerator / max(denominator, _EPS)) ** (3.0 / 5.0) if numerator > 0.0 else 0.0
    return 0.427 * v_bar / max(r0_m, _EPS)


def strehl_ratio_marechal(r0_m: float, aperture_diameter_m: float) -> float:
    """Strehl ratio via the Marechal approximation, ``S = exp(-sigma_phi^2)``.

    ``sigma_phi^2 = (D/r0)^(5/3)`` follows directly from Fried's own
    definition of r0 (the aperture diameter at which the residual mean-
    square wavefront error is exactly 1 rad^2), with no extra leading
    coefficient. This is *not* Noll's (1976) more precise large-aperture,
    piston-only-removed constant (1.03), which is a refinement not applied
    here; nor is it Fried's alternate ``[1+(D/r0)^(5/3)]^(-6/5)``
    interpolation, which stays better-behaved for ``D >> r0`` (the Marechal
    exponential underestimates the already-tiny Strehl there). Adequate for
    the ``D ~ r0`` regime this tool otherwise targets; documented per
    CONTRACT.md's instruction to say which approximation was used.
    """
    ratio = float(aperture_diameter_m) / max(r0_m, 1.0e-9)
    return float(np.exp(-(ratio ** (5.0 / 3.0))))


def compute_turbulence(
    *, scenario: Scenario, path: PathGeometry, profile: AtmosphereProfile
) -> TurbulenceResult:
    """Optical turbulence metrics along ``path``, reported at
    :data:`MWIR_REFERENCE_WAVELENGTH_UM`.

    Binding signature -- called lazily by ``api._attach_turbulence``. Uses
    ``profile`` only for ``profile.surface_altitude_km`` (the height-above-
    ground datum every Cn2(h) model needs); the Cn2 models themselves are
    the standard empirical altitude-only fits named by CONTRACT.md
    (HV5/7, SLC Day/Night, constant), not derived from the thermodynamic
    profile.
    """
    height_agl_m, ds_m, sensor_distance_m = layer_geometry_m(path, profile)
    cn2_m23 = cn2_profile(
        scenario.turbulence, height_agl_m, ground_cn2_m23=scenario.turbulence_cn2_ground
    )

    wavelength_um = MWIR_REFERENCE_WAVELENGTH_UM
    r0_m = fried_parameter_r0_m(cn2_m23, ds_m, wavelength_um)
    theta0_rad = isoplanatic_angle_rad(cn2_m23, sensor_distance_m, ds_m, wavelength_um)
    rytov = rytov_variance_plane_wave(cn2_m23, sensor_distance_m, ds_m, wavelength_um)
    scint = scintillation_index(rytov)
    f_greenwood = greenwood_frequency_hz(cn2_m23, ds_m, height_agl_m, r0_m)

    aperture_m = float(scenario.sensor_aperture_diameter_m)
    strehl = strehl_ratio_marechal(r0_m, aperture_m)

    # Point-source response. Turbulence conserves total power: it widens the
    # spot and lowers the peak, it does not absorb. The long-exposure spot is
    # the quadrature sum of the diffraction core and the seeing disc, so an
    # aperture smaller than r0 stays effectively diffraction-limited.
    lam_m = wavelength_um * 1.0e-6
    diffraction_urad = (lam_m / max(aperture_m, 1.0e-9)) * 1.0e6
    seeing_urad = (lam_m / max(r0_m, 1.0e-9)) * 1.0e6
    effective_urad = float(np.hypot(diffraction_urad, seeing_urad))

    return TurbulenceResult(
        reference_wavelength_um=wavelength_um,
        cn2_profile_m23=cn2_m23,
        fried_parameter_r0_cm=r0_m * 100.0,
        isoplanatic_angle_urad=theta0_rad * 1.0e6,
        rytov_variance=rytov,
        scintillation_index=scint,
        greenwood_frequency_hz=f_greenwood,
        strehl_ratio=strehl,
        aperture_diameter_m=aperture_m,
        diffraction_limit_urad=diffraction_urad,
        seeing_limit_urad=seeing_urad,
        effective_blur_urad=effective_urad,
        is_seeing_limited=bool(aperture_m > r0_m),
        peak_irradiance_ratio=strehl,
    )
