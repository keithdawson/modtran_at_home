"""Physical constants and band definitions.

Unit conventions used throughout ``mwir_atmos``:
  * wavenumber      cm^-1
  * wavelength      micrometres (um)
  * altitude        km
  * pressure        millibar (== hPa)
  * temperature     kelvin
  * column density  molecules cm^-2
  * optical depth   dimensionless
  * spectral radiance   W cm^-2 sr^-1 / cm^-1   (MODTRAN convention)
  * spectral irradiance W cm^-2 / cm^-1
"""

from __future__ import annotations

# --- fundamental (CODATA / SI defining constants) -------------------------
PLANCK_H = 6.62607015e-34          # J s
LIGHT_C = 2.99792458e8             # m s^-1
BOLTZMANN_K = 1.380649e-23         # J K^-1
AVOGADRO = 6.02214076e23           # mol^-1
GAS_CONSTANT_R = 8.314462618       # J mol^-1 K^-1
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4

# Radiation constants in the wavenumber form used by planck():
#   B(nu, T) = C1_RAD * nu^3 / (exp(C2_RAD * nu / T) - 1)
# with nu in cm^-1 and B in W cm^-2 sr^-1 / cm^-1.
C1_RAD = 1.191042972e-12           # W cm^2 sr^-1
C2_RAD = 1.4387768775             # cm K

# --- atmosphere -----------------------------------------------------------
STD_PRESSURE_MBAR = 1013.25
STD_TEMP_K = 273.15
LOSCHMIDT_CM3 = 2.6867811e19       # molecules cm^-3 at STD_TEMP_K / STD_PRESSURE_MBAR
M_AIR_G_MOL = 28.9644              # dry air mean molar mass
M_H2O_G_MOL = 18.01528
G0_M_S2 = 9.80665
EARTH_RADIUS_KM = 6371.0           # mean radius; used for spherical path geometry
TOA_ALTITUDE_KM = 100.0            # top of modelled atmosphere

# --- sun / moon -----------------------------------------------------------
SOLAR_CONSTANT_W_M2 = 1361.0
SOLAR_EFFECTIVE_TEMP_K = 5778.0
SOLAR_ANGULAR_RADIUS_RAD = 4.6543e-3   # at 1 AU
SOLAR_SOLID_ANGLE_SR = 6.807e-5
LUNAR_ANGULAR_RADIUS_RAD = 4.5128e-3   # mean
LUNAR_SOLID_ANGLE_SR = 6.396e-5
LUNAR_GEOMETRIC_ALBEDO = 0.12          # broadband visible/NIR; MWIR handled separately
ASTRONOMICAL_UNIT_KM = 1.495978707e8

# --- spectral band definitions (cm^-1) ------------------------------------
# Default working grid spans the full MWIR plus band wings so that CO2 4.3 um
# and H2O 2.7 um structure is captured rather than clipped.
MWIR_GRID_MIN_CM1 = 1800.0         # 5.56 um
MWIR_GRID_MAX_CM1 = 4300.0         # 2.33 um
DEFAULT_RESOLUTION_CM1 = 1.0

#: Named integration bands, (min_cm1, max_cm1).
BANDS_CM1 = {
    "MWIR_3_5um": (2000.0, 3333.33),
    "MWIR_blue_spike": (2325.0, 2400.0),   # hot CO2 emission, 4.17-4.30 um
    "MWIR_window_1": (2380.95, 2857.14),   # 3.5-4.2 um
    "MWIR_window_2": (2000.0, 2222.22),    # 4.5-5.0 um
    "CO2_4_3um": (2200.0, 2400.0),
    "H2O_2_7um": (3400.0, 3900.0),
    "full_grid": (MWIR_GRID_MIN_CM1, MWIR_GRID_MAX_CM1),
}

# --- species --------------------------------------------------------------
#: Molar mass (g/mol) of every radiatively active species we model in the MWIR.
MOLAR_MASS_G_MOL = {
    "H2O": 18.01528,
    "CO2": 44.0095,
    "O3": 47.9982,
    "N2O": 44.0128,
    "CO": 28.0101,
    "CH4": 16.0425,
    "O2": 31.9988,
    "N2": 28.0134,
}

#: Well-mixed background volume mixing ratios (ppmv) used when a model
#: atmosphere does not tabulate the species. CO2 value is a 2025-era estimate.
BACKGROUND_VMR_PPMV = {
    "CO2": 423.0,
    "N2O": 0.337,
    "CO": 0.10,
    "CH4": 1.92,
    "O2": 2.09e5,
    "N2": 7.81e5,
}
