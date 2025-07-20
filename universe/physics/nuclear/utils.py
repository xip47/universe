"""
Nuclear Constants and Utilities Module.

Constantes y utilidades nucleares para simulaciones realistas.
"""

# Constantes físicas nucleares
PROTON_MASS_MEV: float = 938.27208816  # MeV/c²
NEUTRON_MASS_MEV: float = 939.56542052  # MeV/c²
ELECTRON_MASS_MEV: float = 0.510998950  # MeV/c²
PION_MASS_MEV: float = 139.57039  # MeV/c² (pion cargado)

# Acoplo fuerte efectivo (adimensional, orden 1)
STRONG_COUPLING: float = 1.0
# Masa del mesón mediador (pion) en 1/m
PION_MASS_INVFM: float = 1.43e15  # 1/fm ~ 1.43e15 1/m

# Conversión de unidades
FM_TO_M: float = 1e-15
MEV_TO_J: float = 1.60218e-13

# Utilidades para futuros cálculos
# def reduced_mass(m1: float, m2: float) -> float:
#     """
#     Compute the reduced mass of two particles (MeV/c²).
#     Calcula la masa reducida de dos partículas (MeV/c²).
#     """
#     return m1 * m2 / (m1 + m2)
