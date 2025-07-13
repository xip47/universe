"""Constantes físicas fundamentales del universo."""

from math import pi
from typing import Final


# === Constantes universales === #

C: Final[float] = 299_792_458.0  # Velocidad de la luz (m/s)
HBAR: Final[float] = 1.054_571_817e-34  # Constante de Planck reducida (J·s)
PLANCK: Final[float] = 6.626_070_15e-34  # Constante de Planck (J·s)
E_CHARGE: Final[float] = 1.602_176_634e-19  # Carga elemental (C)
EPSILON_0: Final[float] = 8.854_187_8128e-12  # Permitividad del vacío (F·m⁻¹)

# === Masas fundamentales === #

M_ELECTRON: Final[float] = 9.109_383_7015e-31  # Masa del electrón (kg)
M_PROTON: Final[float] = 1.672_621_923_69e-27  # Masa del protón (kg)
M_NEUTRON: Final[float] = 1.674_927_498_04e-27  # Masa del neutrón (kg)

# === Otras constantes === #

G: Final[float] = 6.674_30e-11  # Constante gravitacional (m³·kg⁻¹·s⁻²)
K_B: Final[float] = 1.380_649e-23  # Constante de Boltzmann (J/K)
N_AVOGADRO: Final[float] = 6.022_140_76e23  # Número de Avogadro

# === Conversiones de unidades === #

MEV_TO_J: Final[float] = 1.602_18e-13  # 1 MeV = 1.60218e-13 J
J_TO_MEV: Final[float] = 1 / MEV_TO_J
FM_TO_M: Final[float] = 1e-15  # Femtómetro a metro
M_TO_FM: Final[float] = 1e15
U_KG: Final[float] = 1.660_539_066_60e-27  # Unidad de masa atómica (kg)


def fine_structure_constant() -> float:
    """
    Calcula la constante de estructura fina en unidades del SI.

    Returns
    -------
    float
        Constante de estructura fina adimensional (≈ 1 / 137.035999).
    """
    return E_CHARGE**2 / (4 * pi * EPSILON_0 * HBAR * C)
