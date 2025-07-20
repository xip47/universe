"""
Nuclear Decays Module: Weak interaction and decay processes.

Módulo de decaimientos nucleares: interacción débil y procesos de decaimiento.
Incluye la interfaz para decaimiento beta y futuros procesos débiles.
"""
from typing import Protocol
import numpy as np

class BaseNuclearDecay(Protocol):
    """
    Nuclear decay interface.

    Interfaz para decaimientos nucleares (débil, radiactivo, etc).

    Methods
    -------
    decay_rate(...) -> float
        Devuelve la tasa de decaimiento (1/s).
    decay_products(...) -> dict
        Devuelve los productos del decaimiento.
    """
    def decay_rate(self, *args, **kwargs) -> float:
        ...
    def decay_products(self, *args, **kwargs) -> dict:
        ...

def beta_decay(
    neutron_mass: float = 939.565,
    proton_mass: float = 938.272,
    electron_mass: float = 0.511
) -> dict:
    """
    Standard beta decay: n → p + e⁻ + ν̄ₑ.

    Decaimiento beta estándar: n → p + e⁻ + ν̄ₑ.

    Parameters
    ----------
    neutron_mass : float
        Masa del neutrón (MeV/c²).
    proton_mass : float
        Masa del protón (MeV/c²).
    electron_mass : float
        Masa del electrón (MeV/c²).

    Returns
    -------
    dict
        Diccionario con productos y energías liberadas.
    """
    Q: float = neutron_mass - proton_mass - electron_mass  # Energía liberada (MeV)
    return {
        'products': ['proton', 'electron', 'antineutrino'],
        'Q_value_MeV': Q,
        'lifetime_s': 880.2  # Vida media del neutrón (s, experimental)
    }

# Placeholders para futuros procesos débiles y neutrinos
# def weak_transition(...):
#     ...
