from __future__ import annotations

from typing import Callable
from universe.numerics.backend import xp

# Constante de Rydberg en eV (energía de ionización del hidrógeno)
RYDBERG_ENERGY_EV: float = 13.605693122994

def hydrogenic_energy(n: int, Z: float = 1.0) -> float:
    """
    Calcula la energía del nivel n para un átomo de hidrógeno o ion hidrogenoide.

    Parameters
    ----------
    n : int
        Número cuántico principal (n ≥ 1).
    Z : float
        Carga nuclear efectiva (Z = 1 para hidrógeno).

    Returns
    -------
    float
        Energía en electronvoltios (eV), valor negativo para estados ligados.
    """
    if n < 1:
        raise ValueError("El número cuántico principal n debe ser ≥ 1.")
    return -RYDBERG_ENERGY_EV * (Z ** 2) / (n ** 2)


def numerov_energy_levels(
    potential_func: Callable[[float], float],
    energy_range: tuple[float, float],
    r_grid: xp.ndarray,
    max_states: int = 5,
    tol: float = 1e-6,
) -> list[float]:
    """
    Encuentra niveles de energía ligados usando el método de Numerov en 1D.

    Parameters
    ----------
    potential_func : Callable[[float], float]
        Función de potencial V(r) en MeV o eV (unidades consistentes).
    energy_range : tuple of float
        Rango de búsqueda (E_min, E_max).
    r_grid : xp.ndarray
        Malla radial donde se evalúa la ecuación.
    max_states : int
        Máximo número de estados ligados a encontrar.
    tol : float
        Tolerancia para convergencia energética.

    Returns
    -------
    list of float
        Energías encontradas, ordenadas de menor a mayor (más negativas primero).
    """
    # Placeholder: implementación posterior con Numerov + CUDA
    raise NotImplementedError("numerov_energy_levels aún no implementado.")
