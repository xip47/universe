from universe.numerics.backend import xp
from universe.physics.constants import E_CHARGE
from universe.physics.constants import EPSILON_0
from typing import Tuple


def coulomb_force(q1: float, q2: float, r1: Tuple[float, ...], r2: Tuple[float, ...]) -> xp.ndarray:
    """
    Calcula el vector de fuerza de Coulomb entre dos cargas puntuales.

    Parameters
    ----------
    q1 : float
        Carga de la primera partícula en coulombs (C).
    q2 : float
        Carga de la segunda partícula en coulombs (C).
    r1 : tuple of float
        Posición (x, y, z) de la primera partícula en metros (m).
    r2 : tuple of float
        Posición (x, y, z) de la segunda partícula en metros (m).

    Returns
    -------
    xp.ndarray
        Vector de fuerza (N) ejercida sobre la partícula 1 por la partícula 2.
    """
    r1_arr = xp.array(r1, dtype=xp.float64)
    r2_arr = xp.array(r2, dtype=xp.float64)
    r_vec = r1_arr - r2_arr
    distance = xp.linalg.norm(r_vec)
    if distance == 0:
        raise ValueError("Las partículas no pueden ocupar la misma posición (singularidad de Coulomb).")
    force_magnitude = (1 / (4 * xp.pi * EPSILON_0)) * (q1 * q2) / (distance ** 2)
    force_vector = force_magnitude * (r_vec / distance)
    return force_vector


def coulomb_potential(q1: float, q2: float, r1: Tuple[float, ...], r2: Tuple[float, ...]) -> float:
    """
    Calcula el potencial escalar de Coulomb entre dos cargas puntuales.

    Parameters
    ----------
    q1 : float
        Carga de la primera partícula en coulombs (C).
    q2 : float
        Carga de la segunda partícula en coulombs (C).
    r1 : tuple of float
        Posición (x, y, z) de la primera partícula en metros (m).
    r2 : tuple of float
        Posición (x, y, z) de la segunda partícula en metros (m).

    Returns
    -------
    float
        Potencial escalar (J) entre las dos partículas.
    """
    r1_arr = xp.array(r1, dtype=xp.float64)
    r2_arr = xp.array(r2, dtype=xp.float64)
    distance = xp.linalg.norm(r1_arr - r2_arr)
    if distance == 0:
        raise ValueError("Las partículas no pueden ocupar la misma posición (singularidad de Coulomb).")
    potential = (1 / (4 * xp.pi * EPSILON_0)) * (q1 * q2) / distance
    return float(potential)


def coulomb_vector_potential(*args, **kwargs):
    """
    Placeholder para el potencial vectorial de Coulomb (no relevante para cargas estáticas).
    """
    raise NotImplementedError("El potencial vectorial de Coulomb solo es relevante para cargas en movimiento (corrientes).")
