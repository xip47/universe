from universe.numerics.backend import xp
from universe.physics.constants import E_CHARGE
from universe.physics.constants import EPSILON_0
from typing import Tuple
from typing import Sequence
from universe.physics.electromagnetism.coulomb import coulomb_force


def electric_field_point(q: float, r_q: Tuple[float, ...], r_eval: Tuple[float, ...]) -> xp.ndarray:
    """
    Calcula el campo eléctrico generado por una carga puntual en un punto del espacio.

    Parameters
    ----------
    q : float
        Carga en coulombs (C).
    r_q : tuple of float
        Posición de la carga (m).
    r_eval : tuple of float
        Punto de evaluación (m).

    Returns
    -------
    xp.ndarray
        Vector de campo eléctrico (V/m) en r_eval.
    """
    rq = xp.array(r_q, dtype=xp.float64)
    r = xp.array(r_eval, dtype=xp.float64)
    r_vec = r - rq
    distance = xp.linalg.norm(r_vec)
    if distance == 0:
        raise ValueError("El campo eléctrico no está definido en la posición de la carga (singularidad).")
    E = (1 / (4 * xp.pi * EPSILON_0)) * q * (r_vec / distance**3)
    return E


def electric_field_system(qs: Sequence[float], rs: Sequence[Tuple[float, ...]], r_eval: Tuple[float, ...]) -> xp.ndarray:
    """
    Calcula el campo eléctrico total en un punto debido a un sistema de cargas puntuales.

    Parameters
    ----------
    qs : Sequence[float]
        Lista de cargas en coulombs (C).
    rs : Sequence[tuple of float]
        Lista de posiciones de las cargas (m).
    r_eval : tuple of float
        Punto de evaluación (m).

    Returns
    -------
    xp.ndarray
        Vector de campo eléctrico (V/m) en r_eval.
    """
    E_total = xp.zeros_like(xp.array(r_eval, dtype=xp.float64))
    for q, rq in zip(qs, rs):
        try:
            E_total += electric_field_point(q, rq, r_eval)
        except ValueError:
            continue
    return E_total


def magnetic_field_point(*args, **kwargs):
    """
    Placeholder para el campo magnético de una carga en movimiento (no implementado).
    """
    raise NotImplementedError("El campo magnético requiere cargas en movimiento (corrientes).")
