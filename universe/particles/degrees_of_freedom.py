"""
Módulo de grados de libertad fundamentales para partículas.
Incluye spin, isospin, carga y estado cuántico.
"""

from typing import Optional
from universe.numerics.backend import xp

class Spin:
    """
    Spin degree of freedom for a particle.

    Grado de libertad de spin para una partícula.

    Parameters
    ----------
    value : float
        Valor del spin (en unidades de ħ).
    vector : Optional[xp.ndarray]
        Vector de spin (si aplica, 3D).
    """
    def __init__(self, value: float, vector: Optional[xp.ndarray] = None) -> None:
        self.value: float = value
        self.vector: Optional[xp.ndarray] = vector

class Isospin:
    """
    Isospin degree of freedom for a particle.

    Grado de libertad de isospin para una partícula.

    Parameters
    ----------
    value : float
        Valor del isospin (en unidades convencionales).
    vector : Optional[xp.ndarray]
        Vector de isospin (si aplica, 3D).
    """
    def __init__(self, value: float, vector: Optional[xp.ndarray] = None) -> None:
        self.value: float = value
        self.vector: Optional[xp.ndarray] = vector

class Charge:
    """
    Electric charge degree of freedom for a particle.

    Grado de libertad de carga eléctrica para una partícula.

    Parameters
    ----------
    value : float
        Valor de la carga (en unidades de e).
    """
    def __init__(self, value: float) -> None:
        self.value: float = value

class QuantumState:
    """
    Quantum state for a particle (números cuánticos principales).

    Estado cuántico para una partícula (n, l, m, s, j, t, etc).

    Parameters
    ----------
    n : int
        Número cuántico principal.
    l : int
        Momento angular orbital.
    m : int
        Proyección del momento angular.
    s : float
        Spin.
    j : float
        Momento angular total.
    t : Optional[float]
        Isospin (opcional).
    """
    def __init__(self, n: int, l: int, m: int, s: float, j: float, t: Optional[float] = None) -> None:
        self.n: int = n
        self.l: int = l
        self.m: int = m
        self.s: float = s
        self.j: float = j
        self.t: Optional[float] = t
