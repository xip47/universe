# universe/quantum_states/wavefunctions.py

from __future__ import annotations

from typing import Callable
from math import factorial
import numpy as np
from scipy.special import genlaguerre, sph_harm_y as sph_harm
from universe.quantum_states.quantum_numbers import QuantumNumbers



def hydrogenic_radial(n: int, l: int, Z: float = 1.0) -> Callable[[float], float]:
    """
    Genera la función radial R_{n,l}(r) normalizada para un átomo hidrogenoide.

    Parameters
    ----------
    n : int
        Número cuántico principal (n ≥ 1).
    l : int
        Número cuántico orbital (l < n).
    Z : float, optional
        Carga nuclear (Z=1 para hidrógeno), por defecto 1.0.

    Returns
    -------
    Callable[[float], float]
        Función radial R_{n,l}(r) evaluable en r.
    """
    if n <= 0 or l < 0 or l >= n:
        raise ValueError("n debe ser ≥1 y l debe cumplir 0 ≤ l < n")

    rho = lambda r: 2 * Z * r / n
    norm = np.sqrt((2 * Z / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    L = genlaguerre(n - l - 1, 2 * l + 1)

    def R(r: float) -> float:
        return norm * rho(r) ** l * np.exp(-rho(r) / 2) * L(rho(r))

    return R


def spherical_harmonic(l: int, m: int) -> Callable[[float, float], complex]:
    """
    Devuelve el armónico esférico Y_{l}^{m}(θ, φ).

    Parameters
    ----------
    l : int
        Número cuántico orbital.
    m : int
        Proyección magnética (|m| ≤ l).

    Returns
    -------
    Callable[[float, float], complex]
        Función Y_{l}^{m}(θ, φ), θ ∈ [0, π], φ ∈ [0, 2π]
    """
    if abs(m) > l:
        raise ValueError("Debe cumplirse |m| ≤ l")

    def Y(theta: float, phi: float) -> complex:
        return sph_harm(m, l, phi, theta)

    return Y


def full_wavefunction(qn: QuantumNumbers, Z: float = 1.0) -> Callable[[float, float, float], complex]:
    """
    Construye la función de onda total ψ(r, θ, φ) = R_{n,l}(r) × Y_{l}^{m}(θ, φ)

    Parameters
    ----------
    qn : QuantumNumbers
        Números cuánticos del sistema.
    Z : float, optional
        Carga nuclear (hidrógeno = 1), por defecto 1.0.

    Returns
    -------
    Callable[[float, float, float], complex]
        Función de onda ψ(r, θ, φ)
    """
    R = hydrogenic_radial(qn.n, qn.l, Z)
    Y = spherical_harmonic(qn.l, qn.m)

    def psi(r: float, theta: float, phi: float) -> complex:
        return R(r) * Y(theta, phi)

    return psi
