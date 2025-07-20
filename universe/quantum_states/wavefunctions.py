# universe/quantum_states/wavefunctions.py

from __future__ import annotations

from typing import Callable, Optional
from math import factorial
from scipy.special import genlaguerre, sph_harm
from universe.numerics.backend import xp
from universe.physics.constants import BOHR_RADIUS
from universe.quantum_states.quantum_numbers import QuantumNumbers


def hydrogenic_radial(n: int, l: int, Z: int = 1) -> Callable[[float | xp.ndarray], xp.ndarray]:
    """
    Genera la función radial R_{n,l}(r) normalizada para un átomo hidrogenoide.

    Parameters
    ----------
    n : int
        Número cuántico principal (n ≥ 1).
    l : int
        Número cuántico orbital (0 ≤ l < n).
    Z : int
        Carga nuclear efectiva (Z=1 para hidrógeno).

    Returns
    -------
    Callable[[xp.ndarray], xp.ndarray]
        Función evaluable R(r) que acepta arreglos tipo xp.ndarray.
    """
    if n <= 0 or l < 0 or l >= n:
        raise ValueError("n debe ser ≥1 y l debe cumplir 0 ≤ l < n")

    def R(r: float | xp.ndarray) -> xp.ndarray:
        r = xp.asarray(r)
        # Usar dominio en Ångströms para evitar underflow
        r = r.astype(xp.float64)
        rho = 2 * Z * r / (n * BOHR_RADIUS)
        a0 = BOHR_RADIUS
        # Normalización explícita para n=1, l=0
        if n == 1 and l == 0:
            norm = 2 / xp.sqrt(a0 ** 3)
            result = norm * xp.exp(-r / a0)
        else:
            norm = xp.sqrt((2 * Z / (n * a0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
            rho_cpu = xp.asnumpy(rho)
            laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
            laguerre_vals = laguerre_poly(rho_cpu)
            laguerre_vals_gpu = xp.asarray(laguerre_vals)
            result = norm * rho**l * xp.exp(-rho / 2) * laguerre_vals_gpu
        if result.shape == ():
            return xp.array(result)
        return result

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


def full_wavefunction(qn: QuantumNumbers, Z: int = 1) -> Callable[[float, float, float], complex]:
    """
    Construye la función de onda completa ψ(r, θ, φ) = R_{n,l}(r) Y_{l}^{m}(θ, φ) del átomo hidrogenoide.

    Parameters
    ----------
    qn : QuantumNumbers
        Números cuánticos del sistema.
    Z : int
        Carga nuclear efectiva.

    Returns
    -------
    Callable[[float, float, float], complex]
        Función de onda completa evaluada sobre (r, θ, φ).
    """
    R_func = hydrogenic_radial(qn.n, qn.l, Z)
    Y_func = spherical_harmonic(qn.l, qn.m)
    def psi(r: float, theta: float, phi: float) -> complex:
        val = R_func(r) * Y_func(theta, phi)
        # Si el resultado es un array escalar, devolver como escalar puro
        if hasattr(val, 'shape') and val.shape == ():
            return val.item()
        return val
    return psi
