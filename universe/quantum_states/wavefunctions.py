# universe/quantum_states/wavefunctions.py

from __future__ import annotations

from typing import Callable, Optional
from math import factorial
from scipy.special import genlaguerre, sph_harm
from universe.numerics.backend import xp
from universe.physics.constants import BOHR_RADIUS
from universe.quantum_states.quantum_numbers import QuantumNumbers


def hydrogenic_radial(n: int, l: int, Z: int = 1) -> Callable[[xp.ndarray], xp.ndarray]:
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

    def R(r: xp.ndarray) -> xp.ndarray:
        rho = 2 * Z * r / (n * BOHR_RADIUS)
        a0 = BOHR_RADIUS
        norm = xp.sqrt((2 * Z / (n * a0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))

        rho_cpu = xp.asnumpy(rho)
        laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
        laguerre_vals = laguerre_poly(rho_cpu)
        laguerre_vals_gpu = xp.asarray(laguerre_vals)

        return norm * rho**l * xp.exp(-rho / 2) * laguerre_vals_gpu

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


def full_wavefunction(qn: QuantumNumbers, r: Optional[xp.ndarray] = None, Z: int = 1) -> xp.ndarray:
    """
    Construye la función de onda radial ψ(r) = R_{n,l}(r) del átomo hidrogenoide.

    Parameters
    ----------
    qn : QuantumNumbers
        Números cuánticos del sistema.
    r : Optional[xp.ndarray]
        Arreglo radial en metros. Si no se provee, se genera un dominio por defecto.
    Z : int
        Carga nuclear efectiva.

    Returns
    -------
    xp.ndarray
        Función de onda radial evaluada sobre r.
    """
    if r is None:
        r = xp.linspace(1e-12, 2e-9, 10_000)

    R_func = hydrogenic_radial(qn.n, qn.l, Z)
    return R_func(r)
