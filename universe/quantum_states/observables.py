from __future__ import annotations

from universe.quantum_states.quantum_numbers import QuantumNumbers
from universe.quantum_states.wavefunctions import full_wavefunction
from universe.numerics.backend import xp



def radial_expectation_value(qn: QuantumNumbers, power: int = 1, Z: int = 1) -> float:
    """
    Calcula el valor esperado <r^power> para una función de onda hidrogenoide.

    Parameters
    ----------
    qn : QuantumNumbers
        Números cuánticos del estado.
    power : int
        Potencia del radio (1 para <r>, 2 para <r^2>, etc).
    Z : int
        Carga nuclear efectiva (Z=1 para hidrógeno).

    Returns
    -------
    float
        Valor esperado de r^power en [m^power].
    """
    r = xp.linspace(1e-12, 2e-9, 10_000)  # Dominio radial en metros
    psi = full_wavefunction(qn, r, Z)
    prob_density = xp.abs(psi)**2 * r**2
    norm = xp.trapz(prob_density, r)
    expected = xp.trapz(prob_density * r**power, r)
    return float(expected / norm)


def angular_momentum_squared(qn: QuantumNumbers) -> float:
    """
    Devuelve el valor esperado de L^2 en mecánica cuántica.

    Parameters
    ----------
    qn : QuantumNumbers
        Números cuánticos del estado.

    Returns
    -------
    float
        Valor esperado de L^2 en [J^2].
    """
    hbar = 1.054571817e-34  # Constante de Planck reducida [J.s]
    l = qn.l
    return float(hbar**2 * l * (l + 1))


def probability_in_region(qn: QuantumNumbers, r_min: float, r_max: float, Z: int = 1) -> float:
    """
    Calcula la probabilidad de encontrar al electrón entre r_min y r_max.

    Parameters
    ----------
    qn : QuantumNumbers
        Números cuánticos del estado.
    r_min : float
        Radio mínimo de la región [m].
    r_max : float
        Radio máximo de la región [m].
    Z : int
        Carga nuclear efectiva.

    Returns
    -------
    float
        Probabilidad (valor entre 0 y 1).
    """
    r = xp.linspace(r_min, r_max, 10_000)
    psi = full_wavefunction(qn, r, Z)
    prob_density = xp.abs(psi)**2 * r**2
    return float(xp.trapz(prob_density, r)) / float(xp.trapz(xp.abs(full_wavefunction(qn, xp.linspace(1e-12, 2e-9, 10_000), Z))**2 * xp.linspace(1e-12, 2e-9, 10_000)**2, xp.linspace(1e-12, 2e-9, 10_000)))
