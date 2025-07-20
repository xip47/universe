from __future__ import annotations

from universe.quantum_states.quantum_numbers import QuantumNumbers
from universe.quantum_states.wavefunctions import full_wavefunction
from universe.numerics.backend import xp



def radial_expectation_value(qn: QuantumNumbers, power: int = 1, Z: int = 1) -> float:
    """
    Calcula el valor esperado <r^power> para el estado hidrogenoide dado, con normalización física.
    """
    r = xp.linspace(1e-12, 2e-9, 10000)
    psi = full_wavefunction(qn, Z)
    psi_r = psi(r, 0.0, 0.0)
    prob_density = xp.abs(psi_r) ** 2
    dr = r[1] - r[0]
    norm = float(xp.sum(prob_density * r ** 2) * dr)
    expected = float(xp.sum(prob_density * r ** power * r ** 2) * dr)
    return expected / norm


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
    Calcula la probabilidad de encontrar el electrón entre r_min y r_max, con normalización física.
    """
    r = xp.linspace(r_min, r_max, 1000)
    psi = full_wavefunction(qn, Z)
    psi_r = psi(r, 0.0, 0.0)
    prob_density = xp.abs(psi_r) ** 2
    dr = r[1] - r[0]
    # Normalizar usando la integral total en [1e-12, 2e-9]
    r_full = xp.linspace(1e-12, 2e-9, 10000)
    psi_full = psi(r_full, 0.0, 0.0)
    prob_full = xp.abs(psi_full) ** 2
    dr_full = r_full[1] - r_full[0]
    norm = float(xp.sum(prob_full * r_full ** 2) * dr_full)
    prob = float(xp.sum(prob_density * r ** 2) * dr)
    return prob / norm
