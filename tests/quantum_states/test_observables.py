"""
Tests for quantum mechanical observables (⟨rⁿ⟩, ⟨L²⟩, probability in region).

Se valida que los valores esperados calculados para funciones de onda
hidrogenoides sean coherentes con resultados físicos conocidos.
"""

from __future__ import annotations

import pytest

from universe.quantum_states.quantum_numbers import QuantumNumbers
from universe.quantum_states.observables import angular_momentum_squared
from universe.quantum_states.observables import probability_in_region
from universe.quantum_states.observables import radial_expectation_value


def test_radial_expectation_value_r1() -> None:
    """
    Test <r> para el estado 1s, valor cercano al radio de Bohr.

    Returns
    -------
    None
    """
    qn: QuantumNumbers = QuantumNumbers(n=1, l=0, m=0, s=0.5, j=0.5)
    result: float = radial_expectation_value(qn, power=1)
    expected: float = 7.937658e-11
    assert abs(result - expected) < 1e-12


def test_angular_momentum_squared_l1() -> None:
    """
    Test ⟨L²⟩ para l=1 (estado p), se espera ℏ² * l(l+1).

    Returns
    -------
    None
    """
    qn: QuantumNumbers = QuantumNumbers(n=2, l=1, m=0, s=0.5, j=1.5)
    result: float = angular_momentum_squared(qn)
    hbar: float = 1.054571817e-34
    expected: float = hbar**2 * 1 * (1 + 1)
    assert abs(result - expected) / expected < 1e-6


def test_probability_in_region_1s() -> None:
    """
    Test de probabilidad de encontrar electrón en r ∈ [0, 1e-10].

    Returns
    -------
    None
    """
    qn: QuantumNumbers = QuantumNumbers(n=1, l=0, m=0, s=0.5, j=0.5)
    result: float = probability_in_region(qn, 0.0, 1e-10)
    assert 0.7 < result < 1.0
