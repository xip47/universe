# tests/quantum_states/test_wavefunctions.py

import numpy as np
from universe.quantum_states.quantum_numbers import QuantumNumbers
from universe.quantum_states.wavefunctions import (
    hydrogenic_radial,
    spherical_harmonic,
    full_wavefunction,
)


def test_hydrogenic_radial_r1s() -> None:
    R = hydrogenic_radial(n=1, l=0)
    assert abs(R(0.0)) > 0.0
    assert R(5.0) < R(0.1)


def test_spherical_harmonic_Y00() -> None:
    Y = spherical_harmonic(l=0, m=0)
    value = Y(np.pi / 2, 0.0)
    assert np.isclose(abs(value), 1 / np.sqrt(4 * np.pi))


def test_full_wavefunction_evaluation() -> None:
    qn = QuantumNumbers(n=2, l=1, m=1, s=0.5, j=1.5)
    psi = full_wavefunction(qn)

    val = psi(1.0, np.pi / 4, np.pi / 3)
    assert isinstance(val, complex)
    assert abs(val) > 0.0
