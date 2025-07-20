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
    # Usar valores en Ångströms (1 Å = 1e-10 m)
    r0 = 0.0
    r1 = 0.1e-10
    r5 = 5.0e-10
    assert abs(R(r0)) > 0.0
    assert R(r5) < R(r1)
    # También probar con arrays
    r_arr = np.array([r0, r1, r5])
    vals = R(r_arr)
    assert vals.shape == (3,)
    assert np.all(vals >= 0)


def test_spherical_harmonic_Y00() -> None:
    Y = spherical_harmonic(l=0, m=0)
    value = Y(np.pi / 2, 0.0)
    assert np.isclose(abs(value), 1 / np.sqrt(4 * np.pi))


def test_full_wavefunction_evaluation() -> None:
    qn = QuantumNumbers(n=2, l=1, m=1, s=0.5, j=1.5)
    psi = full_wavefunction(qn)
    # Usar r físico realista (1 Å = 1e-10 m)
    r = 1e-10
    val = psi(r, np.pi / 4, np.pi / 3)
    assert isinstance(val, complex) or np.isscalar(val)
    assert abs(val) > 0.0
