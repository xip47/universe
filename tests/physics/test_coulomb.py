import pytest
from universe.physics.electromagnetism.coulomb import coulomb_force
from universe.physics.electromagnetism.coulomb import coulomb_potential
from universe.physics.constants import E_CHARGE
from universe.physics.constants import EPSILON_0
from universe.numerics.backend import xp
import numpy as np

def test_coulomb_force_repulsion():
    # Dos electrones a 1 m
    q1 = -E_CHARGE
    q2 = -E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2 = (1.0, 0.0, 0.0)
    f = coulomb_force(q1, q2, r1, r2)
    f_expected = (1 / (4 * np.pi * EPSILON_0)) * (E_CHARGE ** 2) / (1.0 ** 2)
    assert np.isclose(xp.asnumpy(f)[0], f_expected)
    assert np.allclose(xp.asnumpy(f)[1:], 0)

def test_coulomb_force_attraction():
    # Electrón y protón a 1 m
    q1 = -E_CHARGE
    q2 = +E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2 = (1.0, 0.0, 0.0)
    f = coulomb_force(q1, q2, r1, r2)
    f_expected = -(1 / (4 * np.pi * EPSILON_0)) * (E_CHARGE ** 2) / (1.0 ** 2)
    assert np.isclose(xp.asnumpy(f)[0], f_expected)
    assert np.allclose(xp.asnumpy(f)[1:], 0)

def test_coulomb_force_symmetry():
    # Simetría: F_12 = -F_21
    q1 = +E_CHARGE
    q2 = -E_CHARGE
    r1 = (0.0, 1.0, 0.0)
    r2 = (0.0, 0.0, 0.0)
    f12 = coulomb_force(q1, q2, r1, r2)
    f21 = coulomb_force(q2, q1, r2, r1)
    assert np.allclose(xp.asnumpy(f12), -xp.asnumpy(f21))

def test_coulomb_force_singularity():
    # Singularidad: posiciones coincidentes
    q1 = +E_CHARGE
    q2 = +E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2 = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        coulomb_force(q1, q2, r1, r2)

def test_coulomb_potential():
    # Potencial escalar a 1 m
    q1 = +E_CHARGE
    q2 = -E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2 = (1.0, 0.0, 0.0)
    v = coulomb_potential(q1, q2, r1, r2)
    v_expected = -(1 / (4 * np.pi * EPSILON_0)) * (E_CHARGE ** 2) / 1.0
    assert np.isclose(v, v_expected)
