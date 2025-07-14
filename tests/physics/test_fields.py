import pytest
from universe.physics.electromagnetism.fields import electric_field_point
from universe.physics.electromagnetism.fields import electric_field_system
from universe.physics.constants import E_CHARGE
from universe.physics.constants import EPSILON_0
from universe.numerics.backend import xp
import numpy as np

def test_electric_field_point_x_axis():
    # Positive charge at origin, evaluate at x=1
    q = +E_CHARGE
    r_q = (0.0, 0.0, 0.0)
    r_eval = (1.0, 0.0, 0.0)
    E = electric_field_point(q, r_q, r_eval)
    E_expected = (1 / (4 * np.pi * EPSILON_0)) * E_CHARGE / (1.0 ** 2)
    assert np.isclose(xp.asnumpy(E)[0], E_expected)
    assert np.allclose(xp.asnumpy(E)[1:], 0)

def test_electric_field_point_symmetry():
    # Positive charge at origin, evaluate at x=-1
    q = +E_CHARGE
    r_q = (0.0, 0.0, 0.0)
    r_eval = (-1.0, 0.0, 0.0)
    E = electric_field_point(q, r_q, r_eval)
    E_expected = -(1 / (4 * np.pi * EPSILON_0)) * E_CHARGE / (1.0 ** 2)
    assert np.isclose(xp.asnumpy(E)[0], E_expected)
    assert np.allclose(xp.asnumpy(E)[1:], 0)

def test_electric_field_system_superposition():
    # Two equal charges at x=+1 and x=-1, evaluate at origin
    q = +E_CHARGE
    qs = [q, q]
    rs = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    r_eval = (0.0, 0.0, 0.0)
    E = electric_field_system(qs, rs, r_eval)
    # The total field should be zero by symmetry
    assert np.allclose(xp.asnumpy(E), 0)

def test_electric_field_point_singularity():
    # Singularity: evaluate at the position of the charge
    q = +E_CHARGE
    r_q = (0.0, 0.0, 0.0)
    r_eval = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        electric_field_point(q, r_q, r_eval)
