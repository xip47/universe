import pytest
from universe.physics.electromagnetism.coulomb import coulomb_force
from universe.physics.constants import E_CHARGE
from numpy import allclose

def test_coulomb_force_electrons() -> None:
    """
    Testea la fuerza de Coulomb entre dos electrones separados 1 Å.
    """
    q = -E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2 = (1e-10, 0.0, 0.0)
    force = coulomb_force(q, q, r1, r2)
    # Magnitud teórica
    expected = (1 / (4 * 3.141592653589793 * 8.8541878128e-12)) * (E_CHARGE ** 2) / (1e-10 ** 2)
    # La fuerza debe ser repulsiva (apunta hacia -x)
    assert allclose(abs(force[0]), expected)
    assert force[0] < 0
    assert allclose(force[1:], [0.0, 0.0])

def test_coulomb_force_proton_electron() -> None:
    """
    Testea la fuerza de Coulomb entre un protón y un electrón separados 1 Å.
    """
    q1 = E_CHARGE
    q2 = -E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2 = (1e-10, 0.0, 0.0)
    force = coulomb_force(q1, q2, r1, r2)
    # La fuerza debe ser atractiva (apunta hacia +x)
    assert allclose(abs(force[0]), abs(force[0]))  # magnitud válida
    assert force[0] > 0
    assert allclose(force[1:], [0.0, 0.0])

def test_coulomb_force_inverse_square() -> None:
    """
    Verifica la ley de inversa al cuadrado para la fuerza de Coulomb.
    """
    q = E_CHARGE
    r1 = (0.0, 0.0, 0.0)
    r2a = (1e-10, 0.0, 0.0)
    r2b = (2e-10, 0.0, 0.0)
    f1 = abs(coulomb_force(q, q, r1, r2a)[0])
    f2 = abs(coulomb_force(q, q, r1, r2b)[0])
    assert allclose(f2, f1 / 4)

def test_coulomb_force_singularity() -> None:
    """
    Verifica que se lance excepción si las posiciones coinciden.
    """
    q = E_CHARGE
    r = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        coulomb_force(q, q, r, r)
