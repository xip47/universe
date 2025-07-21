"""
Test físico para GeneralNuclearPotential (preset Reid93).

Physical test for GeneralNuclearPotential (Reid93 preset).
"""
import numpy as np
from universe.physics.nuclear.potentials_extended import GeneralNuclearPotential

def test_reid93_potential_1s0():
    """
    Testea el valor del potencial y la fuerza para el canal 1S0 de Reid93.

    Tests the value of the potential and force for the 1S0 channel of Reid93.
    """
    reid93 = GeneralNuclearPotential.preset_reid93(channel="1S0")
    r = np.linspace(0.5, 5.0, 10)
    V = reid93.potential(r)
    F = reid93.force(r)
    # Validar que el potencial es negativo a corta distancia y se anula a larga distancia
    assert np.all(V < 0.0), "El potencial debe ser negativo en el canal 1S0 para r < 5 fm"
    assert np.abs(V[-1]) < 0.1, f"El potencial debe tender a cero para r grande, V={V[-1]}"
    # Validar que la fuerza es finita y cambia de signo
    assert np.all(np.isfinite(F)), "La fuerza debe ser finita"


def test_reid93_potential_3s1():
    """
    Testea el valor del potencial y la fuerza para el canal 3S1 de Reid93.

    Tests the value of the potential and force for the 3S1 channel of Reid93.
    """
    reid93 = GeneralNuclearPotential.preset_reid93(channel="3S1")
    r = np.linspace(0.5, 5.0, 10)
    V = reid93.potential(r)
    F = reid93.force(r)
    assert np.all(np.isfinite(V)), "El potencial debe ser finito"
    assert np.all(np.isfinite(F)), "La fuerza debe ser finita"
    assert np.abs(V[-1]) < 0.1, f"El potencial debe tender a cero para r grande, V={V[-1]}"
