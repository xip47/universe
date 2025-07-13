"""
Tests para el módulo de constantes físicas (`physics/constants.py`).

Verifica que las constantes tengan los valores esperados y que la constante
de estructura fina sea coherente con el valor aceptado experimentalmente.
"""

from universe.physics import constants


def test_speed_of_light() -> None:
    """
    Verifica la velocidad de la luz en el vacío.
    """
    assert constants.C == 299_792_458.0


def test_planck_constants() -> None:
    """
    Verifica los valores de las constantes de Planck y reducida.
    """
    assert abs(constants.PLANCK - 6.626_070_15e-34) < 1e-40
    assert abs(constants.HBAR - 1.054_571_817e-34) < 1e-40


def test_mass_and_charge() -> None:
    """
    Verifica masas fundamentales y la carga elemental.
    """
    assert abs(constants.M_ELECTRON - 9.109_383_7015e-31) < 1e-40
    assert abs(constants.M_PROTON - 1.672_621_923_69e-27) < 1e-40
    assert abs(constants.M_NEUTRON - 1.674_927_498_04e-27) < 1e-40
    assert abs(constants.E_CHARGE - 1.602_176_634e-19) < 1e-30


def test_unit_conversions() -> None:
    """
    Verifica constantes de conversión de unidades.
    """
    assert constants.J_TO_MEV == 1 / constants.MEV_TO_J
    assert abs(constants.FM_TO_M - 1e-15) < 1e-20
    assert abs(constants.M_TO_FM - 1e15) < 1e10


def test_fine_structure_constant() -> None:
    """
    Verifica el valor de la constante de estructura fina (α).
    """
    alpha = constants.fine_structure_constant()
    assert abs(alpha - (1 / 137.035999)) < 1e-6
