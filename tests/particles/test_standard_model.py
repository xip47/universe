import pytest

from universe.particles.standard_model import (
    QUARKS,
    LEPTONS,
    BOSONS,
    STANDARD_MODEL
)


def test_number_of_quarks() -> None:
    """Debe haber exactamente 6 quarks definidos."""
    assert len(QUARKS) == 6


def test_number_of_leptons() -> None:
    """Debe haber exactamente 6 leptones definidos."""
    assert len(LEPTONS) == 6


def test_number_of_bosons() -> None:
    """Debe haber exactamente 5 bosones gauge + Higgs + graviton."""
    assert len(BOSONS) == 6


def test_standard_model_structure() -> None:
    """El diccionario del modelo estándar debe tener 3 claves principales."""
    assert set(STANDARD_MODEL.keys()) == {"quarks", "leptons", "bosons"}


@pytest.mark.parametrize("particle_list", [QUARKS, LEPTONS, BOSONS])
def test_positive_mass(particle_list) -> None:
    """Todas las partículas deben tener masa ≥ 0."""
    for p in particle_list:
        assert p.mass_mev >= 0.0, f"{p.name} tiene masa negativa"


@pytest.mark.parametrize("particle_list", [QUARKS, LEPTONS, BOSONS])
def test_valid_spin_values(particle_list) -> None:
    """El espín debe ser 0, 0.5, 1 o 2 (bosón hipotético)."""
    for p in particle_list:
        assert p.spin in {0.0, 0.5, 1.0, 2.0}, f"{p.name} tiene espín inválido: {p.spin}"


@pytest.mark.parametrize("particle_list", [QUARKS, LEPTONS, BOSONS])
def test_charge_reasonable(particle_list) -> None:
    """La carga debe estar entre -1 y +2e para partículas del SM."""
    for p in particle_list:
        assert -1.0 <= p.charge <= +2.0, f"{p.name} tiene carga fuera de rango: {p.charge}"
