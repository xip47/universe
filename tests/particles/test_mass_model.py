"""Pruebas unitarias para el módulo de modelo de masas."""

from universe.particles.mass_model import MassModel


def test_mass_of_electron() -> None:
    mass_mev = MassModel.get_mass("electron", "MeV")
    mass_kg = MassModel.get_mass("electron", "kg")
    assert abs(mass_mev - 0.51099895) < 1e-7
    assert abs(mass_kg - 9.10938356e-31) < 1e-38


def test_mass_of_photon() -> None:
    assert MassModel.get_mass("photon", "MeV") == 0.0
    assert MassModel.get_mass("photon", "kg") == 0.0


def test_mass_range_of_top_quark() -> None:
    top_mass = MassModel.get_mass("top", "MeV")
    assert 172000.0 < top_mass < 173000.0
