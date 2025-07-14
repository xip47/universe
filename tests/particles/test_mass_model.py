"""Pruebas unitarias para el módulo de modelo de masas."""

from universe.particles.mass_model import MassModel


def test_mass_of_electron() -> None:
    mass_mev = MassModel.get_mass("electron", "MeV")
    mass_kg = MassModel.get_mass("electron", "kg")
    assert abs(mass_mev - 0.51099895) < 1e-7
    # Tolerancia estricta y realista para punto flotante (1e-37)
    assert abs(mass_kg - 9.10938356e-31) < 1e-37


def test_mass_of_neutrinos() -> None:
    # Valores límite: masas muy pequeñas según PDG 2024
    assert MassModel.get_mass("neutrino_e", "MeV") <= 1e-6  # < 0.8 eV
    assert MassModel.get_mass("neutrino_mu", "MeV") <= 0.19  # < 0.19 MeV
    assert MassModel.get_mass("neutrino_tau", "MeV") <= 18.2  # < 18.2 MeV
    # En kg (usando 1 eV/c² = 1.78266192e-36 kg)
    assert MassModel.get_mass("neutrino_e", "kg") <= 1.78266192e-39
    assert MassModel.get_mass("neutrino_mu", "kg") <= 3.38705765e-34
    assert MassModel.get_mass("neutrino_tau", "kg") <= 3.2432447e-32


def test_mass_of_bosons_w_z() -> None:
    w_mass = MassModel.get_mass("w_boson", "MeV")
    z_mass = MassModel.get_mass("z_boson", "MeV")
    assert 80000 < w_mass < 81000
    assert 91000 < z_mass < 92000
    # En kg
    w_mass_kg = MassModel.get_mass("w_boson", "kg")
    z_mass_kg = MassModel.get_mass("z_boson", "kg")
    assert 1.4e-25 < w_mass_kg < 1.5e-25
    assert 1.6e-25 < z_mass_kg < 1.7e-25


def test_mass_of_photon_is_zero() -> None:
    assert MassModel.get_mass("photon", "MeV") == 0.0
    assert MassModel.get_mass("photon", "kg") == 0.0


def test_invalid_particle_raises() -> None:
    import pytest
    with pytest.raises(KeyError):
        MassModel.get_mass("gravitino", "MeV")


def test_mass_range_of_top_quark() -> None:
    top_mass = MassModel.get_mass("top", "MeV")
    assert 172000.0 < top_mass < 173000.0
