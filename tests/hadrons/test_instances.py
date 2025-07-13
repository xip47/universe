import pytest

from universe.hadrons.instances import HADRONS
from universe.hadrons.instances import proton, neutron


def test_number_of_hadrons() -> None:
    """Verifica que se hayan definido al menos dos hadrones (proton, neutron)."""
    assert len(HADRONS) >= 2


def test_proton_charge() -> None:
    """El protón debe tener carga +1."""
    assert pytest.approx(proton.charge, abs=1e-6) == 1.0


def test_neutron_charge() -> None:
    """El neutrón debe ser eléctricamente neutro."""
    assert pytest.approx(neutron.charge, abs=1e-6) == 0.0


def test_hadron_mass_positive() -> None:
    """Todos los hadrones deben tener masa total positiva."""
    for hadron in HADRONS:
        assert hadron.mass_mev > 0.0, f"{hadron.name} tiene masa negativa"


def test_hadron_type_is_baryon() -> None:
    """Los hadrones definidos hasta ahora deben ser bariones (3 quarks)."""
    for hadron in HADRONS:
        assert hadron.type == "baryon"
        assert len(hadron.quarks) == 3


def test_hadron_generation_consistency() -> None:
    """Verifica que la generación del hadrón sea la máxima entre sus quarks."""
    for hadron in HADRONS:
        generations = [q.generation for q in hadron.quarks]
        assert hadron.generation == max(generations)
