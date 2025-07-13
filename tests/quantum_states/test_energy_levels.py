import pytest
from universe.quantum_states.energy_levels import hydrogenic_energy

def test_hydrogenic_energy_ground_state() -> None:
    E = hydrogenic_energy(n=1)
    assert E < 0.0
    assert abs(E + 13.6057) < 1e-3

def test_hydrogenic_energy_excited() -> None:
    E2 = hydrogenic_energy(n=2)
    E3 = hydrogenic_energy(n=3)
    assert E3 > E2  # E3 menos negativo que E2
    assert E2 > -13.7  # energía realista
    assert abs(E2 + 3.4014) < 1e-2

def test_invalid_n_value() -> None:
    with pytest.raises(ValueError):
        hydrogenic_energy(n=0)
