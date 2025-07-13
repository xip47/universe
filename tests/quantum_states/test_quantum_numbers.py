from universe.quantum_states.quantum_numbers import QuantumNumbers


def test_valid_quantum_state() -> None:
    qn = QuantumNumbers(n=2, l=1, m=0, s=0.5, j=1.5)
    assert qn.notation() == "2p_3/2"


def test_invalid_m_exceeds_l() -> None:
    try:
        QuantumNumbers(n=1, l=0, m=1, s=0.5, j=0.5)
        assert False, "Debió lanzar ValueError"
    except ValueError:
        pass


def test_invalid_j_not_in_allowed_set() -> None:
    try:
        QuantumNumbers(n=2, l=1, m=0, s=0.5, j=2.0)
        assert False, "Debió lanzar ValueError"
    except ValueError:
        pass
