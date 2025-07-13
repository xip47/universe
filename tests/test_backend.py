from universe.config.simulation_config import CONFIG
from universe.numerics import backend


def test_backend_basic_operations() -> None:
    """Verifica operaciones básicas de backend y tipo de array."""
    xp = backend.xp
    a = xp.array([1, 2, 3])
    b = xp.array([4, 5, 6])
    c = xp.dot(a, b)

    assert c == 32
    assert hasattr(a, "shape")
    assert callable(xp.zeros)

def test_backend_switching() -> None:
    """Verifica que se pueda cambiar de GPU a CPU dinámicamente."""
    CONFIG.force_cpu()
    assert CONFIG.use_gpu is False
    assert CONFIG.default_backend == "numpy"

    from universe.numerics import backend as backend_cpu
    assert "numpy" in str(type(backend_cpu.xp.array([1])))

    CONFIG.force_gpu()
    assert CONFIG.use_gpu is True
