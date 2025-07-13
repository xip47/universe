"""
Tests para la configuración global de simulación (`SimulationConfig`).

Verifica el funcionamiento de los métodos de control de backend, precisión,
modo debug y consistencia de atributos.
"""

from universe.config.simulation_config import CONFIG


def test_toggle_debug() -> None:
    """
    Verifica que el modo debug se alterne correctamente.
    """
    original_state = CONFIG.debug_mode
    CONFIG.toggle_debug()
    assert CONFIG.debug_mode is not original_state
    CONFIG.toggle_debug()
    assert CONFIG.debug_mode is original_state


def test_force_cpu() -> None:
    """
    Fuerza el uso de CPU y verifica los valores esperados.
    """
    CONFIG.force_cpu()
    assert CONFIG.use_gpu is False
    assert CONFIG.default_backend == "numpy"


def test_force_gpu() -> None:
    """
    Fuerza el uso de GPU y verifica los valores esperados.
    """
    CONFIG.force_gpu()
    assert CONFIG.use_gpu is True
    assert CONFIG.default_backend == "cupy"


def test_precision_and_iterations() -> None:
    """
    Verifica que los atributos de precisión e iteraciones estén correctamente definidos.
    """
    assert CONFIG.precision in {"float32", "float64"}
    assert isinstance(CONFIG.max_iterations, int)
    assert CONFIG.max_iterations > 0
