"""
Módulo de abstracción numérica compatible con NumPy y CuPy.

Este módulo selecciona dinámicamente el backend numérico (NumPy o CuPy)
dependiendo de la configuración global `CONFIG`, facilitando así el uso
transparente de GPU o CPU según disponibilidad y preferencia.

El alias `xp` actúa como módulo unificado para operaciones básicas,
y se proveen alias comunes para facilitar portabilidad.
"""

from typing import Any, Callable, Literal, Union
from universe.config.simulation_config import CONFIG

# Tipos de precisión aceptados
Precision = Literal["float32", "float64"]

# Tipos de arrays genéricos aceptados
Array = Union["numpy.ndarray", "cupy.ndarray"]


def _select_backend() -> Any:
    """
    Selecciona el backend numérico (CuPy o NumPy) de acuerdo a CONFIG.

    Returns
    -------
    module
        Módulo `cupy` si está disponible y `CONFIG.use_gpu` es True,
        en caso contrario retorna `numpy`.
    """
    if CONFIG.use_gpu:
        try:
            import cupy
            return cupy
        except ImportError:
            import numpy
            CONFIG.force_cpu()
            return numpy
    else:
        import numpy
        return numpy


# Inicialización del backend numérico
xp = _select_backend()

# Mapa de precisión soportada
_precision_map: dict[Precision, type] = {
    "float32": xp.float32,
    "float64": xp.float64,
}

# Tipo de dato activo según CONFIG
DTYPE: type = _precision_map.get(CONFIG.precision, xp.float64)

# === Funciones numéricas unificadas === #

array: Callable[..., Array] = lambda *args, **kwargs: xp.array(*args, dtype=DTYPE, **kwargs)
zeros: Callable[..., Array] = lambda shape, **kwargs: xp.zeros(shape, dtype=DTYPE, **kwargs)
ones: Callable[..., Array] = lambda shape, **kwargs: xp.ones(shape, dtype=DTYPE, **kwargs)
linspace: Callable[..., Array] = lambda start, stop, num=50, **kwargs: xp.linspace(
    start, stop, num, dtype=DTYPE, **kwargs
)
arange: Callable[..., Array] = lambda *args, **kwargs: xp.arange(*args, dtype=DTYPE, **kwargs)

dot: Callable[..., Array] = xp.dot
abs: Callable[..., Array] = xp.abs
sqrt: Callable[..., Array] = xp.sqrt
exp: Callable[..., Array] = xp.exp
log: Callable[..., Array] = xp.log
sin: Callable[..., Array] = xp.sin
cos: Callable[..., Array] = xp.cos
tanh: Callable[..., Array] = xp.tanh
allclose: Callable[..., bool] = xp.allclose

# === Conversión entre GPU/CPU === #

asnumpy: Callable[[Array], Any] = xp.asnumpy if CONFIG.use_gpu else lambda x: x
