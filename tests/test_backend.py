"""
Tests del módulo numérico backend.

Verifica que el sistema esté utilizando correctamente CuPy o NumPy
según la configuración, y que las operaciones básicas funcionen.
"""

from universe.numerics import backend
from universe.config.simulation_config import CONFIG


def test_backend_operaciones_basicas() -> None:
    """
    Testea operaciones aritméticas básicas y la consistencia de tipo de datos.

    Returns
    -------
    None
    """
    xp = backend.xp
    a = backend.array([1.0, 2.0, 3.0])
    b = backend.array([4.0, 5.0, 6.0])
    resultado = backend.dot(a, b)

    assert resultado == 32.0
    assert hasattr(a, "shape")
    assert a.dtype == backend.DTYPE
    assert callable(backend.zeros)


def test_backend_tipo_y_origen() -> None:
    """
    Informa si se está utilizando CuPy (GPU) o NumPy (CPU).

    Returns
    -------
    None
    """
    modulo = backend.xp.__name__
    tipo = "GPU (CuPy)" if "cupy" in modulo else "CPU (NumPy)"

    print(f"[Diagnóstico Backend] Backend activo: {modulo} → {tipo}")
    print(f"[Diagnóstico Backend] Tipo de precisión: {backend.DTYPE.__name__}")

    assert tipo in {"GPU (CuPy)", "CPU (NumPy)"}


def test_gpu_disponible_si_configurado() -> None:
    """
    Verifica que CUDA esté realmente activo si `use_gpu` es True.

    Returns
    -------
    None
    """
    if CONFIG.use_gpu:
        import cupy
        assert cupy.is_available()

        device_id = cupy.cuda.runtime.getDevice()
        device_props = cupy.cuda.runtime.getDeviceProperties(device_id)
        device_name = device_props["name"]

        print(f"[Verificación GPU] CUDA activo en: {device_name}")
    else:
        print("[Verificación GPU] Ejecutando en CPU (NumPy)")
