from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuración global de simulaciones del universo."""

    use_gpu: bool = True
    precision: str = "float64"
    debug_mode: bool = False
    max_iterations: int = 10_000
    default_backend: str = "cupy"

    @property
    def backend_name(self) -> str:
        """
        Nombre del backend activo.

        Returns
        -------
        str
            'cupy' si se usa GPU, 'numpy' en caso contrario.
        """
        return "cupy" if self.use_gpu else "numpy"

    def toggle_debug(self) -> None:
        """Activa o desactiva el modo debug."""
        self.debug_mode = not self.debug_mode

    def force_cpu(self) -> None:
        """Fuerza el uso de NumPy como backend."""
        self.use_gpu = False
        self.default_backend = "numpy"

    def force_gpu(self) -> None:
        """Fuerza el uso de CuPy como backend."""
        self.use_gpu = True
        self.default_backend = "cupy"


# Instancia global de configuración accesible desde cualquier módulo
CONFIG = SimulationConfig()
