from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSolver(ABC):
    """
    Abstract base class for all physical solvers in Universe.

    Clase base abstracta para todos los solvers físicos del framework Universe.
    Define la interfaz común y los métodos esenciales para cualquier solver numérico.

    Methods
    -------
    step(**kwargs)
        Avanza la simulación un paso temporal.
    run(steps: int, **kwargs)
        Ejecuta la simulación durante un número de pasos.
    set_boundary_conditions(bc: Dict[str, Any])
        Define las condiciones de frontera del sistema.
    set_materials(materials: Dict[str, Any])
        Define las propiedades materiales del dominio.
    """

    @abstractmethod
    def step(self, **kwargs) -> None:
        """
        Advance the simulation by one time step.

        Avanza la simulación un paso temporal.
        """
        pass

    @abstractmethod
    def run(self, steps: int, **kwargs) -> None:
        """
        Run the simulation for a given number of steps.

        Ejecuta la simulación durante un número de pasos temporales.
        """
        pass

    @abstractmethod
    def set_boundary_conditions(self, bc: Dict[str, Any]) -> None:
        """
        Set the boundary conditions for the simulation domain.

        Define las condiciones de frontera del sistema.
        """
        pass

    @abstractmethod
    def set_materials(self, materials: Dict[str, Any]) -> None:
        """
        Set the material properties for the simulation domain.

        Define las propiedades materiales del dominio.
        """
        pass
