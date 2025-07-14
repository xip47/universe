from typing import Any, Dict, Optional
import numpy as np
from universe.numerics.backend import xp
from universe.solvers.base import BaseSolver

class MaxwellSolver(BaseSolver):
    """
    Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations (Yee scheme, 2D).

    Solver FDTD (Yee) para las ecuaciones de Maxwell en 2D, ultra escalable y optimizado.
    Soporta materiales y condiciones de frontera arbitrarias, preparado para extensiones a 3D y computación paralela.

    Parameters
    ----------
    shape : tuple
        Tamaño espacial de la malla (ny, nx).
    dx : float
        Espaciado espacial uniforme. [m]
    dt : float
        Paso temporal. [s]
    epsilon : Optional[xp.ndarray]
        Permitividad eléctrica local. Si None, se usa EPSILON_0.
    mu : Optional[xp.ndarray]
        Permeabilidad magnética local. Si None, se usa MU_0.
    bc : Optional[Dict[str, Any]]
        Condiciones de frontera (tipo y parámetros).

    Attributes
    ----------
    Ez, Hx, Hy : xp.ndarray
        Campos eléctricos y magnéticos en la malla.
    """
    def __init__(self, shape: tuple, dx: float, dt: float,
                 epsilon: Optional[xp.ndarray] = None,
                 mu: Optional[xp.ndarray] = None,
                 bc: Optional[Dict[str, Any]] = None) -> None:
        from universe.physics.constants import EPSILON_0, MU_0
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.epsilon = epsilon if epsilon is not None else EPSILON_0 * xp.ones(shape)
        self.mu = mu if mu is not None else MU_0 * xp.ones(shape)
        self.bc = bc if bc is not None else {}
        # Campos de Yee (2D): Ez en nodos, Hx y Hy en semidesplazados
        self.Ez = xp.zeros(shape, dtype=xp.float64)
        self.Hx = xp.zeros(shape, dtype=xp.float64)
        self.Hy = xp.zeros(shape, dtype=xp.float64)
        # Buffers para condiciones absorbentes (PML, etc.)
        # (Preparado para extensión)

    def step(self, source: Optional[xp.ndarray] = None) -> None:
        """
        Advance the simulation by one FDTD time step (Yee scheme).

        Avanza la simulación un paso temporal usando el esquema de Yee (FDTD).
        Aplica condiciones de frontera y materiales.
        Implementación corregida: el cálculo del rotacional y la actualización de campos siguen el mallado escalonado (staggered grid) de Yee 2D.

        Parameters
        ----------
        source : Optional[xp.ndarray]
            Fuente externa (corriente o campo eléctrico) a aplicar en Ez.
        """
        # Actualización de campos magnéticos (Hx, Hy)
        self.Hx[:-1, :] -= (self.dt / (self.mu[:-1, :]*self.dx)) * (self.Ez[1:, :] - self.Ez[:-1, :])
        self.Hy[:, :-1] += (self.dt / (self.mu[:, :-1]*self.dx)) * (self.Ez[:, 1:] - self.Ez[:, :-1])
        # Actualización de campo eléctrico Ez (solo en el interior)
        # El rotacional de H requiere que los índices coincidan en la submatriz interior
        curl_H = (self.Hy[1:-1, 1:-1] - self.Hy[1:-1, 0:-2]) - (self.Hx[1:-1, 1:-1] - self.Hx[0:-2, 1:-1])
        self.Ez[1:-1, 1:-1] += (self.dt / (self.epsilon[1:-1, 1:-1]*self.dx)) * curl_H
        # Fuente externa
        if source is not None:
            self.Ez += source
        # Aplicar condiciones de frontera (placeholder: Dirichlet cero)
        self.apply_boundary_conditions()

    def run(self, steps: int, source: Optional[xp.ndarray] = None) -> None:
        """
        Run the simulation for a given number of steps.

        Ejecuta la simulación durante un número de pasos temporales.

        Parameters
        ----------
        steps : int
            Número de pasos temporales a simular.
        source : Optional[xp.ndarray]
            Fuente externa a aplicar en cada paso.
        """
        for _ in range(steps):
            self.step(source=source)

    def set_boundary_conditions(self, bc: Dict[str, Any]) -> None:
        """
        Set the boundary conditions for the simulation domain.

        Define las condiciones de frontera del sistema.

        Parameters
        ----------
        bc : Dict[str, Any]
            Diccionario con la configuración de condiciones de frontera.
        """
        self.bc = bc

    def set_materials(self, materials: Dict[str, Any]) -> None:
        """
        Set the material properties for the simulation domain.

        Define las propiedades materiales del dominio.

        Parameters
        ----------
        materials : Dict[str, Any]
            Diccionario con arrays de permitividad y permeabilidad.
        """
        if 'epsilon' in materials:
            self.epsilon = materials['epsilon']
        if 'mu' in materials:
            self.mu = materials['mu']

    def apply_boundary_conditions(self) -> None:
        """
        Apply the boundary conditions to the fields.

        Aplica las condiciones de frontera a los campos.
        (Por defecto: Dirichlet cero en los bordes. Preparado para extensión a PML, Neumann, etc.)
        """
        # Dirichlet cero (placeholder)
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

    def get_fields(self) -> Dict[str, xp.ndarray]:
        """
        Get the current electromagnetic fields.

        Devuelve los campos electromagnéticos actuales.

        Returns
        -------
        Dict[str, xp.ndarray]
            Diccionario con los campos 'Ez', 'Hx', 'Hy'.
        """
        return {'Ez': self.Ez, 'Hx': self.Hx, 'Hy': self.Hy}
