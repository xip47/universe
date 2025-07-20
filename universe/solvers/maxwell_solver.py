from typing import Any, Dict, Optional, Tuple
import numpy as np
from universe.numerics.backend import xp
from universe.solvers.base import BaseSolver

class MaxwellSolver(BaseSolver):
    """
    Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations (Yee scheme, 2D).

    Solver FDTD (Yee) para las ecuaciones de Maxwell en 2D, ultra escalable y optimizado.
    Soporta materiales, condiciones de frontera arbitrarias y PML, preparado para extensiones a 3D y computación paralela.

    Parameters
    ----------
    shape : tuple of int
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
    pml_thickness : int, optional
        Grosor de la capa PML (en celdas). Por defecto 10.
    pml_sigma_max : float, optional
        Valor máximo de la conductividad en el PML. Por defecto 1.0.
    use_pml : bool, optional
        Si True, activa PML en los bordes. Por defecto False.

    Attributes
    ----------
    Ez : xp.ndarray
        Campo eléctrico en la malla (componente z).
    Hx : xp.ndarray
        Campo magnético en la malla (componente x).
    Hy : xp.ndarray
        Campo magnético en la malla (componente y).
    sigma_x : xp.ndarray
        Perfil de conductividad PML en x.
    sigma_y : xp.ndarray
        Perfil de conductividad PML en y.
    """
    def __init__(
        self,
        shape: Tuple[int, int],
        dx: float,
        dt: float,
        epsilon: Optional[xp.ndarray] = None,
        mu: Optional[xp.ndarray] = None,
        bc: Optional[Dict[str, Any]] = None,
        pml_thickness: int = 10,
        pml_sigma_max: float = 1.0,
        use_pml: bool = False
    ) -> None:
        """
        Initialize the MaxwellSolver with optional PML support.

        Inicializa el solver de Maxwell con soporte opcional para PML.

        Parameters
        ----------
        shape : tuple of int
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
        pml_thickness : int, optional
            Grosor de la capa PML (en celdas). Por defecto 10.
        pml_sigma_max : float, optional
            Valor máximo de la conductividad en el PML. Por defecto 1.0.
        use_pml : bool, optional
            Si True, activa PML en los bordes. Por defecto False.
        """
        from universe.physics.constants import EPSILON_0, MU_0
        self.shape: Tuple[int, int] = shape
        self.dx: float = dx
        self.dt: float = dt
        self.epsilon: xp.ndarray = epsilon if epsilon is not None else EPSILON_0 * xp.ones(shape)
        self.mu: xp.ndarray = mu if mu is not None else MU_0 * xp.ones(shape)
        self.bc: Dict[str, Any] = bc if bc is not None else {}
        self.pml_thickness: int = pml_thickness
        self.pml_sigma_max: float = pml_sigma_max
        self.use_pml: bool = use_pml
        self.Ez: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        self.Hx: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        self.Hy: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        self.sigma_x: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        self.sigma_y: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        if self.use_pml:
            self._init_pml_profiles()
            # Solo Ez se divide en Ezx y Ezy (Berenger TMz)
            self.Ez_x: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Ez_y: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        # Buffers adicionales para PML (campos auxiliares)
        # (Placeholder: se inicializarán en la implementación real)

    def _init_pml_profiles(self) -> None:
        """
        Initialize the sigma profiles for the PML regions.

        Inicializa los perfiles de conductividad sigma en las regiones PML.
        """
        ny, nx = self.shape
        t = self.pml_thickness
        sigma_max = self.pml_sigma_max
        for i in range(t):
            sigma = sigma_max * ((t - i) / t) ** 2
            self.sigma_x[i, :] = sigma
            self.sigma_x[ny - 1 - i, :] = sigma
        for j in range(t):
            sigma = sigma_max * ((t - j) / t) ** 2
            self.sigma_y[:, j] = sigma
            self.sigma_y[:, nx - 1 - j] = sigma

    def step(self, source: Optional[xp.ndarray] = None) -> None:
        """
        Advance the simulation by one FDTD time step (Yee scheme).

        Avanza la simulación un paso temporal usando el esquema de Yee (FDTD).
        Si PML está activado, aplica las ecuaciones modificadas en la región PML (Berenger TMz, minimalista y robusto).

        Parameters
        ----------
        source : Optional[xp.ndarray]
            Fuente externa (corriente o campo eléctrico) a aplicar en Ez.
        """
        if self.use_pml:
            dt = self.dt
            dx = self.dx
            epsilon = self.epsilon
            mu = self.mu
            sigma_x = self.sigma_x
            sigma_y = self.sigma_y
            # --- Actualización de Hx (estándar, pero con σy en PML) ---
            C_hx = (1 - dt * sigma_y[:, :-1] / (2 * mu[:, :-1])) / (1 + dt * sigma_y[:, :-1] / (2 * mu[:, :-1]))
            D_hx = dt / (mu[:, :-1] * dx) / (1 + dt * sigma_y[:, :-1] / (2 * mu[:, :-1]))
            self.Hx[:, :-1] = (
                C_hx * self.Hx[:, :-1]
                - D_hx * (self.Ez[:, 1:] - self.Ez[:, :-1])
            )
            # --- Actualización de Hy (estándar, pero con σx en PML) ---
            C_hy = (1 - dt * sigma_x[:-1, :] / (2 * mu[:-1, :])) / (1 + dt * sigma_x[:-1, :] / (2 * mu[:-1, :]))
            D_hy = dt / (mu[:-1, :] * dx) / (1 + dt * sigma_x[:-1, :] / (2 * mu[:-1, :]))
            self.Hy[:-1, :] = (
                C_hy * self.Hy[:-1, :]
                + D_hy * (self.Ez[1:, :] - self.Ez[:-1, :])
            )
            # --- Actualización de Ez split-field (solo en PML) ---
            C_ex = (1 - dt * sigma_x[1:-1, 1:-1] / (2 * epsilon[1:-1, 1:-1])) / (1 + dt * sigma_x[1:-1, 1:-1] / (2 * epsilon[1:-1, 1:-1]))
            D_ex = dt / (epsilon[1:-1, 1:-1] * dx) / (1 + dt * sigma_x[1:-1, 1:-1] / (2 * epsilon[1:-1, 1:-1]))
            C_ey = (1 - dt * sigma_y[1:-1, 1:-1] / (2 * epsilon[1:-1, 1:-1])) / (1 + dt * sigma_y[1:-1, 1:-1] / (2 * epsilon[1:-1, 1:-1]))
            D_ey = dt / (epsilon[1:-1, 1:-1] * dx) / (1 + dt * sigma_y[1:-1, 1:-1] / (2 * epsilon[1:-1, 1:-1]))
            # Máscara de PML interior
            mask = xp.zeros(self.shape, dtype=bool)
            mask[1:-1, 1:-1] = True
            pml_interior = ((sigma_x[1:-1, 1:-1] > 0) | (sigma_y[1:-1, 1:-1] > 0)) & mask[1:-1, 1:-1]
            not_pml_interior = ~pml_interior
            # Ez_x: absorbe en x, actualiza con dHy/dx
            self.Ez_x[1:-1, 1:-1][pml_interior] = (
                C_ex[pml_interior] * self.Ez_x[1:-1, 1:-1][pml_interior]
                + D_ex[pml_interior] * (self.Hy[1:-1, 1:-1][pml_interior] - self.Hy[0:-2, 1:-1][pml_interior])
            )
            # Ez_y: absorbe en y, actualiza con dHx/dy
            self.Ez_y[1:-1, 1:-1][pml_interior] = (
                C_ey[pml_interior] * self.Ez_y[1:-1, 1:-1][pml_interior]
                - D_ey[pml_interior] * (self.Hx[1:-1, 1:-1][pml_interior] - self.Hx[1:-1, 0:-2][pml_interior])
            )
            # Sumar Ez_x + Ez_y solo en la PML
            self.Ez[1:-1, 1:-1][pml_interior] = (
                self.Ez_x[1:-1, 1:-1][pml_interior] + self.Ez_y[1:-1, 1:-1][pml_interior]
            )
            # Fuera de la PML, usar ecuación estándar
            curl_H = (self.Hy[1:-1, 1:-1] - self.Hy[0:-2, 1:-1]) - (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, 0:-2])
            self.Ez[1:-1, 1:-1][not_pml_interior] += (
                dt / (epsilon[1:-1, 1:-1][not_pml_interior] * dx)
            ) * curl_H[not_pml_interior]
            # Fuente externa
            if source is not None:
                self.Ez += source
        else:
            # FDTD estándar
            self.Hx[:, :-1] -= (self.dt / (self.mu[:, :-1] * self.dx)) * (self.Ez[:, 1:] - self.Ez[:, :-1])
            self.Hy[:-1, :] += (self.dt / (self.mu[:-1, :] * self.dx)) * (self.Ez[1:, :] - self.Ez[:-1, :])
            curl_H = (self.Hy[1:-1, 1:-1] - self.Hy[0:-2, 1:-1]) - (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, 0:-2])
            self.Ez[1:-1, 1:-1] += (self.dt / (self.epsilon[1:-1, 1:-1] * self.dx)) * curl_H
            if source is not None:
                self.Ez += source
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
        Si PML está activado, no aplica Dirichlet en la región PML.
        """
        if self.use_pml:
            # Placeholder: las condiciones de frontera se absorben en la PML
            pass
        else:
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
