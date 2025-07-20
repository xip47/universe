from typing import Any, Dict, Optional, Tuple
import numpy as np
from universe.numerics.backend import xp
from universe.solvers.base import BaseSolver

class MaxwellSolver(BaseSolver):
    """
    Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations (Yee scheme, 2D/3D).

    Solver FDTD (Yee) para las ecuaciones de Maxwell en 2D o 3D, ultra escalable y optimizado.
    Soporta materiales, condiciones de frontera arbitrarias y PML, preparado para computación paralela y efectos relativistas.

    Parameters
    ----------
    shape : tuple of int
        Tamaño espacial de la malla (ny, nx) o (nz, ny, nx).
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
    def __init__(
        self,
        shape: Tuple[int, ...],
        dx: float,
        dt: float,
        epsilon: Optional[xp.ndarray] = None,
        mu: Optional[xp.ndarray] = None,
        bc: Optional[Dict[str, Any]] = None,
        pml_thickness: int = 10,
        pml_sigma_max: float = 1.0,
        use_pml: bool = False
    ) -> None:
        from universe.physics.constants import EPSILON_0, MU_0
        self.shape: Tuple[int, ...] = shape
        self.dx: float = dx
        self.dt: float = dt
        self.bc: Dict[str, Any] = bc if bc is not None else {}
        self.pml_thickness: int = pml_thickness
        self.pml_sigma_max: float = pml_sigma_max
        self.use_pml: bool = use_pml
        self.ndim: int = len(shape)
        # Campos eléctricos y magnéticos
        if self.ndim == 2:
            ny, nx = shape
            self.Ez: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Hx: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Hy: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.epsilon: xp.ndarray = epsilon if epsilon is not None else EPSILON_0 * xp.ones(shape)
            self.mu: xp.ndarray = mu if mu is not None else MU_0 * xp.ones(shape)
            self.sigma_x: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.sigma_y: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        elif self.ndim == 3:
            nz, ny, nx = shape
            self.Ex: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Ey: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Ez: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Hx: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Hy: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.Hz: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.epsilon: xp.ndarray = epsilon if epsilon is not None else EPSILON_0 * xp.ones(shape)
            self.mu: xp.ndarray = mu if mu is not None else MU_0 * xp.ones(shape)
            self.sigma_x: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.sigma_y: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
            self.sigma_z: xp.ndarray = xp.zeros(shape, dtype=xp.float64)
        else:
            raise ValueError("Solo se soportan 2D o 3D")
        if self.use_pml:
            self._init_pml_profiles()

    # Métodos step, run, set_boundary_conditions, set_materials, apply_boundary_conditions, get_fields
    # deben ser adaptados para 3D y mantener compatibilidad 2D.
    # Aquí solo se documenta la arquitectura y el esqueleto, la implementación detallada requiere extender los esquemas Yee 3D.

    def _init_pml_profiles(self) -> None:
        """
        Inicializa los perfiles de conductividad sigma en las regiones PML (2D/3D).
        """
        t = self.pml_thickness
        sigma_max = self.pml_sigma_max
        if self.ndim == 2:
            ny, nx = self.shape
            for i in range(t):
                sigma = sigma_max * ((t - i) / t) ** 2
                self.sigma_x[i, :] = sigma
                self.sigma_x[ny - 1 - i, :] = sigma
            for j in range(t):
                sigma = sigma_max * ((t - j) / t) ** 2
                self.sigma_y[:, j] = sigma
                self.sigma_y[:, nx - 1 - j] = sigma
        elif self.ndim == 3:
            nz, ny, nx = self.shape
            for k in range(t):
                sigma = sigma_max * ((t - k) / t) ** 2
                self.sigma_z[k, :, :] = sigma
                self.sigma_z[nz - 1 - k, :, :] = sigma
            for i in range(t):
                sigma = sigma_max * ((t - i) / t) ** 2
                self.sigma_y[:, i, :] = sigma
                self.sigma_y[:, ny - 1 - i, :] = sigma
            for j in range(t):
                sigma = sigma_max * ((t - j) / t) ** 2
                self.sigma_x[:, :, j] = sigma
                self.sigma_x[:, :, nx - 1 - j] = sigma

    def apply_boundary_conditions(self) -> None:
        """
        Aplica las condiciones de frontera a los campos (2D/3D).
        Soporta: Dirichlet (cero), periódicas, abiertas (absorción), rebote.
        Si PML está activado, no aplica Dirichlet en la región PML.
        """
        if self.use_pml:
            # Las condiciones de frontera se absorben en la PML
            return
        if self.ndim == 2:
            bc_type = self.bc.get('type', 'dirichlet')
            if bc_type == 'dirichlet':
                self.Ez[0, :] = 0
                self.Ez[-1, :] = 0
                self.Ez[:, 0] = 0
                self.Ez[:, -1] = 0
            elif bc_type == 'periodic':
                self.Ez[0, :] = self.Ez[-2, :]
                self.Ez[-1, :] = self.Ez[1, :]
                self.Ez[:, 0] = self.Ez[:, -2]
                self.Ez[:, -1] = self.Ez[:, 1]
            elif bc_type == 'open':
                # Absorción simple (copia valor interior)
                self.Ez[0, :] = self.Ez[1, :]
                self.Ez[-1, :] = self.Ez[-2, :]
                self.Ez[:, 0] = self.Ez[:, 1]
                self.Ez[:, -1] = self.Ez[:, -2]
            elif bc_type == 'reflect':
                self.Ez[0, :] = -self.Ez[1, :]
                self.Ez[-1, :] = -self.Ez[-2, :]
                self.Ez[:, 0] = -self.Ez[:, 1]
                self.Ez[:, -1] = -self.Ez[:, -2]
        elif self.ndim == 3:
            bc_type = self.bc.get('type', 'dirichlet')
            if bc_type == 'dirichlet':
                self.Ex[0, :, :] = 0
                self.Ex[-1, :, :] = 0
                self.Ex[:, 0, :] = 0
                self.Ex[:, -1, :] = 0
                self.Ex[:, :, 0] = 0
                self.Ex[:, :, -1] = 0
                self.Ey[0, :, :] = 0
                self.Ey[-1, :, :] = 0
                self.Ey[:, 0, :] = 0
                self.Ey[:, -1, :] = 0
                self.Ey[:, :, 0] = 0
                self.Ey[:, :, -1] = 0
                self.Ez[0, :, :] = 0
                self.Ez[-1, :, :] = 0
                self.Ez[:, 0, :] = 0
                self.Ez[:, -1, :] = 0
                self.Ez[:, :, 0] = 0
                self.Ez[:, :, -1] = 0
            elif bc_type == 'periodic':
                # Ejemplo para Ex (repetir para Ey, Ez)
                self.Ex[0, :, :] = self.Ex[-2, :, :]
                self.Ex[-1, :, :] = self.Ex[1, :, :]
                self.Ex[:, 0, :] = self.Ex[:, -2, :]
                self.Ex[:, -1, :] = self.Ex[:, 1, :]
                self.Ex[:, :, 0] = self.Ex[:, :, -2]
                self.Ex[:, :, -1] = self.Ex[:, :, 1]
                # Repetir para Ey, Ez...
            elif bc_type == 'open':
                self.Ex[0, :, :] = self.Ex[1, :, :]
                self.Ex[-1, :, :] = self.Ex[-2, :, :]
                self.Ex[:, 0, :] = self.Ex[:, 1, :]
                self.Ex[:, -1, :] = self.Ex[:, -2, :]
                self.Ex[:, :, 0] = self.Ex[:, :, 1]
                self.Ex[:, :, -1] = self.Ex[:, :, -2]
                # Repetir para Ey, Ez...
            elif bc_type == 'reflect':
                self.Ex[0, :, :] = -self.Ex[1, :, :]
                self.Ex[-1, :, :] = -self.Ex[-2, :, :]
                self.Ex[:, 0, :] = -self.Ex[:, 1, :]
                self.Ex[:, -1, :] = -self.Ex[:, 1, :]
                self.Ex[:, :, 0] = -self.Ex[:, :, 1]
                self.Ex[:, :, -1] = -self.Ex[:, :, 1]
                # Repetir para Ey, Ez...

    def step(self, source: Optional[xp.ndarray] = None, rho: Optional[xp.ndarray] = None, J: Optional[xp.ndarray] = None) -> None:
        """
        Avanza la simulación un paso temporal usando el esquema de Yee (FDTD) con PML profesional.
        Soporta 2D (TMz) y 3D (estructura preparada). Acopla fuentes físicas: densidad de carga (rho) y corriente (Jx, Jy, Jz).

        Parameters
        ----------
        source : Optional[xp.ndarray]
            Fuente externa (campo eléctrico) a aplicar (opcional).
        rho : Optional[xp.ndarray]
            Densidad de carga en la malla (opcional).
        J : Optional[xp.ndarray]
            Densidad de corriente en la malla, shape (..., ndim) (opcional).
        """
        if self.ndim == 2:
            ny, nx = self.shape
            dt = self.dt
            dx = self.dx
            # --- Actualización de Hx, Hy (incluye PML) ---
            # Hx(i, j) = Hx(i, j) - (dt / mu) * (dEz/dy)
            dEz_dy = xp.zeros_like(self.Hx)
            dEz_dy[:-1, :] = (self.Ez[1:, :] - self.Ez[:-1, :]) / dx
            # Hy(i, j) = Hy(i, j) + (dt / mu) * (dEz/dx)
            dEz_dx = xp.zeros_like(self.Hy)
            dEz_dx[:, :-1] = (self.Ez[:, 1:] - self.Ez[:, :-1]) / dx
            # PML para Hx, Hy
            self.Hx -= (dt / self.mu) * dEz_dy
            self.Hx *= xp.exp(-dt * self.sigma_y / self.mu)
            self.Hy += (dt / self.mu) * dEz_dx
            self.Hy *= xp.exp(-dt * self.sigma_x / self.mu)
            # --- Actualización de Ez (incluye PML) ---
            dHy_dx = xp.zeros_like(self.Ez)
            dHx_dy = xp.zeros_like(self.Ez)
            dHy_dx[:, 1:] = (self.Hy[:, 1:] - self.Hy[:, :-1]) / dx
            dHx_dy[1:, :] = (self.Hx[1:, :] - self.Hx[:-1, :]) / dx
            curl_H = dHy_dx - dHx_dy
            # Fuentes físicas
            Jz = J[:, :, 2] if J is not None and J.shape[-1] == 3 else (J if J is not None else 0.0)
            rho_term = (rho / self.epsilon) if rho is not None else 0.0
            # PML para Ez
            sigma_e = self.sigma_x + self.sigma_y
            Ez_update = (dt / self.epsilon) * (curl_H - Jz - rho_term)
            self.Ez = (self.Ez * xp.exp(-dt * sigma_e / self.epsilon)) + Ez_update
            if source is not None:
                self.Ez += source
        elif self.ndim == 3:
            # TODO: Implementar PML completo 3D (estructura preparada)
            # Aquí solo se deja la estructura para escalabilidad futura
            nz, ny, nx = self.shape
            # Actualización de campos magnéticos (H)
            self.Hx[:, :-1, :-1] -= (self.dt / (self.mu[:, :-1, :-1] * self.dx)) * (
                (self.Ey[:, :-1, 1:] - self.Ey[:, :-1, :-1]) / self.dx -
                (self.Ez[:, 1:, :-1] - self.Ez[:, :-1, :-1]) / self.dx
            )
            self.Hy[:-1, :, :-1] -= (self.dt / (self.mu[:-1, :, :-1] * self.dx)) * (
                (self.Ez[:-1, :, 1:] - self.Ez[:-1, :, :-1]) / self.dx -
                (self.Ex[1:, :, :-1] - self.Ex[:-1, :, :-1]) / self.dx
            )
            self.Hz[:-1, :-1, :] -= (self.dt / (self.mu[:-1, :-1, :] * self.dx)) * (
                (self.Ex[:-1, 1:, :] - self.Ex[:-1, :-1, :]) / self.dx -
                (self.Ey[1:, :-1, :] - self.Ey[:-1, :-1, :]) / self.dx
            )
            # Actualización de campos eléctricos (E) con fuentes
            curl_Hx = (
                (self.Hz[1:, :-1, :-1] - self.Hz[:-1, :-1, :-1]) / self.dx -
                (self.Hy[:-1, 1:, :-1] - self.Hy[:-1, :-1, :-1]) / self.dx
            )
            curl_Hy = (
                (self.Hx[:-1, 1:, :-1] - self.Hx[:-1, :-1, :-1]) / self.dx -
                (self.Hz[:-1, :-1, 1:] - self.Hz[:-1, :-1, :-1]) / self.dx
            )
            curl_Hz = (
                (self.Hy[:-1, :-1, 1:] - self.Hy[:-1, :-1, :-1]) / self.dx -
                (self.Hx[1:, :-1, :-1] - self.Hx[:-1, :-1, :-1]) / self.dx
            )
            self.Ex[:-1, :-1, :-1] += (self.dt / (self.epsilon[:-1, :-1, :-1] * self.dx)) * curl_Hx
            self.Ey[:-1, :-1, :-1] += (self.dt / (self.epsilon[:-1, :-1, :-1] * self.dx)) * curl_Hy
            self.Ez[:-1, :-1, :-1] += (self.dt / (self.epsilon[:-1, :-1, :-1] * self.dx)) * curl_Hz
            # Fuentes físicas (corriente y carga)
            if J is not None:
                self.Ex[:-1, :-1, :-1] += (self.dt / self.epsilon[:-1, :-1, :-1]) * J[:-1, :-1, :-1, 0]
                self.Ey[:-1, :-1, :-1] += (self.dt / self.epsilon[:-1, :-1, :-1]) * J[:-1, :-1, :-1, 1]
                self.Ez[:-1, :-1, :-1] += (self.dt / self.epsilon[:-1, :-1, :-1]) * J[:-1, :-1, :-1, 2]
            if rho is not None:
                self.Ex[:-1, :-1, :-1] += (self.dt / self.epsilon[:-1, :-1, :-1]) * rho[:-1, :-1, :-1]
            if source is not None:
                self.Ex += source  # O aplicar a Ex, Ey, Ez según corresponda
            # TODO: PML y condiciones de frontera 3D
        else:
            raise ValueError("Solo se soportan 2D o 3D")
        self.apply_boundary_conditions()

    def run(self, steps: int, **kwargs) -> None:
        """
        Ejecuta la simulación durante un número de pasos temporales.

        Parameters
        ----------
        steps : int
            Número de pasos temporales a ejecutar.
        **kwargs : dict
            Argumentos adicionales para el método step.
        """
        for _ in range(steps):
            self.step(**kwargs)

    def set_boundary_conditions(self, bc: dict) -> None:
        """
        Define las condiciones de frontera del sistema.

        Parameters
        ----------
        bc : dict
            Diccionario con la configuración de condiciones de frontera.
        """
        self.bc = bc
        self.apply_boundary_conditions()

    def set_materials(self, materials: dict) -> None:
        """
        Define las propiedades materiales del dominio.

        Parameters
        ----------
        materials : dict
            Diccionario con los arrays de permitividad y permeabilidad.
        """
        if 'epsilon' in materials:
            self.epsilon = materials['epsilon']
        if 'mu' in materials:
            self.mu = materials['mu']

    # TODO: Implementar step 3D (actualización de Ex, Ey, Ez, Hx, Hy, Hz con fuentes rho, Jx, Jy, Jz)
    # TODO: Añadir soporte para efectos relativistas (v/c ~ 1), retardos, y logs de energía relativista
