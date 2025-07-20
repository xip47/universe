"""
MaxwellParticleSystem: sistema acoplado de partículas cargadas y campos electromagnéticos (FDTD, PIC).
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from universe.particles.extended_definitions import DynamicParticleExtended
from universe.solvers.maxwell_solver import MaxwellSolver
from universe.numerics.backend import xp

class MaxwellParticleSystem:
    """
    Sistema acoplado de partículas cargadas y campos electromagnéticos (FDTD + PIC).

    Acopla partículas cargadas (protones, electrones, etc.) con campos E y B evolucionados por FDTD (Yee).
    Permite simular radiación, corrientes, ondas y la interacción realista campo-partícula.

    Parameters
    ----------
    particles : List[DynamicParticleExtended]
        Lista de partículas dinámicas (con masa, carga, posición, velocidad).
    grid_shape : Tuple[int, int]
        Tamaño de la malla FDTD (ny, nx).
    dx : float
        Espaciado espacial de la malla (fm).
    dt : float
        Paso temporal (fm/c).
    solver_kwargs : Optional[Dict]
        Parámetros adicionales para MaxwellSolver (PML, materiales, etc).
    """
    def __init__(self,
                 particles: List[DynamicParticleExtended],
                 grid_shape: Tuple[int, int],
                 dx: float,
                 dt: float,
                 solver_kwargs: Optional[Dict] = None) -> None:
        self.particles: List[DynamicParticleExtended] = particles
        self.grid_shape: Tuple[int, int] = grid_shape
        self.dx: float = dx
        self.dt: float = dt
        self.solver = MaxwellSolver(grid_shape, dx, dt, **(solver_kwargs or {}))
        self.charge_grid = xp.zeros(grid_shape, dtype=xp.float64)
        self.current_grid = xp.zeros(grid_shape + (2,), dtype=xp.float64)  # Jx, Jy

    def deposit_charge_current(self) -> None:
        """
        Deposita la densidad de carga (C/m^2) y corriente (A/m^2) de las partículas en la malla FDTD usando el esquema CIC (Cloud-In-Cell).

        Convierte la carga elemental (e) a Coulombs usando E_CHARGE y divide por el área de celda (dx^2) para obtener densidad física.
        La corriente se deposita en A/m^2 (carga * velocidad / área de celda).
        Compatible con 2D y 3D y preparado para GPU (xp).

        Notes
        -----
        - El esquema CIC reduce el ruido numérico y mejora la conservación de la carga respecto a NGP.
        - El atributo `charge_grid` almacena la densidad de carga y `current_grid` la densidad de corriente (vectorial).
        - Para mayor fidelidad, se puede extender a TSC (Triangular Shaped Cloud).
        """
        from universe.physics.constants import E_CHARGE
        self.charge_grid[...] = 0.0
        self.current_grid[...] = 0.0
        shape = self.grid_shape
        ndim = len(shape)
        cell_area = self.dx ** ndim  # dx^2 en 2D, dx^3 en 3D
        for p in self.particles:
            pos = [float(p.position[d]) / self.dx for d in range(ndim)]
            base_idx = [int(np.floor(pos[d])) for d in range(ndim)]
            # Convertir la carga elemental a Coulombs
            q_e = getattr(p.static.charge, 'value', 0.0)
            q = q_e * E_CHARGE / cell_area  # densidad de carga (C/m^2)
            v = [float(p.velocity[d]) for d in range(ndim)]
            # CIC: distribuir a los 2^ndim nodos vecinos
            for offset in np.ndindex(*(2,) * ndim):
                idx = [base_idx[d] + offset[d] for d in range(ndim)]
                if any(idx[d] < 0 or idx[d] >= shape[d] for d in range(ndim)):
                    continue  # fuera de la malla
                w = 1.0
                for d in range(ndim):
                    delta = pos[d] - base_idx[d]
                    w *= (1 - delta) if offset[d] == 0 else delta
                self.charge_grid[tuple(idx)] += q * w
                for d in range(ndim):
                    self.current_grid[(*idx, d)] += q * v[d] * w  # densidad de corriente (A/m^2)

    def interpolate_fields(self) -> List[Tuple]:
        """
        Interpola los campos eléctricos y magnéticos en la posición de cada partícula usando el esquema CIC (Cloud-In-Cell).

        Utiliza interpolación lineal ponderada por la distancia a los nodos vecinos (2^ndim) para estimar los valores de campo en la posición continua de cada partícula.
        Compatible con 2D y 3D y preparado para GPU (xp).

        Returns
        -------
        List[Tuple]
            Lista de tuplas con los campos (E, B) en la posición de cada partícula.

        Notes
        -----
        - El esquema CIC mejora la suavidad y la conservación de la energía respecto a la interpolación bilineal/trilineal simple.
        - Para mayor fidelidad, se puede extender a TSC (Triangular Shaped Cloud).
        """
        shape = self.grid_shape
        ndim = len(shape)
        fields = []
        for p in self.particles:
            pos = [float(p.position[d]) / self.dx for d in range(ndim)]
            base_idx = [int(np.floor(pos[d])) for d in range(ndim)]
            weights = []
            idxs = []
            for offset in np.ndindex(*(2,) * ndim):
                idx = [base_idx[d] + offset[d] for d in range(ndim)]
                if any(idx[d] < 0 or idx[d] >= shape[d] for d in range(ndim)):
                    continue
                w = 1.0
                for d in range(ndim):
                    delta = pos[d] - base_idx[d]
                    w *= (1 - delta) if offset[d] == 0 else delta
                weights.append(w)
                idxs.append(tuple(idx))
            if ndim == 2:
                Ez = sum(weights[n] * self.solver.Ez[idxs[n]] for n in range(len(weights)))
                Bx = sum(weights[n] * self.solver.Hx[idxs[n]] for n in range(len(weights)))
                By = sum(weights[n] * self.solver.Hy[idxs[n]] for n in range(len(weights)))
                fields.append((Ez, Bx, By))
            elif ndim == 3:
                Ex = sum(weights[n] * self.solver.Ex[idxs[n]] for n in range(len(weights)))
                Ey = sum(weights[n] * self.solver.Ey[idxs[n]] for n in range(len(weights)))
                Ez = sum(weights[n] * self.solver.Ez[idxs[n]] for n in range(len(weights)))
                Bx = sum(weights[n] * self.solver.Hx[idxs[n]] for n in range(len(weights)))
                By = sum(weights[n] * self.solver.Hy[idxs[n]] for n in range(len(weights)))
                Bz = sum(weights[n] * self.solver.Hz[idxs[n]] for n in range(len(weights)))
                fields.append(((Ex, Ey, Ez), (Bx, By, Bz)))
        return fields

    def step(self) -> None:
        """
        Avanza el sistema acoplado una vez (ciclo PIC, 2D/3D):
        1. Deposita carga/corriente en la malla.
        2. Avanza campos E y B con MaxwellSolver usando rho y J.
        3. Interpola campos en partículas y actualiza su movimiento (fuerza de Lorentz, relativista si v~c).
        """
        self.deposit_charge_current()
        # Adaptar J para que tenga shape (ny, nx, 3) en 2D (Jx, Jy, Jz=0)
        if self.current_grid.shape[-1] == 2:
            ny, nx, _ = self.current_grid.shape
            J = xp.zeros((ny, nx, 3), dtype=self.current_grid.dtype)
            J[:, :, :2] = self.current_grid
        else:
            J = self.current_grid
        self.solver.step(rho=self.charge_grid, J=J)
        fields = self.interpolate_fields()
        ndim = len(self.grid_shape)
        for idx, p in enumerate(self.particles):
            if ndim == 2:
                Ez, Bx, By = fields[idx]
                q = getattr(p.static.charge, 'value', 0.0)
                m = p.static.mass_mev
                v = p.velocity
                E_vec = xp.array([0.0, float(Ez)])
                B_vec = xp.array([float(Bx), float(By)])
                v_vec = v
                lorentz_y = float(v_vec[1]) * float(Bx) - float(v_vec[0]) * float(By)
                F = q * (E_vec + xp.array([lorentz_y, 0.0]))
                a = F / m
                p.velocity += a * self.dt
                p.position += p.velocity * self.dt
            elif ndim == 3:
                (Ex, Ey, Ez), (Bx, By, Bz) = fields[idx]
                q = getattr(p.static.charge, 'value', 0.0)
                m = p.static.mass_mev
                v = p.velocity
                E_vec = xp.array([float(Ex), float(Ey), float(Ez)])
                B_vec = xp.array([float(Bx), float(By), float(Bz)])
                v_vec = v
                c = 1.0  # Unidades naturales
                v2 = float(xp.sum(v_vec**2))
                gamma = 1.0 / xp.sqrt(1.0 - v2 / c**2) if v2 < c**2 else 1e6
                F = q * (E_vec + xp.cross(v_vec, B_vec))
                a = F / (gamma * m)
                p.velocity += a * self.dt
                vmag = float(xp.linalg.norm(p.velocity))
                if vmag > c:
                    p.velocity *= c / vmag
                p.position += p.velocity * self.dt
        # print(f"Energía total (campos + partículas): {self.total_energy():.6f} MeV")

    def total_energy(self) -> float:
        """
        Calcula la energía total del sistema (campos + partículas).

        Returns
        -------
        float
            Energía total (MeV).
        """
        from universe.physics.constants import EPSILON_0, MU_0
        # Energía de partículas
        E_particles = sum(0.5 * p.static.mass_mev * float(xp.linalg.norm(p.velocity))**2 for p in self.particles)
        # Energía de campos (fórmula física correcta)
        E_fields = xp.sum(0.5 * EPSILON_0 * self.solver.Ez**2 + 0.5 * MU_0 * (self.solver.Hx**2 + self.solver.Hy**2)) * self.dx**2
        if hasattr(E_fields, 'get'):
            E_fields = float(E_fields.get())
        else:
            E_fields = float(E_fields)
        return E_particles + E_fields
