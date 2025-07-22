"""
Sistema N-cuerpos generalizado para simulaciones nucleares escalables.

N-body system for scalable nuclear simulations.

Permite simular cualquier número de nucleones con potenciales de dos y tres cuerpos, usando GPU (Cupy) o CPU (Numpy).
"""

from typing import List, Optional, Callable
from universe.particles.extended_definitions import DynamicParticleExtended
from universe.numerics.backend import xp

class NBodySystem:
    """
    Sistema N-cuerpos para simulaciones nucleares con potenciales de dos y tres cuerpos.

    N-body system for nuclear simulations with two- and three-body potentials.

    Parameters
    ----------
    particles : List[DynamicParticleExtended]
        Lista de partículas dinámicas extendidas.
    two_body_potential : Optional[Callable]
        Potencial de dos cuerpos (por ejemplo, Yukawa, Reid93).
    three_body_potential : Optional[Callable]
        Potencial de tres cuerpos (opcional).
    """
    def __init__(
        self,
        particles: List[DynamicParticleExtended],
        two_body_potential: Optional[Callable] = None,
        three_body_potential: Optional[Callable] = None
    ) -> None:
        self.particles: List[DynamicParticleExtended] = particles
        self.two_body_potential = two_body_potential
        self.three_body_potential = three_body_potential

    def compute_forces(self) -> List[xp.ndarray]:
        """
        Calcula la fuerza total sobre cada partícula (dos y tres cuerpos).

        Returns
        -------
        List[xp.ndarray]
            Lista de fuerzas sobre cada partícula (MeV/fm).
        """
        n = len(self.particles)
        forces = [xp.zeros_like(p.position) for p in self.particles]
        # Fuerzas de dos cuerpos
        for i in range(n):
            for j in range(i + 1, n):
                if self.two_body_potential is not None:
                    r_vec = self.particles[j].position - self.particles[i].position
                    r = float(xp.linalg.norm(r_vec))
                    if r == 0:
                        continue
                    f = float(self.two_body_potential.force(
                        r,
                        spin1=self.particles[i].spin,
                        spin2=self.particles[j].spin,
                        isospin1=self.particles[i].isospin,
                        isospin2=self.particles[j].isospin,
                        quantum_state1=self.particles[i].quantum_state,
                        quantum_state2=self.particles[j].quantum_state
                    ))
                    fij = f * r_vec / r
                    forces[i] += fij
                    forces[j] -= fij  # Acción-reacción
        # Fuerzas de tres cuerpos
        if self.three_body_potential is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        fij, fjk, fki = self.three_body_potential.forces(
                            self.particles[i], self.particles[j], self.particles[k]
                        )
                        forces[i] += fij
                        forces[j] += fjk
                        forces[k] += fki
        return forces

    def step(self, dt: float) -> None:
        """
        Avanza el sistema una vez usando integración de Euler (puede mejorarse a Verlet/Runge-Kutta).

        Parameters
        ----------
        dt : float
            Paso temporal (fm/c).
        """
        forces = self.compute_forces()
        for i, p in enumerate(self.particles):
            a = forces[i] / p.static.mass_mev  # (MeV/fm) / (MeV/c^2) = c^2/fm
            p.velocity += a * dt  # velocidad en fm/c
            p.position += p.velocity * dt  # posición en fm

    def total_energy(self) -> float:
        """
        Calcula la energía total del sistema (cinética + potencial).

        Returns
        -------
        float
            Energía total (MeV).
        """
        kinetic = 0.0
        for p in self.particles:
            v2 = float(xp.sum(p.velocity ** 2))
            kinetic += 0.5 * p.static.mass_mev * v2
        potential = 0.0
        n = len(self.particles)
        # Energía potencial de dos cuerpos
        if self.two_body_potential is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    r = float(xp.linalg.norm(self.particles[j].position - self.particles[i].position))
                    potential += float(self.two_body_potential.potential(
                        r,
                        spin1=self.particles[i].spin,
                        spin2=self.particles[j].spin,
                        isospin1=self.particles[i].isospin,
                        isospin2=self.particles[j].isospin,
                        quantum_state1=self.particles[i].quantum_state,
                        quantum_state2=self.particles[j].quantum_state
                    ))
        # Energía potencial de tres cuerpos
        if self.three_body_potential is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        potential += float(self.three_body_potential.potential(
                            self.particles[i], self.particles[j], self.particles[k]
                        ))
        return kinetic + potential
