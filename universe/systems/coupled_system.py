"""
Módulo de sistemas acoplados: evolución de partículas bajo fuerzas nucleares y electromagnéticas.
"""

from typing import Callable, List, Optional
from universe.particles.extended_definitions import DynamicParticleExtended
from universe.physics.nuclear.potentials import Reid93Potential, YukawaPotential
from universe.physics.electromagnetism.coulomb import coulomb_force_mevfm, coulomb_potential_mevfm
from universe.numerics.backend import xp

class CoupledSystem:
    """
    Coupled system of extended particles under nuclear and electromagnetic forces.

    Sistema acoplado de partículas extendidas bajo fuerzas nucleares y electromagnéticas.

    Parameters
    ----------
    particles : List[DynamicParticleExtended]
        Lista de partículas dinámicas extendidas.
    nuclear_potential : Callable
        Potencial nuclear (por ejemplo, instancia de Reid93Potential).
    use_coulomb : bool
        Si True, activa la fuerza de Coulomb entre partículas cargadas.
    """
    def __init__(self, particles: List[DynamicParticleExtended], nuclear_potential: Optional[Callable] = None, use_coulomb: bool = True) -> None:
        self.particles: List[DynamicParticleExtended] = particles
        self.nuclear_potential = nuclear_potential
        self.use_coulomb = use_coulomb

    def compute_forces(self) -> List[xp.ndarray]:
        """
        Calcula la fuerza total sobre cada partícula (nuclear + electromagnética) en unidades naturales (MeV/fm).

        Returns
        -------
        List[xp.ndarray]
            Lista de fuerzas sobre cada partícula (MeV/fm).
        """
        n = len(self.particles)
        forces = [xp.zeros_like(p.position) for p in self.particles]
        for i, pi in enumerate(self.particles):
            for j, pj in enumerate(self.particles):
                if i == j:
                    continue
                # Fuerza nuclear (MeV/fm)
                if self.nuclear_potential is not None:
                    r_vec = pj.position - pi.position
                    r = float(xp.linalg.norm(r_vec))
                    if r == 0:
                        continue
                    f_nuc = float(self.nuclear_potential.force(
                        r,
                        spin1=pi.spin, spin2=pj.spin,
                        isospin1=pi.isospin, isospin2=pj.isospin,
                        quantum_state1=pi.quantum_state, quantum_state2=pj.quantum_state
                    ))
                    forces[i] += f_nuc * r_vec / r
                # Fuerza de Coulomb (MeV/fm)
                if self.use_coulomb and hasattr(pi.static, 'charge') and hasattr(pj.static, 'charge'):
                    q1 = getattr(pi.static.charge, 'value', 0.0)
                    q2 = getattr(pj.static.charge, 'value', 0.0)
                    if q1 != 0.0 and q2 != 0.0:
                        f_coul = coulomb_force_mevfm(q1, q2, tuple(xp.asnumpy(pi.position)), tuple(xp.asnumpy(pj.position)))
                        forces[i] += xp.asarray(f_coul)
        return forces

    def step(self, dt: float) -> None:
        """
        Avanza el sistema una vez usando integración de Euler en unidades naturales (fm, MeV, MeV/fm, MeV/c²).

        Parameters
        ----------
        dt : float
            Paso temporal (fm/c).
        """
        forces = self.compute_forces()
        for i, p in enumerate(self.particles):
            a = forces[i] / p.static.mass_mev  # Unidades: (MeV/fm) / (MeV/c²) = c²/fm
            # En unidades naturales c=1, así que a está en 1/fm
            p.velocity += a * dt  # velocidad en fm/c
            p.position += p.velocity * dt  # posición en fm

    def total_energy(self) -> float:
        """
        Calcula la energía total (cinética + nuclear + electromagnética) en MeV.

        Returns
        -------
        float
            Energía total del sistema (MeV).
        """
        KE = sum(0.5 * p.static.mass_mev * float(xp.linalg.norm(p.velocity))**2 for p in self.particles)
        PE_nuc = 0.0
        PE_coul = 0.0
        n = len(self.particles)
        for i in range(n):
            for j in range(i+1, n):
                r = float(xp.linalg.norm(self.particles[j].position - self.particles[i].position))
                if self.nuclear_potential is not None:
                    PE_nuc += float(self.nuclear_potential.potential(
                        r,
                        spin1=self.particles[i].spin, spin2=self.particles[j].spin,
                        isospin1=self.particles[i].isospin, isospin2=self.particles[j].isospin,
                        quantum_state1=self.particles[i].quantum_state, quantum_state2=self.particles[j].quantum_state
                    ))
                if self.use_coulomb and hasattr(self.particles[i].static, 'charge') and hasattr(self.particles[j].static, 'charge'):
                    q1 = getattr(self.particles[i].static.charge, 'value', 0.0)
                    q2 = getattr(self.particles[j].static.charge, 'value', 0.0)
                    if q1 != 0.0 and q2 != 0.0:
                        PE_coul += coulomb_potential_mevfm(q1, q2, tuple(xp.asnumpy(self.particles[i].position)), tuple(xp.asnumpy(self.particles[j].position)))
        return KE + PE_nuc + PE_coul
