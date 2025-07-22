"""
Potenciales de tres cuerpos para simulaciones nucleares.

Three-body nuclear potentials for nuclear simulations.

Incluye interfaz general y ejemplo simple (potencial tipo Fujita-Miyazawa o dependiente de distancias).
"""

from typing import Tuple
from universe.particles.extended_definitions import DynamicParticleExtended
from universe.numerics.backend import xp

class BaseThreeBodyPotential:
    """
    Interfaz para potenciales de tres cuerpos.

    Three-body potential interface.

    Methods
    -------
    potential(p1, p2, p3) -> float
        Devuelve la energía potencial de tres cuerpos para el trío dado.
    forces(p1, p2, p3) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]
        Devuelve las fuerzas sobre cada partícula del trío.
    """
    def potential(
        self,
        p1: DynamicParticleExtended,
        p2: DynamicParticleExtended,
        p3: DynamicParticleExtended
    ) -> float:
        raise NotImplementedError

    def forces(
        self,
        p1: DynamicParticleExtended,
        p2: DynamicParticleExtended,
        p3: DynamicParticleExtended
    ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        raise NotImplementedError

class SimpleThreeBodyPotential(BaseThreeBodyPotential):
    """
    Potencial de tres cuerpos simple: depende solo de las distancias entre partículas.

    Simple three-body potential: depends only on interparticle distances.

    Parameters
    ----------
    strength : float
        Intensidad del potencial (MeV).
    """
    def __init__(self, strength: float = 1.0) -> None:
        self.strength: float = strength

    def potential(
        self,
        p1: DynamicParticleExtended,
        p2: DynamicParticleExtended,
        p3: DynamicParticleExtended
    ) -> float:
        r12 = float(xp.linalg.norm(p1.position - p2.position))
        r23 = float(xp.linalg.norm(p2.position - p3.position))
        r31 = float(xp.linalg.norm(p3.position - p1.position))
        # Ejemplo: potencial armónico simétrico
        V = self.strength * (r12**2 + r23**2 + r31**2)
        return V

    def forces(
        self,
        p1: DynamicParticleExtended,
        p2: DynamicParticleExtended,
        p3: DynamicParticleExtended
    ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
        # Derivada del potencial respecto a cada posición
        r12 = p1.position - p2.position
        r23 = p2.position - p3.position
        r31 = p3.position - p1.position
        f1 = 2 * self.strength * (r12 - r31)
        f2 = 2 * self.strength * (r23 - r12)
        f3 = 2 * self.strength * (r31 - r23)
        return f1, f2, f3
