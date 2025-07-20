"""
Definiciones extendidas de partículas con todos los grados de libertad.
"""

from typing import Optional
from universe.numerics.backend import xp
from universe.particles.degrees_of_freedom import Spin
from universe.particles.degrees_of_freedom import Isospin
from universe.particles.degrees_of_freedom import Charge
from universe.particles.degrees_of_freedom import QuantumState

class BaseParticleExtended:
    """
    Base class for a fundamental particle with all degrees of freedom.

    Clase base para una partícula fundamental con todos los grados de libertad.

    Parameters
    ----------
    name : str
        Nombre de la partícula.
    mass_mev : float
        Masa en MeV/c².
    charge : Charge
        Carga eléctrica.
    spin : Spin
        Spin.
    isospin : Isospin
        Isospin.
    quantum_state : Optional[QuantumState]
        Estado cuántico (opcional).
    """
    def __init__(self, name: str, mass_mev: float, charge: Charge, spin: Spin, isospin: Isospin, quantum_state: Optional[QuantumState] = None) -> None:
        self.name: str = name
        self.mass_mev: float = mass_mev
        self.charge: Charge = charge
        self.spin: Spin = spin
        self.isospin: Isospin = isospin
        self.quantum_state: Optional[QuantumState] = quantum_state

class DynamicParticleExtended:
    """
    Dynamic particle with all degrees of freedom for simulation.

    Partícula dinámica con todos los grados de libertad para simulación.

    Parameters
    ----------
    static : BaseParticleExtended
        Partícula base con propiedades fundamentales.
    position : xp.ndarray
        Vector de posición (fm).
    velocity : xp.ndarray
        Vector de velocidad (fm/s).
    spin : Spin
        Spin dinámico.
    isospin : Isospin
        Isospin dinámico.
    quantum_state : Optional[QuantumState]
        Estado cuántico (opcional).
    """
    def __init__(self, static: BaseParticleExtended, position: xp.ndarray, velocity: xp.ndarray, spin: Spin, isospin: Isospin, quantum_state: Optional[QuantumState] = None) -> None:
        self.static: BaseParticleExtended = static
        self.position: xp.ndarray = position
        self.velocity: xp.ndarray = velocity
        self.spin: Spin = spin
        self.isospin: Isospin = isospin
        self.quantum_state: Optional[QuantumState] = quantum_state
