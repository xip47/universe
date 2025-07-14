from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from typing import Sequence
from typing import Optional
import numpy as np


class BaseParticle(ABC):
    """
    Clase abstracta para definir una partícula fundamental.

    Attributes
    ----------
    name : str
        Nombre completo de la partícula.
    symbol : str
        Símbolo representativo estándar.
    mass_mev : float
        Masa de la partícula en MeV/c².
    charge : float
        Carga eléctrica en unidades de e.
    spin : float
        Espín intrínseco en múltiplos de ħ.
    generation : int
        Generación dentro del Modelo Estándar (1, 2 o 3).
    """

    name: str
    symbol: str
    mass_mev: float
    charge: float
    spin: float
    generation: int

    @property
    @abstractmethod
    def interaction_type(self) -> str:
        """Tipo de interacción dominante que experimenta la partícula."""
        pass

    @abstractmethod
    def is_fermion(self) -> bool:
        """Retorna True si la partícula es un fermión (espín semientero)."""
        pass

    @abstractmethod
    def is_boson(self) -> bool:
        """Retorna True si la partícula es un bosón (espín entero)."""
        pass


@dataclass(frozen=True)
class Quark(BaseParticle):
    """
    Representación de un quark.

    Attributes
    ----------
    name : str
    symbol : str
    mass_mev : float
    charge : float
    spin : float
    generation : int
    color_charge : bool
        True si participa en interacción fuerte (QCD).
    """

    name: str
    symbol: str
    mass_mev: float
    charge: float
    spin: float
    generation: int
    color_charge: bool = True

    @property
    def interaction_type(self) -> Literal["strong"]:
        return "strong"

    def is_fermion(self) -> bool:
        return True

    def is_boson(self) -> bool:
        return False


@dataclass(frozen=True)
class Lepton(BaseParticle):
    """
    Representación de un leptón.

    Attributes
    ----------
    name : str
    symbol : str
    mass_mev : float
    charge : float
    spin : float
    generation : int
    lepton_number : int
        Conservación del número leptónico (±1).
    """

    name: str
    symbol: str
    mass_mev: float
    charge: float
    spin: float
    generation: int
    lepton_number: int

    @property
    def interaction_type(self) -> Literal["electroweak"]:
        return "electroweak"

    def is_fermion(self) -> bool:
        return True

    def is_boson(self) -> bool:
        return False


@dataclass(frozen=True)
class Boson(BaseParticle):
    """
    Representación de un bosón (gauge o escalar).

    Attributes
    ----------
    name : str
    symbol : str
    mass_mev : float
    charge : float
    spin : float
    generation : int
    force_carrier : Literal["electroweak", "strong", "gravitational"]
        Interacción fundamental que media la partícula.
    """

    name: str
    symbol: str
    mass_mev: float
    charge: float
    spin: float
    generation: int
    force_carrier: Literal["electroweak", "strong", "gravitational"]

    @property
    def interaction_type(self) -> Literal["electroweak", "strong", "gravitational"]:
        return self.force_carrier

    def is_fermion(self) -> bool:
        return False

    def is_boson(self) -> bool:
        return True


@dataclass
class DynamicParticle:
    """
    Representa una partícula fundamental con estado dinámico para simulaciones.

    Hereda propiedades físicas de una instancia de BaseParticle (o sus hijas) y añade atributos dinámicos
    como posición, velocidad y aceleración en el espacio.

    Parameters
    ----------
    static : BaseParticle
        Instancia de la partícula fundamental (Quark, Lepton, Boson).
    position : np.ndarray
        Vector de posición (m) en el espacio (dimensión arbitraria, típicamente 3).
    velocity : np.ndarray
        Vector de velocidad (m/s).
    acceleration : Optional[np.ndarray]
        Vector de aceleración (m/s²), opcional.
    """
    static: BaseParticle
    position: np.ndarray
    velocity: np.ndarray
    acceleration: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.position.shape != self.velocity.shape:
            raise ValueError("La posición y la velocidad deben tener la misma dimensión.")
        if self.acceleration is not None and self.acceleration.shape != self.position.shape:
            raise ValueError("La aceleración debe tener la misma dimensión que la posición.")

class ParticleSystem:
    """
    Sistema de partículas fundamentales con dinámica clásica.

    Permite simular la evolución temporal de un conjunto de partículas bajo fuerzas externas e internas.

    Parameters
    ----------
    particles : Sequence[DynamicParticle]
        Lista de partículas dinámicas.
    """
    def __init__(self, particles: Sequence[DynamicParticle]):
        self.particles = list(particles)
        self.dim = self.particles[0].position.shape[0] if self.particles else 0

    def get_positions(self) -> np.ndarray:
        """Devuelve un array de posiciones de todas las partículas."""
        return np.stack([p.position for p in self.particles])

    def get_velocities(self) -> np.ndarray:
        """Devuelve un array de velocidades de todas las partículas."""
        return np.stack([p.velocity for p in self.particles])

    def step(self, dt: float, force_fn) -> None:
        """
        Avanza el sistema una vez usando integración de Euler.

        Parameters
        ----------
        dt : float
            Paso temporal (s).
        force_fn : Callable[[ParticleSystem], np.ndarray]
            Función que calcula la fuerza sobre cada partícula (N).
        """
        forces = force_fn(self)
        for i, p in enumerate(self.particles):
            a = forces[i] / (p.static.mass_mev * 1.78266192e-30)  # Conversión MeV/c² a kg
            p.velocity += a * dt
            p.position += p.velocity * dt
