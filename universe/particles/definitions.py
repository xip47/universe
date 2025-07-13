from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


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
