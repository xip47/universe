from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from universe.particles.definitions import Quark


@dataclass(frozen=True)
class Hadron:
    """
    Partícula compuesta por quarks (bariones o mesones).

    Attributes
    ----------
    name : str
        Nombre del hadrón (ej: 'proton').
    symbol : str
        Símbolo representativo (ej: 'p⁺').
    quarks : tuple[Quark, ...]
        Tupla de quarks constituyentes.
    type : Literal['baryon', 'meson']
        Tipo de hadrón: barión (3 quarks) o mesón (quark + antiquark).
    """

    name: str
    symbol: str
    quarks: tuple[Quark, ...]
    type: Literal["baryon", "meson"]

    @property
    def charge(self) -> float:
        """Carga total del hadrón en unidades de e."""
        return sum(q.charge for q in self.quarks)

    @property
    def mass_mev(self) -> float:
        """
        Masa aproximada del hadrón en MeV/c².
        Suma de masas de quarks (sin energía de ligadura aún).
        """
        return sum(q.mass_mev for q in self.quarks)

    @property
    def generation(self) -> int:
        """Generación más alta entre los quarks constituyentes."""
        return max(q.generation for q in self.quarks)
