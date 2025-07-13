# universe/quantum_states/quantum_numbers.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Final
import math


SpinValue = Literal[0.0, 0.5, 1.0, 1.5, 2.0]  # Escalable
JValue = float


@dataclass(frozen=True)
class QuantumNumbers:
    """
    Representación de un conjunto de números cuánticos para un sistema físico.

    Attributes
    ----------
    n : int
        Número cuántico principal (n ≥ 1).
    l : int
        Número cuántico orbital (l ≥ 0).
    m : int
        Proyección magnética del momento angular orbital (|m| ≤ l).
    s : float
        Espín intrínseco (ej. 0.5 para fermiones).
    j : float
        Momento angular total (j = l ± s).
    """

    n: int
    l: int
    m: int
    s: SpinValue
    j: JValue

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("El número cuántico principal n debe ser ≥ 1")
        if self.l < 0:
            raise ValueError("El número cuántico orbital l debe ser ≥ 0")
        if abs(self.m) > self.l:
            raise ValueError("Debe cumplirse |m| ≤ l")
        if self.j not in self._allowed_j_values():
            raise ValueError(f"El valor j={self.j} no es compatible con l={self.l} y s={self.s}")

    def _allowed_j_values(self) -> set[float]:
        """
        Devuelve el conjunto de valores válidos de j para este l y s.
        """
        jmin = abs(self.l - self.s)
        jmax = self.l + self.s
        steps = int((jmax - jmin) * 2) + 1
        return {round(jmin + 0.5 * i, 1) for i in range(steps)}

    def notation(self) -> str:
        """
        Retorna la notación espectroscópica: nl_j.

        Returns
        -------
        str
            Representación tipo 1s_1/2, 2p_3/2, etc.
        """
        l_symbols = ['s', 'p', 'd', 'f', 'g', 'h']
        l_str = l_symbols[self.l] if self.l < len(l_symbols) else f"l{self.l}"
        return f"{self.n}{l_str}_{int(2 * self.j)}/{2}"
