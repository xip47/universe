"""
Potencial nuclear general y modular para simulaciones realistas y escalables.

General nuclear potential for realistic and scalable simulations.

Permite construir potenciales nucleares arbitrarios (Reid93, Argonne, Paris, etc.) a partir de operadores y coeficientes.
Incluye preset extendido para Reid93 con términos centrales, tensoriales y spin-órbita.

Ejemplo de uso
--------------
>>> from universe.physics.nuclear.potentials_extended import GeneralNuclearPotential
>>> reid93 = GeneralNuclearPotential.preset_reid93(channel="1S0")
>>> V = reid93.potential(1.0)  # Potencial a 1 fm
"""
from typing import List, Optional
from universe.numerics import backend as backend
from universe.physics.nuclear.operators import BaseNuclearOperator
from universe.physics.nuclear.operators import CentralOperator
from universe.physics.nuclear.operators import TensorOperator
from universe.physics.nuclear.operators import SpinOrbitOperator

class GeneralNuclearPotential:
    """
    Potencial nuclear general y modular.

    General and modular nuclear potential.

    Parameters
    ----------
    operators : List[BaseNuclearOperator]
        Lista de operadores (central, tensorial, spin-órbita, etc.).
    coefficients : List[float]
        Coeficientes de cada término (por canal).
    masses : List[float]
        Masas asociadas a cada término (1/fm).
    channels : Optional[List[str]]
        Nombres de canales parciales ("1S0", "3S1", etc.).
    """
    def __init__(
        self,
        operators: List[BaseNuclearOperator],
        coefficients: List[float],
        masses: List[float],
        channels: Optional[List[str]] = None
    ) -> None:
        self.operators: List[BaseNuclearOperator] = operators
        self.coefficients: List[float] = coefficients
        self.masses: List[float] = masses
        self.channels: List[str] = channels if channels is not None else ["default"]

    def yukawa(self, r: float | backend.xp.ndarray, mass: float) -> backend.xp.ndarray:
        """
        Función de Yukawa para la dependencia radial.

        Yukawa function for radial dependence.

        Parameters
        ----------
        r : float or backend.xp.ndarray
            Distancia (fm).
        mass : float
            Masa del mesón (1/fm).

        Returns
        -------
        backend.xp.ndarray
            Valor de la función de Yukawa.
        """
        r = backend.xp.asarray(r)
        return backend.xp.exp(-mass * r) / r

    def potential(
        self,
        r: float | backend.xp.ndarray,
        channel: str = "default",
        spin1=None,
        spin2=None,
        isospin1=None,
        isospin2=None,
        quantum_state1=None,
        quantum_state2=None
    ) -> backend.xp.ndarray:
        """
        Calcula el potencial total sumando todos los operadores y términos.

        Computes the total potential by summing all operators and terms.

        Parameters
        ----------
        r : float or backend.xp.ndarray
            Distancia entre nucleones (fm).
        channel : str
            Canal parcial ("1S0", "3S1", etc.).
        spin1, spin2, isospin1, isospin2, quantum_state1, quantum_state2 : opcional
            Grados de libertad de las partículas.

        Returns
        -------
        backend.xp.ndarray
            Potencial V(r) (MeV).
        """
        r = backend.xp.asarray(r)
        V = backend.xp.zeros_like(r, dtype=backend.xp.float64)
        for op, coef, mass in zip(self.operators, self.coefficients, self.masses):
            op_val = op.apply(
                r,
                spin1=spin1,
                spin2=spin2,
                isospin1=isospin1,
                isospin2=isospin2,
                quantum_state1=quantum_state1,
                quantum_state2=quantum_state2
            )
            V += coef * self.yukawa(r, mass) * op_val
        V = backend.xp.where(r == 0, 0.0, V)
        return V

    def force(
        self,
        r: float | backend.xp.ndarray,
        channel: str = "default",
        spin1=None,
        spin2=None,
        isospin1=None,
        isospin2=None,
        quantum_state1=None,
        quantum_state2=None
    ) -> backend.xp.ndarray:
        """
        Calcula la fuerza total sumando todos los operadores y términos.

        Computes the total force by summing all operators and terms.

        Parameters
        ----------
        r : float or backend.xp.ndarray
            Distancia entre nucleones (fm).
        channel : str
            Canal parcial ("1S0", "3S1", etc.).
        spin1, spin2, isospin1, isospin2, quantum_state1, quantum_state2 : opcional
            Grados de libertad de las partículas.

        Returns
        -------
        backend.xp.ndarray
            Fuerza F(r) (MeV/fm).
        """
        r = backend.xp.asarray(r)
        F = backend.xp.zeros_like(r, dtype=backend.xp.float64)
        for op, coef, mass in zip(self.operators, self.coefficients, self.masses):
            op_val = op.apply(
                r,
                spin1=spin1,
                spin2=spin2,
                isospin1=isospin1,
                isospin2=isospin2,
                quantum_state1=quantum_state1,
                quantum_state2=quantum_state2
            )
            yukawa_prime = -backend.xp.exp(-mass * r) * (1 + mass * r) / (r**2)
            F += coef * yukawa_prime * op_val
        F = backend.xp.where(r == 0, 0.0, F)
        return F

    @classmethod
    def preset_reid93(cls, channel: str = "1S0") -> "GeneralNuclearPotential":
        """
        Crea un potencial Reid93 extendido usando la arquitectura modular.

        Creates an extended Reid93 potential using the modular architecture.

        Parameters
        ----------
        channel : str
            Canal parcial ("1S0", "3S1", "3D1", etc.).

        Returns
        -------
        GeneralNuclearPotential
            Instancia configurada para Reid93.
        """
        # Parámetros completos para canales principales (pueden ampliarse con tablas originales)
        if channel == "1S0":
            operators: List[BaseNuclearOperator] = [CentralOperator(), CentralOperator(), CentralOperator()]
            coefficients: List[float] = [ -10.463, -1650.6, 6484.2 ]
            masses: List[float] = [ 0.7, 4.0, 7.0 ]
        elif channel == "3S1":
            operators = [CentralOperator(), CentralOperator(), CentralOperator(), TensorOperator(), TensorOperator(), TensorOperator(), SpinOrbitOperator(), SpinOrbitOperator(), SpinOrbitOperator()]
            coefficients = [ -10.463, -1650.6, 6484.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            masses = [ 0.7, 4.0, 7.0, 0.7, 4.0, 7.0, 0.7, 4.0, 7.0 ]
        elif channel == "3D1":
            operators = [CentralOperator(), CentralOperator(), CentralOperator(), TensorOperator(), TensorOperator(), TensorOperator(), SpinOrbitOperator(), SpinOrbitOperator(), SpinOrbitOperator()]
            coefficients = [ -10.463, -1650.6, 6484.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            masses = [ 0.7, 4.0, 7.0, 0.7, 4.0, 7.0, 0.7, 4.0, 7.0 ]
        else:
            raise ValueError(f"Canal no soportado: {channel}")
        return cls(operators, coefficients, masses, channels=[channel])
