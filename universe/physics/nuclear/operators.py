"""
Operadores nucleares escalables para potenciales realistas.

Nuclear operators for scalable and realistic potentials.

Incluye operadores central, tensorial y spin-órbita, preparados para simulaciones vectorizadas y uso de GPU.
"""
from typing import Protocol
from universe.numerics import backend as backend
import numpy as np

class BaseNuclearOperator(Protocol):
    """
    Interfaz base para operadores nucleares efectivos.

    Base interface for effective nuclear operators.

    Methods
    -------
    apply(r, spin1, spin2, isospin1, isospin2, quantum_state1, quantum_state2) -> backend.xp.ndarray
        Aplica el operador sobre los grados de libertad y retorna el valor escalar o tensorial.
    """
    def apply(
        self,
        r: float | backend.xp.ndarray,
        spin1=None,
        spin2=None,
        isospin1=None,
        isospin2=None,
        quantum_state1=None,
        quantum_state2=None
    ) -> backend.xp.ndarray:
        ...

class CentralOperator:
    """
    Operador central: identidad sobre grados de libertad.

    Central operator: identity on degrees of freedom.
    """
    def apply(
        self,
        r: float | backend.xp.ndarray,
        **kwargs
    ) -> backend.xp.ndarray:
        r = backend.xp.asarray(r)
        return backend.xp.ones_like(r, dtype=backend.xp.float64)

class TensorOperator:
    """
    Operador tensorial S12 para interacción nucleón-nucleón.

    Tensor operator S12 for nucleon-nucleon interaction.

    S12 = 3 (sigma1·r̂)(sigma2·r̂) - sigma1·sigma2
    """
    def apply(
        self,
        r: float | backend.xp.ndarray,
        spin1=None,
        spin2=None,
        **kwargs
    ) -> backend.xp.ndarray:
        """
        Parameters
        ----------
        r : float or backend.xp.ndarray
            Vector de distancia entre nucleones (fm).
        spin1, spin2 : objetos Spin
            Deben tener atributo 'vector' (array 3D).

        Returns
        -------
        backend.xp.ndarray
            Valor escalar del operador tensorial S12.
        """
        r = backend.xp.asarray(r)
        if spin1 is None or spin2 is None or spin1.vector is None or spin2.vector is None:
            return backend.xp.zeros_like(r, dtype=backend.xp.float64)
        # Normalizar r̂
        r_vec = r if r.ndim == 1 else r[0]  # Soporta batch o single
        r_norm = backend.xp.linalg.norm(r_vec)
        if r_norm == 0:
            return backend.xp.zeros_like(r, dtype=backend.xp.float64)
        r_hat = r_vec / r_norm
        # Producto escalar sigma·r̂
        s1_dot_r = backend.xp.dot(spin1.vector, r_hat)
        s2_dot_r = backend.xp.dot(spin2.vector, r_hat)
        s1_dot_s2 = backend.xp.dot(spin1.vector, spin2.vector)
        S12 = 3.0 * s1_dot_r * s2_dot_r - s1_dot_s2
        return backend.xp.asarray(S12)

class SpinOrbitOperator:
    """
    Operador de spin-órbita L·S para interacción nucleón-nucleón.

    Spin-orbit operator L·S for nucleon-nucleon interaction.
    """
    def apply(
        self,
        r: float | backend.xp.ndarray,
        quantum_state1=None,
        quantum_state2=None,
        **kwargs
    ) -> backend.xp.ndarray:
        """
        Parameters
        ----------
        r : float or backend.xp.ndarray
            Vector de distancia entre nucleones (fm).
        quantum_state1, quantum_state2 : objetos QuantumState
            Deben tener atributos l (orbital), s (spin), j (total).

        Returns
        -------
        backend.xp.ndarray
            Valor escalar del operador L·S.
        """
        # Para sistemas clásicos, L·S = 0. Para estados cuánticos, usar números cuánticos.
        if quantum_state1 is None or quantum_state2 is None:
            return backend.xp.zeros_like(backend.xp.asarray(r), dtype=backend.xp.float64)
        l = quantum_state1.l
        s = quantum_state1.s
        j = quantum_state1.j
        # L·S = 0.5 [j(j+1) - l(l+1) - s(s+1)]
        LS = 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))
        return backend.xp.full_like(backend.xp.asarray(r), LS, dtype=backend.xp.float64)
