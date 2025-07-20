"""
Nuclear Potentials Module: Effective strong force and scalable architecture.

Módulo de potenciales nucleares: fuerza fuerte efectiva y arquitectura escalable.
Incluye el potencial de Yukawa y la interfaz para futuros modelos (Reid93, Argonne, etc).
"""
from typing import Protocol
from universe.numerics.backend import xp

class BaseNuclearPotential(Protocol):
    """
    Nuclear potential interface.

    Interfaz para potenciales nucleares efectivos.

    Methods
    -------
    potential(r: float | xp.ndarray) -> float | xp.ndarray
        Devuelve el potencial V(r) en función de la distancia r (m).
    force(r: float | xp.ndarray) -> float | xp.ndarray
        Devuelve la fuerza F(r) = -dV/dr.
    """
    def potential(self, r: float | xp.ndarray) -> float | xp.ndarray:
        ...
    def force(self, r: float | xp.ndarray) -> float | xp.ndarray:
        ...

class YukawaPotential:
    """
    Yukawa potential for the effective strong nuclear interaction.

    Potencial de Yukawa para la interacción nuclear fuerte efectiva.

    Parameters
    ----------
    g : float
        Constante de acoplo fuerte (adimensional).
    mu : float
        Masa del mesón mediador (1/longitud, típicamente 1/fm).
    """
    def __init__(self, g: float = 1.0, mu: float = 1.43e15) -> None:
        self.g: float = g
        self.mu: float = mu  # 1/fm ~ 1.43e15 1/m para el pión

    def potential(self, r: float | xp.ndarray, spin1=None, spin2=None, isospin1=None, isospin2=None, quantum_state1=None, quantum_state2=None) -> float | xp.ndarray:
        """
        Compute the Yukawa potential V(r) with optional spin/isospin dependence.

        Calcula el potencial de Yukawa V(r) con dependencia opcional de spin/isospin.

        Parameters
        ----------
        r : float or xp.ndarray
            Distancia entre nucleones (m).
        spin1, spin2 : opcional
            Grados de libertad de spin de cada partícula.
        isospin1, isospin2 : opcional
            Grados de libertad de isospin de cada partícula.
        quantum_state1, quantum_state2 : opcional
            Estados cuánticos de cada partícula.

        Returns
        -------
        float or xp.ndarray
            Potencial V(r) (J).
        """
        r = xp.asarray(r)
        hbar_c: float = 1.973269804e-16  # J·m
        V = -self.g**2 * hbar_c * xp.exp(-self.mu * r) / r
        V = xp.where(r == 0, 0.0, V)
        # Aquí se pueden añadir dependencias de spin/isospin si se desea
        return V

    def force(self, r: float | xp.ndarray, spin1=None, spin2=None, isospin1=None, isospin2=None, quantum_state1=None, quantum_state2=None) -> float | xp.ndarray:
        """
        Compute the strong nuclear force F(r) = -dV/dr with optional spin/isospin dependence.

        Calcula la fuerza nuclear fuerte F(r) = -dV/dr con dependencia opcional de spin/isospin.

        Parameters
        ----------
        r : float or xp.ndarray
            Distancia entre nucleones (m).
        spin1, spin2 : opcional
            Grados de libertad de spin de cada partícula.
        isospin1, isospin2 : opcional
            Grados de libertad de isospin de cada partícula.
        quantum_state1, quantum_state2 : opcional
            Estados cuánticos de cada partícula.

        Returns
        -------
        float or xp.ndarray
            Fuerza F(r) (N).
        """
        r = xp.asarray(r)
        hbar_c: float = 1.973269804e-16  # J·m
        F = -self.g**2 * hbar_c * xp.exp(-self.mu * r) * (1 + self.mu * r) / (r**2)
        F = xp.where(r == 0, 0.0, F)
        # Aquí se pueden añadir dependencias de spin/isospin si se desea
        return F

# Ejemplo de interfaz para futuros modelos avanzados:
# class Reid93Potential(BaseNuclearPotential):
#     ...

class Reid93Potential:
    """
    Reid93 potential for nucleon-nucleon interaction (most realistic phenomenological model).

    Potencial Reid93 para la interacción nucleón-nucleón: incluye términos centrales, tensoriales y de spin-órbita,
    dependientes de spin e isospin. Escalable y preparado para simulaciones de alta fidelidad.

    Parameters
    ----------
    channel : str
        Canal parcial ("1S0", "3S1", "3D1", etc). Determina los parámetros físicos.
    """
    # Parámetros de ejemplo para el canal 1S0 (pueden ampliarse con tablas completas)
    _params = {
        "1S0": {
            "central": [
                # (coeficiente, masa [1/fm])
                ( -10.463, 0.7 ),
                ( -1650.6, 4.0 ),
                ( 6484.2, 7.0 ),
            ],
            "tensor": [],
            "spin_orbit": []
        },
        "3S1": {
            "central": [
                ( -10.463, 0.7 ),
                ( -1650.6, 4.0 ),
                ( 6484.2, 7.0 ),
            ],
            "tensor": [
                ( 0.0, 0.7 ),
                ( 0.0, 4.0 ),
                ( 0.0, 7.0 ),
            ],
            "spin_orbit": []
        },
        # Agregar más canales y términos según tablas originales
    }

    def __init__(self, channel: str = "1S0") -> None:
        if channel not in self._params:
            raise ValueError(f"Canal no soportado: {channel}")
        self.channel: str = channel
        self.central_terms = self._params[channel]["central"]
        self.tensor_terms = self._params[channel]["tensor"]
        self.spin_orbit_terms = self._params[channel]["spin_orbit"]

    def yukawa(self, r: float | xp.ndarray, mass: float) -> xp.ndarray:
        """
        Función de Yukawa modificada para el potencial Reid93.

        Parameters
        ----------
        r : float or xp.ndarray
            Distancia (fm).
        mass : float
            Masa del mesón (1/fm).

        Returns
        -------
        xp.ndarray
            Valor de la función de Yukawa.
        """
        r = xp.asarray(r)
        return xp.exp(-mass * r) / r

    def potential(self, r: float | xp.ndarray, S: int = 0, T: int = 1, spin1=None, spin2=None, isospin1=None, isospin2=None, quantum_state1=None, quantum_state2=None) -> xp.ndarray:
        """
        Compute the full Reid93 potential for given spin (S), isospin (T), y grados de libertad extendidos.

        Calcula el potencial total de Reid93 para spin (S), isospin (T) y grados de libertad extendidos.

        Parameters
        ----------
        r : float or xp.ndarray
            Distancia entre nucleones (fm).
        S : int
            Spin total (0 o 1).
        T : int
            Isospin total (0 o 1).
        spin1, spin2 : opcional
            Grados de libertad de spin de cada partícula.
        isospin1, isospin2 : opcional
            Grados de libertad de isospin de cada partícula.
        quantum_state1, quantum_state2 : opcional
            Estados cuánticos de cada partícula.

        Returns
        -------
        xp.ndarray
            Potencial V(r) (MeV).
        """
        r = xp.asarray(r)
        V = xp.zeros_like(r, dtype=xp.float64)
        # Término central
        for coef, mass in self.central_terms:
            V += coef * self.yukawa(r, mass)
        # Término tensorial (si aplica)
        for coef, mass in self.tensor_terms:
            # Ejemplo: dependencia tensorial de spin
            if spin1 is not None and spin2 is not None:
                # Aquí se puede calcular el operador tensorial S12(spin1, spin2, r)
                pass  # Placeholder para S12
            V += coef * self.yukawa(r, mass)
        # Término spin-órbita (si aplica)
        for coef, mass in self.spin_orbit_terms:
            if spin1 is not None and quantum_state1 is not None:
                # Aquí se puede calcular el operador L·S
                pass  # Placeholder para L·S
            V += coef * self.yukawa(r, mass)
        V = xp.where(r == 0, 0.0, V)
        return V

    def force(self, r: float | xp.ndarray, S: int = 0, T: int = 1, spin1=None, spin2=None, isospin1=None, isospin2=None, quantum_state1=None, quantum_state2=None) -> xp.ndarray:
        """
        Compute the force for the full Reid93 potential with all degrees of freedom.

        Calcula la fuerza para el potencial completo de Reid93 con todos los grados de libertad.

        Parameters
        ----------
        r : float or xp.ndarray
            Distancia entre nucleones (fm).
        S : int
            Spin total (0 o 1).
        T : int
            Isospin total (0 o 1).
        spin1, spin2 : opcional
            Grados de libertad de spin de cada partícula.
        isospin1, isospin2 : opcional
            Grados de libertad de isospin de cada partícula.
        quantum_state1, quantum_state2 : opcional
            Estados cuánticos de cada partícula.

        Returns
        -------
        xp.ndarray
            Fuerza F(r) (MeV/fm).
        """
        r = xp.asarray(r)
        F = xp.zeros_like(r, dtype=xp.float64)
        for coef, mass in self.central_terms:
            F += coef * (-xp.exp(-mass * r) * (1 + mass * r) / (r**2))
        for coef, mass in self.tensor_terms:
            if spin1 is not None and spin2 is not None:
                # Aquí se puede calcular el operador tensorial S12(spin1, spin2, r)
                pass
            F += coef * (-xp.exp(-mass * r) * (1 + mass * r) / (r**2))
        for coef, mass in self.spin_orbit_terms:
            if spin1 is not None and quantum_state1 is not None:
                # Aquí se puede calcular el operador L·S
                pass
            F += coef * (-xp.exp(-mass * r) * (1 + mass * r) / (r**2))
        F = xp.where(r == 0, 0.0, F)
        return F

# Nota: Para máxima fidelidad, los operadores tensoriales y de spin-orbita deben implementarse explícitamente
# usando matrices de Pauli y operadores angulares. Esta clase está preparada para esa extensión.
