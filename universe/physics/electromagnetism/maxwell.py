"""
Módulo para las ecuaciones de Maxwell en forma diferencial e integral.
Implementación numérica avanzada y realista para sistemas discretos multidimensionales.
"""

from typing import Optional
from universe.numerics.backend import xp
from universe.numerics.vector_ops import divergence
from universe.numerics.vector_ops import curl
from universe.physics.constants import EPSILON_0
from universe.physics.constants import MU_0


def gauss_electric_field(E: xp.ndarray, rho: Optional[xp.ndarray] = None, epsilon: Optional[xp.ndarray] = None, dx: float = 1.0, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Compute the Gauss law for the electric field (differential form).

    Calcula la ecuación de Gauss para el campo eléctrico en forma diferencial: div(E) = rho / epsilon.
    Permite permitividad variable y condiciones de frontera.

    Parameters
    ----------
    E : xp.ndarray
        Campo eléctrico, con forma (..., 3) para 3D o (..., 2) para 2D. [V/m]
    rho : Optional[xp.ndarray]
        Densidad de carga eléctrica. Si no se proporciona, solo se calcula la divergencia. [C/m^3]
    epsilon : Optional[xp.ndarray]
        Permitividad eléctrica local. Si es None, se usa EPSILON_0. [F/m]
    dx : float
        Espaciado espacial uniforme de la malla. [m]
    boundary : Optional[str]
        Tipo de condición de frontera ('dirichlet', 'neumann', etc.).

    Returns
    -------
    xp.ndarray
        Resultado de div(E) - rho/epsilon. Si rho es None, retorna solo div(E).

    Warnings
    --------
    La fidelidad depende de la precisión del operador de divergencia y del mallado. Para permitividad variable, epsilon debe tener la misma forma que E.
    """
    divE = divergence(E, dx=dx, boundary=boundary)
    if epsilon is None:
        epsilon = EPSILON_0
    if rho is not None:
        return divE - rho / epsilon
    return divE


def gauss_magnetic_field(B: xp.ndarray, dx: float = 1.0, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Compute the Gauss law for the magnetic field (differential form).

    Calcula la ecuación de Gauss para el campo magnético en forma diferencial: div(B) = 0.
    Permite condiciones de frontera.

    Parameters
    ----------
    B : xp.ndarray
        Campo magnético, con forma (..., 3) para 3D o (..., 2) para 2D. [T]
    dx : float
        Espaciado espacial uniforme de la malla. [m]
    boundary : Optional[str]
        Tipo de condición de frontera ('dirichlet', 'neumann', etc.).

    Returns
    -------
    xp.ndarray
        Divergencia de B. Debe ser cercano a cero en simulaciones físicas realistas.

    Warnings
    --------
    La precisión depende del operador de divergencia y la resolución de la malla.
    """
    return divergence(B, dx=dx, boundary=boundary)


def faraday_law(E: xp.ndarray, B: xp.ndarray, dt: float = 1.0, dx: float = 1.0, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Compute the Faraday law of induction (differential form).

    Calcula la ley de Faraday de la inducción en forma diferencial: rot(E) = -dB/dt.
    Permite condiciones de frontera.

    Parameters
    ----------
    E : xp.ndarray
        Campo eléctrico, (..., 3) o (..., 2). [V/m]
    B : xp.ndarray
        Campo magnético, (..., 3) o (..., 2). [T]
    dt : float
        Paso temporal para la derivada temporal. [s]
    dx : float
        Espaciado espacial uniforme de la malla. [m]
    boundary : Optional[str]
        Tipo de condición de frontera ('dirichlet', 'neumann', etc.).

    Returns
    -------
    xp.ndarray
        rot(E) + dB/dt, que debe ser cercano a cero si se cumple la ley de Faraday.

    Warnings
    --------
    La precisión depende de los operadores de rotacional y derivada temporal.
    """
    rotE = curl(E, dx=dx, boundary=boundary)
    dBdt = xp.gradient(B, dt, axis=0)
    return rotE + dBdt


def ampere_maxwell_law(B: xp.ndarray, J: xp.ndarray, E: xp.ndarray, dt: float = 1.0, dx: float = 1.0, mu: Optional[xp.ndarray] = None, epsilon: Optional[xp.ndarray] = None, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Compute the Ampère-Maxwell law (differential form).

    Calcula la ley de Ampère-Maxwell en forma diferencial: rot(B) = mu*J + mu*epsilon*dE/dt.
    Permite permeabilidad y permitividad variables y condiciones de frontera.

    Parameters
    ----------
    B : xp.ndarray
        Campo magnético, (..., 3) o (..., 2). [T]
    J : xp.ndarray
        Densidad de corriente eléctrica, (..., 3) o (..., 2). [A/m^2]
    E : xp.ndarray
        Campo eléctrico, (..., 3) o (..., 2). [V/m]
    dt : float
        Paso temporal para la derivada temporal. [s]
    dx : float
        Espaciado espacial uniforme de la malla. [m]
    mu : Optional[xp.ndarray]
        Permeabilidad magnética local. Si es None, se usa MU_0. [H/m]
    epsilon : Optional[xp.ndarray]
        Permitividad eléctrica local. Si es None, se usa EPSILON_0. [F/m]
    boundary : Optional[str]
        Tipo de condición de frontera ('dirichlet', 'neumann', etc.).

    Returns
    -------
    xp.ndarray
        rot(B) - mu*J - mu*epsilon*dE/dt, que debe ser cercano a cero si se cumple la ley de Ampère-Maxwell.

    Warnings
    --------
    La fidelidad depende de la precisión de los operadores y de la resolución de la malla. Para materiales no homogéneos, mu y epsilon deben ser arrays compatibles.
    """
    if mu is None:
        mu = MU_0
    if epsilon is None:
        epsilon = EPSILON_0
    rotB = curl(B, dx=dx, boundary=boundary)
    dEdt = xp.gradient(E, dt, axis=0)
    return rotB - mu * J - mu * epsilon * dEdt
