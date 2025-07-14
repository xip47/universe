"""
Operadores vectoriales para mallas regulares multidimensionales.
Incluye divergencia, rotacional y gradiente.
"""

from typing import Optional
from universe.numerics.backend import xp


def divergence(F: xp.ndarray, dx: float = 1.0, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Calcula la divergencia vectorial de un campo F en malla regular multidimensional.
    Usa diferencias finitas centrales. Soporta condiciones de frontera Dirichlet o Neumann.

    Parameters
    ----------
    F : xp.ndarray
        Campo vectorial, shape (N, dim) o (Nx, Ny, dim) o (Nx, Ny, Nz, dim)
    dx : float
        Paso espacial.
    boundary : {None, 'dirichlet', 'neumann'}
        Tipo de condición de frontera (None: sin tratamiento especial).

    Returns
    -------
    xp.ndarray
        Divergencia escalar en cada punto de la malla.
    """
    dim = F.shape[-1]
    shape = F.shape[:-1]
    div = xp.zeros(shape, dtype=F.dtype)
    for d in range(dim):
        f = F[..., d]
        grad = xp.zeros_like(f)
        grad[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
        if boundary == 'dirichlet':
            grad[0] = (f[1] - f[0]) / dx
            grad[-1] = (f[-1] - f[-2]) / dx
        elif boundary == 'neumann':
            grad[0] = grad[1]
            grad[-1] = grad[-2]
        else:
            grad[0] = (f[1] - f[0]) / dx
            grad[-1] = (f[-1] - f[-2]) / dx
        div += grad
    return div


def curl(F: xp.ndarray, dx: float = 1.0, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Calcula el rotacional vectorial de un campo F en 3D usando diferencias finitas centrales.
    Soporta condiciones de frontera Dirichlet o Neumann.

    Parameters
    ----------
    F : xp.ndarray
        Campo vectorial, shape (N, 3)
    dx : float
        Paso espacial.
    boundary : {None, 'dirichlet', 'neumann'}
        Tipo de condición de frontera.

    Returns
    -------
    xp.ndarray
        Rotacional, shape (N, 3)
    """
    assert F.shape[-1] == 3, "El rotacional solo está definido para campos 3D."
    N = F.shape[0]
    curlF = xp.zeros_like(F)
    fx, fy, fz = F[:, 0], F[:, 1], F[:, 2]
    dFz_dy = xp.zeros(N)
    dFy_dz = xp.zeros(N)
    dFx_dz = xp.zeros(N)
    dFz_dx = xp.zeros(N)
    dFy_dx = xp.zeros(N)
    dFx_dy = xp.zeros(N)
    dFz_dy[1:-1] = (fz[2:] - fz[:-2]) / (2 * dx)
    dFy_dz[1:-1] = (fy[2:] - fy[:-2]) / (2 * dx)
    dFx_dz[1:-1] = (fx[2:] - fx[:-2]) / (2 * dx)
    dFz_dx[1:-1] = (fz[2:] - fz[:-2]) / (2 * dx)
    dFy_dx[1:-1] = (fy[2:] - fy[:-2]) / (2 * dx)
    dFx_dy[1:-1] = (fx[2:] - fx[:-2]) / (2 * dx)
    for arr in [dFz_dy, dFy_dz, dFx_dz, dFz_dx, dFy_dx, dFx_dy]:
        arr[0] = arr[1]
        arr[-1] = arr[-2]
    curlF[:, 0] = dFz_dy - dFy_dz
    curlF[:, 1] = dFx_dz - dFz_dx
    curlF[:, 2] = dFy_dx - dFx_dy
    return curlF


def grad(f: xp.ndarray, dx: float = 1.0, boundary: Optional[str] = None) -> xp.ndarray:
    """
    Placeholder para el gradiente escalar.
    """
    raise NotImplementedError("Gradiente escalar aún no implementado.")
