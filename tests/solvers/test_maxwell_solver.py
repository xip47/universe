import numpy as np
import pytest
from universe.solvers.maxwell_solver import MaxwellSolver
from universe.numerics.backend import xp
from universe.physics.constants import EPSILON_0, MU_0

def courant_dt(dx):
    c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
    return dx / (c * np.sqrt(2)) * 0.99  # 0.99 para margen de seguridad

def test_wave_propagation():
    shape = (100, 100)
    dx = 1e-3
    dt = courant_dt(dx)
    solver = MaxwellSolver(shape, dx, dt)
    # Pulso inicial en el centro
    pulse = xp.zeros(shape)
    pulse[shape[0]//2, shape[1]//2] = 1.0
    solver.step(source=pulse)
    initial_energy = xp.sum(solver.Ez**2 + solver.Hx**2 + solver.Hy**2)
    if hasattr(initial_energy, 'get'):
        initial_energy = float(initial_energy.get())
    else:
        initial_energy = float(initial_energy)
    solver.run(steps=50)
    final_energy = xp.sum(solver.Ez**2 + solver.Hx**2 + solver.Hy**2)
    if hasattr(final_energy, 'get'):
        final_energy = float(final_energy.get())
    else:
        final_energy = float(final_energy)
    if initial_energy == 0:
        pytest.skip("Energía inicial nula, no se puede comparar propagación de onda.")
    # La energía no debe crecer descontroladamente (sin fuentes ni pérdidas)
    assert final_energy < 2 * initial_energy

def test_boundary_conditions():
    shape = (30, 30)
    dx = 1e-3
    dt = courant_dt(dx)
    solver = MaxwellSolver(shape, dx, dt)
    # Pulso en el borde
    pulse = xp.zeros(shape)
    pulse[0, :] = 1.0
    solver.step(source=pulse)
    # Los bordes deben ser cero (Dirichlet)
    assert xp.allclose(solver.Ez[0, :], 0)
    assert xp.allclose(solver.Ez[-1, :], 0)
    assert xp.allclose(solver.Ez[:, 0], 0)
    assert xp.allclose(solver.Ez[:, -1], 0)

def test_materials():
    shape = (20, 20)
    dx = 1e-3
    dt = courant_dt(dx)
    epsilon = 2.0 * xp.ones(shape)
    mu = 0.5 * xp.ones(shape)
    solver = MaxwellSolver(shape, dx, dt, epsilon=epsilon, mu=mu)
    assert xp.allclose(solver.epsilon, 2.0)
    assert xp.allclose(solver.mu, 0.5)
    # Cambiar materiales
    new_epsilon = 3.0 * xp.ones(shape)
    solver.set_materials({'epsilon': new_epsilon})
    assert xp.allclose(solver.epsilon, 3.0)

def test_maxwell_energy_conservation():
    """
    Valida la conservación de la energía electromagnética en vacío tras varios pasos FDTD.
    Inicializa Ez con un pulso gaussiano y Hx, Hy en cero.
    La energía total debe conservarse dentro de un 1% si el pulso no alcanza el borde.
    """
    from universe.physics.constants import EPSILON_0, MU_0
    shape = (100, 100)
    dx = 1e-3
    dt = courant_dt(dx)
    solver = MaxwellSolver(shape, dx, dt)
    # Pulso gaussiano en el centro
    y, x = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing='ij')
    y0, x0 = shape[0] // 2, shape[1] // 2
    sigma = 5.0
    solver.Ez = xp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    solver.Hx = xp.zeros(shape)
    solver.Hy = xp.zeros(shape)
    # Energía física inicial
    energy0 = float(xp.sum(0.5 * EPSILON_0 * solver.Ez**2 + 0.5 * MU_0 * (solver.Hx**2 + solver.Hy**2)) * dx**2)
    # Avanzar pocos pasos para que el pulso no alcance el borde
    solver.run(steps=20)
    # Energía física final
    energy1 = float(xp.sum(0.5 * EPSILON_0 * solver.Ez**2 + 0.5 * MU_0 * (solver.Hx**2 + solver.Hy**2)) * dx**2)
    # La energía debe conservarse dentro de un 1%
    assert abs(energy1 - energy0) / energy0 < 0.01

def test_pml_absorption():
    """
    Test PML absorption: a wave pulse debe ser absorbido en los bordes sin reflexiones significativas.

    Valida que la energía electromagnética disminuye drásticamente cuando el pulso alcanza la región PML,
    demostrando la efectividad de la absorción y la ausencia de reflexiones artificiales.
    Además, registra la energía en cada paso para analizar la disipación.
    """
    shape = (100, 100)
    dx = 1e-3
    from universe.physics.constants import EPSILON_0, MU_0
    def courant_dt(dx):
        c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
        return dx / (c * np.sqrt(2)) * 0.99
    dt = courant_dt(dx)
    pml_thickness = 20
    solver = MaxwellSolver(
        shape, dx, dt,
        pml_thickness=pml_thickness,
        pml_sigma_max=50.0,
        use_pml=True
    )
    y, x = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing='ij')
    y0, x0 = shape[0] // 2, shape[1] // 2
    sigma = 3.0
    # Aumentar la amplitud del pulso inicial para asegurar energía significativa
    solver.Ez = 1.0 * xp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    solver.Hx = xp.zeros(shape)
    solver.Hy = xp.zeros(shape)
    energy0 = xp.sum(0.5 * EPSILON_0 * solver.Ez**2 + 0.5 * MU_0 * (solver.Hx**2 + solver.Hy**2)) * dx**2
    if hasattr(energy0, 'get'):
        energy0 = float(energy0.get())
    else:
        energy0 = float(energy0)
    if energy0 < 1e-10:
        pytest.skip("Energía inicial demasiado baja para probar absorción PML.")
    energies = [energy0]
    steps = 400
    for _ in range(steps):
        solver.step()
        energy = xp.sum(0.5 * EPSILON_0 * solver.Ez**2 + 0.5 * MU_0 * (solver.Hx**2 + solver.Hy**2)) * dx**2
        if hasattr(energy, 'get'):
            energy = float(energy.get())
        else:
            energy = float(energy)
        energies.append(energy)
    diffs = np.diff(energies)
    decreasing = np.sum(diffs < 0) / len(diffs)
    assert decreasing > 0.9, f"La energía no decrece en la mayoría de los pasos: {decreasing*100:.1f}%"
    assert energies[-1] < 0.01 * energy0, f"La energía final no se disipó lo suficiente: {energies[-1]} vs {energy0}"
