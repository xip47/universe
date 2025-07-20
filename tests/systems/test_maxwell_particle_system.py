import numpy as np
import pytest
from universe.systems.maxwell_particle_system import MaxwellParticleSystem
from universe.particles.extended_definitions import DynamicParticleExtended, BaseParticleExtended
from universe.particles.degrees_of_freedom import Charge, Spin, Isospin
from universe.numerics.backend import xp


def create_centered_particle(q: float, m: float, shape: tuple, v: float = 0.0) -> DynamicParticleExtended:
    """
    Crea una partícula centrada en la malla con carga y masa dadas.
    """
    pos = xp.array([s // 2 for s in shape], dtype=xp.float64)
    vel = xp.array([v for _ in shape], dtype=xp.float64)
    spin = Spin(0.5, xp.array([0, 0, 1]))
    isospin = Isospin(0.5, xp.array([0, 0, 1]))
    base = BaseParticleExtended("test", m, Charge(q), spin, isospin)
    return DynamicParticleExtended(base, pos, vel, spin, isospin)


def courant_dt(dx: float) -> float:
    """
    Calcula el paso temporal estable según la condición de Courant para FDTD 2D.
    """
    from universe.physics.constants import EPSILON_0, MU_0
    c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
    return dx / (c * np.sqrt(2)) * 0.99


def test_cic_charge_conservation():
    """
    Verifica que la suma de la carga depositada en la malla es igual a la suma de las cargas de las partículas (CIC).
    Compara ambas cantidades en Coulombs para consistencia física.
    """
    from universe.physics.constants import E_CHARGE
    shape = (32, 32)
    dx = 1.0
    dt = courant_dt(dx)
    q = 1.0
    m = 1.0
    particles = [create_centered_particle(q, m, shape)]
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    system.deposit_charge_current()
    total_grid_charge = float(xp.sum(system.charge_grid))
    total_particle_charge = sum(getattr(p.static.charge, 'value', 0.0) for p in particles)
    # Comparar en Coulombs
    assert np.isclose(total_grid_charge, total_particle_charge * E_CHARGE, atol=1e-12)


def test_total_energy_conservation():
    """
    Verifica que la energía total (campos + partículas) se conserva en ausencia de fuentes y pérdidas.
    Usa dt estable (Courant) y pulso inicial de amplitud 1.0 para asegurar energía física significativa y robustez numérica.
    """
    shape = (32, 32)
    dx = 1.0
    dt = courant_dt(dx)
    q = 1.0
    m = 1.0
    particles = [create_centered_particle(q, m, shape)]
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    # Inicializar campo Ez con un pulso gaussiano en el centro (amplitud 1.0)
    y, x = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing='ij')
    y0, x0 = shape[0] // 2, shape[1] // 2
    sigma = 3.0
    system.solver.Ez = 1.0 * xp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    energy0 = system.total_energy()
    for _ in range(10):
        system.step()
        energy = system.total_energy()
        if not np.isfinite(energy):
            pytest.fail("Energía numérica inestable (NaN o inf) durante la simulación.")
    energy1 = system.total_energy()
    assert abs(energy1 - energy0) / energy0 < 0.05


def test_cic_interpolation_smoothness():
    """
    Verifica que la interpolación CIC produce campos suaves y continuos para posiciones intermedias.
    """
    shape = (16, 16)
    dx = 1.0
    dt = 0.1
    q = 1.0
    m = 1.0
    # Campo Ez lineal
    from universe.solvers.maxwell_solver import MaxwellSolver
    solver = MaxwellSolver(shape, dx, dt)
    for i in range(shape[0]):
        for j in range(shape[1]):
            solver.Ez[i, j] = i + j
    particles = []
    for frac in np.linspace(0, 1, 10):
        # Usar create_centered_particle pero con posición variable
        pos = xp.array([frac * (shape[0] - 2), frac * (shape[1] - 2)], dtype=xp.float64)
        vel = xp.zeros(2, dtype=xp.float64)
        spin = Spin(0.5, xp.array([0, 0, 1]))
        isospin = Isospin(0.5, xp.array([0, 0, 1]))
        base = BaseParticleExtended("test", m, Charge(q), spin, isospin)
        particles.append(DynamicParticleExtended(base, pos, vel, spin, isospin))
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    system.solver = solver
    fields = system.interpolate_fields()
    values = [float(Ez.get()) if hasattr(Ez, 'get') else float(Ez) for (Ez, _, _) in fields]
    # Debe ser monótono creciente
    assert all(values[i] <= values[i+1] for i in range(len(values)-1))


def test_gpu_compatibility():
    """
    Verifica que el sistema funciona correctamente con el backend GPU (si está disponible).
    """
    shape = (8, 8)
    dx = 1.0
    dt = 0.1
    q = 1.0
    m = 1.0
    particles = [create_centered_particle(q, m, shape)]
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    system.deposit_charge_current()
    assert xp.all(xp.isfinite(system.charge_grid))
    assert xp.all(xp.isfinite(system.current_grid))


def test_boundary_conditions():
    """
    Verifica que las condiciones de frontera (Dirichlet) se aplican correctamente en la malla.
    """
    shape = (16, 16)
    dx = 1.0
    dt = 0.1
    q = 1.0
    m = 1.0
    particles = [create_centered_particle(q, m, shape)]
    system = MaxwellParticleSystem(particles, shape, dx, dt, solver_kwargs={"bc": {"type": "dirichlet"}})
    system.solver.Ez[0, :] = 1.0
    system.solver.Ez[-1, :] = 1.0
    system.solver.apply_boundary_conditions()
    assert xp.allclose(system.solver.Ez[0, :], 0)
    assert xp.allclose(system.solver.Ez[-1, :], 0)


def test_pml_absorption():
    """
    Verifica que la energía de los campos decrece al usar PML en los bordes.
    Usa dt estable (Courant), pulso inicial de amplitud 1.0 y sigma_max=20.0 para robustez física y numérica.
    Si la energía inicial es demasiado baja, salta el test.
    """
    shape = (32, 32)
    dx = 1.0
    dt = courant_dt(dx)
    q = 1.0
    m = 1.0
    particles = [create_centered_particle(q, m, shape)]
    system = MaxwellParticleSystem(
        particles, shape, dx, dt,
        solver_kwargs={"use_pml": True, "pml_thickness": 4, "pml_sigma_max": 20.0}
    )
    # Inicializar campo Ez con un pulso gaussiano en el centro (amplitud 1.0)
    y, x = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing='ij')
    y0, x0 = shape[0] // 2, shape[1] // 2
    sigma = 3.0
    system.solver.Ez = 1.0 * xp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    energy0 = system.total_energy()
    if energy0 < 1e-8:
        pytest.skip("Energía inicial demasiado baja para probar absorción PML.")
    for _ in range(30):
        system.step()
        energy = system.total_energy()
        if not np.isfinite(energy):
            pytest.fail("Energía numérica inestable (NaN o inf) durante la simulación.")
    energy1 = system.total_energy()
    assert energy1 < energy0


def test_out_of_bounds_robustness():
    """
    Verifica que el sistema no falla si una partícula sale de la malla (la carga no se deposita y no hay error).
    """
    shape = (16, 16)
    dx = 1.0
    dt = 0.1
    q = 1.0
    m = 1.0
    particles = [create_centered_particle(q, m, shape)]
    particles[0].position = xp.array([100.0, 100.0], dtype=xp.float64)  # fuera de la malla
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    try:
        system.deposit_charge_current()
    except Exception as e:
        pytest.fail(f"Unexpected error with out-of-bounds particle: {e}")


def test_cic_charge_conservation_multiple_particles():
    """
    Verifica la conservación de la carga total en la malla con múltiples partículas de diferentes cargas y posiciones.
    """
    from universe.physics.constants import E_CHARGE
    shape = (32, 32)
    dx = 1.0
    dt = courant_dt(dx)
    particles = [
        create_centered_particle(1.0, 1.0, shape),
        create_centered_particle(-1.0, 1.0, shape, v=0.2),
        create_centered_particle(2.0, 1.0, shape, v=-0.1)
    ]
    # Desplazar la segunda partícula al borde
    particles[1].position = xp.array([0.0, 0.0], dtype=xp.float64)
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    system.deposit_charge_current()
    total_grid_charge = float(xp.sum(system.charge_grid))
    total_particle_charge = sum(getattr(p.static.charge, 'value', 0.0) for p in particles)
    assert np.isclose(total_grid_charge, total_particle_charge * E_CHARGE, atol=1e-12)


def test_cic_charge_conservation_zero_charge():
    """
    Verifica que partículas con carga cero no afectan la malla de carga ni corriente.
    """
    from universe.physics.constants import E_CHARGE
    shape = (32, 32)
    dx = 1.0
    dt = courant_dt(dx)
    particles = [
        create_centered_particle(0.0, 1.0, shape),
        create_centered_particle(0.0, 1.0, shape, v=0.5)
    ]
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    system.deposit_charge_current()
    total_grid_charge = float(xp.sum(system.charge_grid))
    total_grid_current = float(xp.sum(system.current_grid))
    assert np.isclose(total_grid_charge, 0.0, atol=1e-16)
    assert np.isclose(total_grid_current, 0.0, atol=1e-16)


def test_cic_charge_conservation_high_velocity():
    """
    Verifica que partículas con velocidad alta (pero menor a c) no generan inestabilidad numérica en la malla de corriente.
    """
    from universe.physics.constants import E_CHARGE
    shape = (32, 32)
    dx = 1.0
    dt = courant_dt(dx)
    v = 0.9  # velocidad alta, menor a c=1
    particles = [create_centered_particle(1.0, 1.0, shape, v=v)]
    system = MaxwellParticleSystem(particles, shape, dx, dt)
    system.deposit_charge_current()
    total_grid_charge = float(xp.sum(system.charge_grid))
    total_particle_charge = sum(getattr(p.static.charge, 'value', 0.0) for p in particles)
    assert np.isclose(total_grid_charge, total_particle_charge * E_CHARGE, atol=1e-12)
    # La corriente debe ser finita y no NaN
    assert np.all(np.isfinite(system.current_grid))
