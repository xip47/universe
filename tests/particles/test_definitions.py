import numpy as np
import pytest
from universe.particles.definitions import Quark, Lepton, Boson, DynamicParticle, ParticleSystem
from universe.physics.electromagnetism import coulomb_forces_on_particles
from universe.physics.constants import E_CHARGE

class DummyQuark(Quark):
    def __init__(self):
        super().__init__(
            name="up",
            symbol="u",
            mass_mev=2.2,
            charge=+2/3,
            spin=0.5,
            generation=1,
            color_charge=True
        )

class DummyLepton(Lepton):
    def __init__(self):
        super().__init__(
            name="electron",
            symbol="e-",
            mass_mev=0.511,
            charge=-1.0,
            spin=0.5,
            generation=1,
            lepton_number=1
        )

def test_dynamic_particle_init():
    q = DummyQuark()
    pos = np.zeros(3)
    vel = np.ones(3)
    p = DynamicParticle(static=q, position=pos, velocity=vel)
    assert np.allclose(p.position, pos)
    assert np.allclose(p.velocity, vel)
    assert p.static.name == "up"

    # Error si dimensiones no coinciden
    with pytest.raises(ValueError):
        DynamicParticle(static=q, position=np.zeros(2), velocity=np.ones(3))

    # Error si aceleración de dimensión incorrecta
    with pytest.raises(ValueError):
        DynamicParticle(static=q, position=pos, velocity=vel, acceleration=np.zeros(2))

def test_particle_system_init_and_getters():
    q = DummyQuark()
    l = DummyLepton()
    p1 = DynamicParticle(static=q, position=np.zeros(3), velocity=np.ones(3))
    p2 = DynamicParticle(static=l, position=np.ones(3), velocity=-np.ones(3))
    system = ParticleSystem([p1, p2])
    pos = system.get_positions()
    vel = system.get_velocities()
    assert pos.shape == (2, 3)
    assert vel.shape == (2, 3)
    assert np.allclose(pos[0], np.zeros(3))
    assert np.allclose(pos[1], np.ones(3))
    assert np.allclose(vel[0], np.ones(3))
    assert np.allclose(vel[1], -np.ones(3))

def test_particle_system_step():
    q = DummyQuark()
    p = DynamicParticle(static=q, position=np.zeros(3), velocity=np.zeros(3))
    system = ParticleSystem([p])
    # Fuerza constante en x
    def force_fn(sys):
        return np.array([[1.0, 0.0, 0.0]])
    dt = 0.1
    system.step(dt, force_fn)
    # La aceleración es F/m
    m_kg = q.mass_mev * 1.78266192e-30
    expected_vx = (1.0 / m_kg) * dt
    expected_x = expected_vx * dt
    assert np.isclose(system.particles[0].velocity[0], expected_vx)
    assert np.isclose(system.particles[0].position[0], expected_x)

def test_coulomb_forces_on_particles_simple():
    # Dos electrones separados 1 metro
    class DummyElectron(Lepton):
        def __init__(self):
            super().__init__(
                name="electron",
                symbol="e-",
                mass_mev=0.511,
                charge=-1.0,
                spin=0.5,
                generation=1,
                lepton_number=1
            )
    e1 = DummyElectron()
    e2 = DummyElectron()
    p1 = DynamicParticle(static=e1, position=np.zeros(3), velocity=np.zeros(3))
    p2 = DynamicParticle(static=e2, position=np.array([1.0, 0.0, 0.0]), velocity=np.zeros(3))
    system = ParticleSystem([p1, p2])
    forces = coulomb_forces_on_particles(system)
    # La fuerza debe ser repulsiva y de igual magnitud/opuesta
    f_expected = (1 / (4 * np.pi * 8.854187817e-12)) * (E_CHARGE ** 2) / (1.0 ** 2)
    assert np.isclose(forces[0][0], f_expected)
    assert np.isclose(forces[1][0], -f_expected)
    assert np.allclose(forces[0][1:], 0)
    assert np.allclose(forces[1][1:], 0)

def test_coulomb_forces_on_particles_opposite():
    # Electrón y protón separados 1 metro
    class DummyProton(Quark):
        def __init__(self):
            super().__init__(
                name="proton",
                symbol="p+",
                mass_mev=938.272,
                charge=+1.0,
                spin=0.5,
                generation=1,
                color_charge=False
            )
    class DummyElectron(Lepton):
        def __init__(self):
            super().__init__(
                name="electron",
                symbol="e-",
                mass_mev=0.511,
                charge=-1.0,
                spin=0.5,
                generation=1,
                lepton_number=1
            )
    p = DummyProton()
    e = DummyElectron()
    dp = DynamicParticle(static=p, position=np.zeros(3), velocity=np.zeros(3))
    de = DynamicParticle(static=e, position=np.array([1.0, 0.0, 0.0]), velocity=np.zeros(3))
    system = ParticleSystem([dp, de])
    forces = coulomb_forces_on_particles(system)
    f_expected = (1 / (4 * np.pi * 8.854187817e-12)) * (E_CHARGE ** 2) / (1.0 ** 2)
    # Fuerza atractiva: protón hacia +x, electrón hacia -x
    assert np.isclose(forces[0][0], -f_expected)
    assert np.isclose(forces[1][0], f_expected)
    assert np.allclose(forces[0][1:], 0)
    assert np.allclose(forces[1][1:], 0)
