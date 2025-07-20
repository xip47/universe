import numpy as np
from universe.physics.nuclear.potentials import YukawaPotential
from universe.physics.nuclear.decays import beta_decay
from universe.physics.nuclear.utils import PROTON_MASS_MEV, NEUTRON_MASS_MEV, ELECTRON_MASS_MEV
from universe.physics.nuclear.potentials import Reid93Potential
from universe.numerics.backend import xp

def test_yukawa_potential_values():
    """
    Testea el potencial de Yukawa para valores físicos realistas.
    """
    yukawa = YukawaPotential(g=1.0, mu=1.43e15)
    r = xp.linspace(1e-15, 5e-15, 5)  # 1-5 fm
    V = yukawa.potential(r)
    F = yukawa.force(r)
    assert xp.all(xp.isfinite(V))
    assert xp.all(xp.isfinite(F))
    assert V[0] < 0  # Atracción a corta distancia
    assert F[0] < 0  # Fuerza atractiva

def test_beta_decay_interface():
    """
    Testea la interfaz y valores del decaimiento beta.
    """
    result = beta_decay(NEUTRON_MASS_MEV, PROTON_MASS_MEV, ELECTRON_MASS_MEV)
    assert 'products' in result
    assert 'Q_value_MeV' in result
    assert 'lifetime_s' in result
    assert result['Q_value_MeV'] > 0
    assert result['lifetime_s'] > 0

def test_reid93_potential_values():
    """
    Testea el potencial Reid93 para valores físicos realistas y canal 1S0.
    """
    import numpy as np
    import universe.physics.nuclear.potentials as npot
    xp = np
    reid = Reid93Potential(channel="1S0")
    r = xp.linspace(0.5, 5.0, 10)  # 0.5-5 fm
    V = reid.potential(r)
    F = reid.force(r)
    assert xp.all(xp.isfinite(V)), "Potencial no finito"
    assert xp.all(xp.isfinite(F)), "Fuerza no finita"
    assert V[0] < 0, "El potencial debe ser atractivo a corta distancia"
    assert F[0] < 0, "La fuerza debe ser atractiva a corta distancia"
    assert V.shape == r.shape
    assert F.shape == r.shape

def test_energy_conservation_euler():
    """
    Testea la conservación de la energía total usando integración de Euler para dos nucleones bajo Reid93.
    """
    mass_n = 938.92  # MeV/c^2
    dt = 1e-22  # s
    steps = 200
    reid = Reid93Potential(channel="1S0")
    pos1 = 0.0
    pos2 = 5.0
    vel1 = 0.0
    vel2 = 0.0
    E_hist = []
    for _ in range(steps):
        r12 = abs(pos2 - pos1)
        f = float(reid.force(r12))
        acc1 = f / mass_n
        acc2 = -f / mass_n
        vel1 += acc1 * dt
        vel2 += acc2 * dt
        pos1 += vel1 * dt
        pos2 += vel2 * dt
        v1 = vel1
        v2 = vel2
        KE = 0.5 * mass_n * (v1**2 + v2**2)
        PE = float(reid.potential(r12))
        E_hist.append(KE + PE)
    E_hist = xp.array(E_hist)
    delta_E = xp.abs(E_hist - E_hist[0])
    assert xp.max(delta_E) < 1e-2, f"No se conserva la energía: variación máxima {xp.max(delta_E)} MeV"

def test_energy_conservation_verlet():
    """
    Testea la conservación de la energía total usando integración de Verlet para dos nucleones bajo Reid93.
    """
    mass_n = 938.92
    dt = 1e-22
    steps = 200
    reid = Reid93Potential(channel="1S0")
    pos1 = 0.0
    pos2 = 5.0
    vel1 = 0.0
    vel2 = 0.0
    acc1 = float(reid.force(abs(pos2-pos1))) / mass_n
    acc2 = -acc1
    E_hist = []
    for _ in range(steps):
        pos1_new = pos1 + vel1 * dt + 0.5 * acc1 * dt**2
        pos2_new = pos2 + vel2 * dt + 0.5 * acc2 * dt**2
        r12_new = abs(pos2_new - pos1_new)
        f_new = float(reid.force(r12_new))
        acc1_new = f_new / mass_n
        acc2_new = -acc1_new
        vel1 += 0.5 * (acc1 + acc1_new) * dt
        vel2 += 0.5 * (acc2 + acc2_new) * dt
        pos1, pos2 = pos1_new, pos2_new
        acc1, acc2 = acc1_new, acc2_new
        KE = 0.5 * mass_n * (vel1**2 + vel2**2)
        PE = float(reid.potential(r12_new))
        E_hist.append(KE + PE)
    E_hist = xp.array(E_hist)
    delta_E = xp.abs(E_hist - E_hist[0])
    assert xp.max(delta_E) < 1e-3, f"No se conserva la energía con Verlet: variación máxima {xp.max(delta_E)} MeV"

def test_robustness_dt_range():
    """
    Testea la robustez numérica del integrador de Euler para dt muy grande y muy pequeño.
    """
    mass_n = 938.92
    reid = Reid93Potential(channel="1S0")
    for dt in [1e-24, 1e-22, 1e-20]:
        pos1 = 0.0
        pos2 = 5.0
        vel1 = 0.0
        vel2 = 0.0
        for _ in range(100):
            r12 = abs(pos2 - pos1)
            f = float(reid.force(r12))
            acc1 = f / mass_n
            acc2 = -f / mass_n
            vel1 += acc1 * dt
            vel2 += acc2 * dt
            pos1 += vel1 * dt
            pos2 += vel2 * dt
        assert xp.isfinite(pos1) and xp.isfinite(pos2), f"Posición no finita para dt={dt}"
        assert xp.abs(pos1) < 1e6 and xp.abs(pos2) < 1e6, f"Posición divergente para dt={dt}"

def test_reid93_tensor_channel():
    """
    Testea que el canal tensorial de Reid93 esté implementado y no cause errores.
    """
    reid = Reid93Potential(channel="3S1")
    r = xp.linspace(0.5, 5.0, 10)
    V = reid.potential(r)
    F = reid.force(r)
    assert xp.all(xp.isfinite(V)), "Potencial tensorial no finito"
    assert xp.all(xp.isfinite(F)), "Fuerza tensorial no finita"
    assert V.shape == r.shape
    assert F.shape == r.shape

def test_energy_conservation_2d():
    """
    Testea la conservación de la energía total en 2D para dos nucleones bajo Reid93 usando integración de Verlet y dt pequeño.
    Unidades: posición en fm, masa en MeV/c², dt en s.
    """
    mass_n = 938.92
    dt = 1e-24  # s, más pequeño para estabilidad
    steps = 200
    from universe.particles.definitions import DynamicParticle, ParticleSystem
    reid = Reid93Potential(channel="1S0")
    # Posiciones iniciales en fm
    p1 = DynamicParticle(static=None, position=xp.array([0.0, 0.0]), velocity=xp.array([0.0, 0.0]))
    p2 = DynamicParticle(static=None, position=xp.array([5.0, 0.0]), velocity=xp.array([0.0, 0.0]))
    p1.static = type('Dummy', (), {'mass_mev': mass_n})()
    p2.static = type('Dummy', (), {'mass_mev': mass_n})()
    system = ParticleSystem([p1, p2])
    def force_fn(sys):
        r12 = sys.particles[1].position - sys.particles[0].position
        d = float(xp.linalg.norm(r12))
        f = float(reid.force(d))
        fvec = f * r12 / d if d > 0 else xp.zeros_like(r12)
        return xp.stack([-fvec, fvec])
    # Inicialización para Verlet
    forces = force_fn(system)
    for p, f in zip(system.particles, forces):
        a = f / mass_n
        p.position -= p.velocity * dt - 0.5 * a * dt**2  # Paso atrás para Verlet
    E_hist = []
    for _ in range(steps):
        # Verlet
        forces = force_fn(system)
        for i, p in enumerate(system.particles):
            a = forces[i] / mass_n
            new_pos = 2 * p.position - (p.position - p.velocity * dt + 0.5 * a * dt**2) + a * dt**2
            p.velocity = (new_pos - p.position) / dt
            p.position = new_pos
        v1 = float(xp.linalg.norm(system.particles[0].velocity))
        v2 = float(xp.linalg.norm(system.particles[1].velocity))
        KE = 0.5 * mass_n * (v1**2 + v2**2)
        d = float(xp.linalg.norm(system.particles[1].position - system.particles[0].position))
        PE = float(reid.potential(d))
        E_hist.append(KE + PE)
    E_hist = xp.array(E_hist)
    delta_E = xp.abs(E_hist - E_hist[0])
    assert float(xp.max(delta_E)) < 1e-2, f"No se conserva la energía en 2D (Verlet): variación máxima {float(xp.max(delta_E))} MeV"

def test_multi_particle_interaction():
    """
    Testea la robustez y conservación de energía en un sistema de 3 nucleones en 2D bajo Reid93 usando Verlet y dt pequeño.
    Unidades: posición en fm, masa en MeV/c², dt en s.
    """
    mass_n = 938.92
    dt = 1e-24
    steps = 100
    from universe.particles.definitions import DynamicParticle, ParticleSystem
    reid = Reid93Potential(channel="1S0")
    ps = [DynamicParticle(static=None, position=xp.array([x, 0.0]), velocity=xp.array([0.0, 0.0])) for x in [0.0, 3.0, 6.0]]
    for p in ps:
        p.static = type('Dummy', (), {'mass_mev': mass_n})()
    system = ParticleSystem(ps)
    def force_fn(sys):
        n = len(sys.particles)
        forces = [xp.zeros(2) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r_ij = sys.particles[j].position - sys.particles[i].position
                d = float(xp.linalg.norm(r_ij))
                if d == 0:
                    continue
                f = float(reid.force(d))
                fvec = f * r_ij / d
                forces[i] += fvec
        return xp.stack(forces)
    # Inicialización para Verlet
    forces = force_fn(system)
    for p, f in zip(system.particles, forces):
        a = f / mass_n
        p.position -= p.velocity * dt - 0.5 * a * dt**2
    E_hist = []
    for _ in range(steps):
        forces = force_fn(system)
        for i, p in enumerate(system.particles):
            a = forces[i] / mass_n
            new_pos = 2 * p.position - (p.position - p.velocity * dt + 0.5 * a * dt**2) + a * dt**2
            p.velocity = (new_pos - p.position) / dt
            p.position = new_pos
        KE = sum(0.5 * mass_n * float(xp.linalg.norm(p.velocity))**2 for p in system.particles)
        PE = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                d = float(xp.linalg.norm(system.particles[j].position - system.particles[i].position))
                PE += float(reid.potential(d))
        E_hist.append(KE + PE)
    E_hist = xp.array(E_hist)
    delta_E = xp.abs(E_hist - E_hist[0])
    assert float(xp.max(delta_E)) < 1e-2, f"No se conserva la energía en 3 partículas (Verlet): variación máxima {float(xp.max(delta_E))} MeV"

def test_symmetry_translation_rotation():
    """
    Testea la simetría bajo traslación y rotación: la energía potencial debe ser invariante.
    """
    from universe.particles.definitions import DynamicParticle, ParticleSystem
    reid = Reid93Potential(channel="1S0")
    mass_n = 938.92
    p1 = DynamicParticle(static=None, position=xp.array([0.0, 0.0]), velocity=xp.array([0.0, 0.0]))
    p2 = DynamicParticle(static=None, position=xp.array([5.0, 0.0]), velocity=xp.array([0.0, 0.0]))
    p1.static = type('Dummy', (), {'mass_mev': mass_n})()
    p2.static = type('Dummy', (), {'mass_mev': mass_n})()
    d = float(xp.linalg.norm(p2.position - p1.position))
    PE0 = float(reid.potential(d))
    # Traslación
    shift = xp.array([10.0, -7.0])
    p1.position += shift
    p2.position += shift
    d_shift = float(xp.linalg.norm(p2.position - p1.position))
    PE_shift = float(reid.potential(d_shift))
    # Rotación
    theta = xp.pi / 3
    R = xp.array([[xp.cos(theta), -xp.sin(theta)], [xp.sin(theta), xp.cos(theta)]] )
    p1.position = R @ p1.position
    p2.position = R @ p2.position
    d_rot = float(xp.linalg.norm(p2.position - p1.position))
    PE_rot = float(reid.potential(d_rot))
    assert xp.isclose(PE0, PE_shift), "La energía potencial no es invariante bajo traslación."
    assert xp.isclose(PE0, PE_rot), "La energía potencial no es invariante bajo rotación."
