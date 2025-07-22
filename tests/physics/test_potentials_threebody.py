"""
Tests para potenciales de tres cuerpos (SimpleThreeBodyPotential).
"""
import numpy as np
from universe.particles.extended_definitions import DynamicParticleExtended, BaseParticleExtended
from universe.particles.degrees_of_freedom import Spin, Isospin, Charge, QuantumState
from universe.physics.nuclear.potentials_threebody import SimpleThreeBodyPotential
from universe.numerics.backend import xp

def crear_nucleon(pos):
    spin = Spin(0.5, xp.array([0, 0, 1]))
    isospin = Isospin(0.5, xp.array([1, 0, 0]))
    charge = Charge(1.0)
    qs = QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
    base = BaseParticleExtended("nucleon", 938.27, charge, spin, isospin, qs)
    return DynamicParticleExtended(base, xp.array(pos), xp.array([0.0, 0.0]), spin, isospin, qs)

def test_simple_threebody_potential_value():
    """
    Testea el valor del potencial de tres cuerpos para una configuración simétrica.
    """
    p1 = crear_nucleon([0.0])
    p2 = crear_nucleon([1.0])
    p3 = crear_nucleon([2.0])
    pot = SimpleThreeBodyPotential(strength=0.5)
    V = pot.potential(p1, p2, p3)
    assert np.isfinite(V), "El potencial debe ser finito"
    assert V > 0.0, "El potencial armónico debe ser positivo"

def test_simple_threebody_forces_symmetry():
    """
    Testea que la suma de las fuerzas de tres cuerpos es cero (simetría de traslación).
    """
    p1 = crear_nucleon([0.0])
    p2 = crear_nucleon([1.0])
    p3 = crear_nucleon([2.0])
    pot = SimpleThreeBodyPotential(strength=0.5)
    f1, f2, f3 = pot.forces(p1, p2, p3)
    suma = xp.sum(f1 + f2 + f3)
    assert abs(suma) < 1e-10, f"La suma de fuerzas debe ser cero, suma={suma}"

def test_simple_threebody_forces_finiteness():
    """
    Testea que las fuerzas de tres cuerpos son finitas y no nulas para una configuración genérica (no alineada).
    """
    p1 = crear_nucleon([0.0, 0.0])
    p2 = crear_nucleon([1.0, 0.0])
    p3 = crear_nucleon([0.5, 1.0])
    pot = SimpleThreeBodyPotential(strength=0.5)
    f1, f2, f3 = pot.forces(p1, p2, p3)
    for f in [f1, f2, f3]:
        assert xp.all(xp.isfinite(f)), "Las fuerzas deben ser finitas"
        assert xp.linalg.norm(f) > 0.0, f"Las fuerzas no deben ser nulas, fuerza={f}"
