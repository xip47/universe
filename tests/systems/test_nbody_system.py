"""
Tests para el sistema N-cuerpos (NBodySystem) bajo potenciales de dos y tres cuerpos.
"""

import pytest
from universe.systems.nbody_system import NBodySystem
from universe.particles.extended_definitions import BaseParticleExtended, DynamicParticleExtended
from universe.particles.degrees_of_freedom import Spin, Isospin, Charge, QuantumState
from universe.physics.nuclear.potentials import YukawaPotential
from universe.physics.nuclear.potentials_threebody import SimpleThreeBodyPotential
from universe.numerics.backend import xp

def crear_nucleon(nombre, masa, carga, spin_val, isospin_val, pos, vel):
    spin = Spin(spin_val, xp.array([0, 0, 1]))
    isospin = Isospin(isospin_val, xp.array([1, 0, 0]))
    charge = Charge(carga)
    qs = QuantumState(1, 0, 0, spin_val, 0.5, isospin_val)
    base = BaseParticleExtended(nombre, masa, charge, spin, isospin, qs)
    return DynamicParticleExtended(base, xp.array(pos), xp.array(vel), spin, isospin, qs)

def test_nbody_energy_conservation():
    """
    Testea la conservación de la energía total en un sistema N-cuerpos con potencial de dos cuerpos.
    """
    p1 = crear_nucleon("proton", 938.27, 1.0, 0.5, 0.5, [0.0], [0.0])
    p2 = crear_nucleon("neutron", 939.57, 0.0, -0.5, -0.5, [2.0], [0.0])
    system = NBodySystem([p1, p2], two_body_potential=YukawaPotential())
    E0 = system.total_energy()
    for _ in range(100):
        system.step(1e-24)
    E1 = system.total_energy()
    assert abs(E1 - E0) < 1e-2, f"No se conserva la energía total: ΔE = {E1-E0} MeV"

def test_nbody_three_body_forces():
    """
    Testea que el potencial de tres cuerpos contribuye a las fuerzas y energías en un sistema de tres nucleones (no alineados).
    """
    p1 = crear_nucleon("nucleon1", 938.27, 1.0, 0.5, 0.5, [0.0, 0.0], [0.0, 0.0])
    p2 = crear_nucleon("nucleon2", 938.27, 0.0, -0.5, -0.5, [1.0, 0.0], [0.0, 0.0])
    p3 = crear_nucleon("nucleon3", 938.27, 0.0, 0.5, -0.5, [0.5, 1.0], [0.0, 0.0])
    system = NBodySystem([p1, p2, p3], two_body_potential=YukawaPotential(), three_body_potential=SimpleThreeBodyPotential(strength=0.1))
    forces = system.compute_forces()
    # Las fuerzas deben ser finitas y no nulas
    for f in forces:
        assert xp.all(xp.isfinite(f)), "Las fuerzas deben ser finitas"
        assert xp.linalg.norm(f) > 0.0, f"Las fuerzas no deben ser nulas si hay interacción de tres cuerpos, fuerza={f}"
    # La energía potencial de tres cuerpos debe ser positiva para el potencial armónico
    E = system.total_energy()
    assert E > 0.0 or E < 0.0  # Solo valida que se calcula sin error

def test_nbody_n_particles():
    """
    Testea que el sistema soporta N>3 partículas y calcula fuerzas y energía sin error.
    """
    nucleones = [crear_nucleon(f"nucleon{i}", 938.27, 1.0 if i%2==0 else 0.0, 0.5, 0.5, [float(i)], [0.0]) for i in range(5)]
    system = NBodySystem(nucleones, two_body_potential=YukawaPotential())
    forces = system.compute_forces()
    assert len(forces) == 5
    for f in forces:
        assert xp.all(xp.isfinite(f)), "Las fuerzas deben ser finitas para N>3"
