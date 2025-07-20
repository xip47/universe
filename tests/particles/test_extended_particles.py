"""
Tests para grados de libertad y partículas extendidas.
"""

import numpy as np
import pytest
from universe.particles.degrees_of_freedom import Spin
from universe.particles.degrees_of_freedom import Isospin
from universe.particles.degrees_of_freedom import Charge
from universe.particles.degrees_of_freedom import QuantumState
from universe.particles.extended_definitions import BaseParticleExtended
from universe.particles.extended_definitions import DynamicParticleExtended

def test_spin_init():
    """
    Testea la inicialización y atributos de Spin.
    """
    s = Spin(0.5, np.array([0, 0, 1]))
    assert s.value == 0.5
    assert np.allclose(s.vector, [0, 0, 1])

def test_isospin_init():
    """
    Testea la inicialización y atributos de Isospin.
    """
    t = Isospin(0.5, np.array([1, 0, 0]))
    assert t.value == 0.5
    assert np.allclose(t.vector, [1, 0, 0])

def test_charge_init():
    """
    Testea la inicialización y atributos de Charge.
    """
    q = Charge(1.0)
    assert q.value == 1.0

def test_quantum_state_init():
    """
    Testea la inicialización y atributos de QuantumState.
    """
    qs = QuantumState(n=1, l=0, m=0, s=0.5, j=0.5, t=0.5)
    assert qs.n == 1
    assert qs.l == 0
    assert qs.m == 0
    assert qs.s == 0.5
    assert qs.j == 0.5
    assert qs.t == 0.5

def test_base_particle_extended():
    """
    Testea la inicialización de BaseParticleExtended con todos los grados de libertad.
    """
    spin = Spin(0.5, np.array([0, 0, 1]))
    isospin = Isospin(0.5, np.array([1, 0, 0]))
    charge = Charge(1.0)
    qs = QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
    p = BaseParticleExtended("proton", 938.27, charge, spin, isospin, qs)
    assert p.name == "proton"
    assert p.mass_mev == 938.27
    assert p.charge.value == 1.0
    assert p.spin.value == 0.5
    assert p.isospin.value == 0.5
    assert p.quantum_state.n == 1

def test_dynamic_particle_extended():
    """
    Testea la inicialización de DynamicParticleExtended y la compatibilidad de atributos.
    """
    spin = Spin(0.5, np.array([0, 0, 1]))
    isospin = Isospin(0.5, np.array([1, 0, 0]))
    charge = Charge(1.0)
    qs = QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
    base = BaseParticleExtended("proton", 938.27, charge, spin, isospin, qs)
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    dp = DynamicParticleExtended(base, pos, vel, spin, isospin, qs)
    assert np.allclose(dp.position, [0, 0, 0])
    assert np.allclose(dp.velocity, [0, 0, 0])
    assert dp.spin.value == 0.5
    assert dp.isospin.value == 0.5
    assert dp.quantum_state.n == 1
