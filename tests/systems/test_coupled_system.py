"""
Tests para el sistema acoplado de partículas extendidas bajo fuerzas nucleares y electromagnéticas.
"""

import pytest
from universe.systems.coupled_system import CoupledSystem
from universe.particles.extended_definitions import BaseParticleExtended, DynamicParticleExtended
from universe.particles.degrees_of_freedom import Spin, Isospin, Charge, QuantumState
from universe.physics.nuclear.potentials import Reid93Potential
from universe.numerics.backend import xp


def test_coupled_system_energy_conservation():
    """
    Testea la conservación de la energía total en un sistema acoplado (nuclear + Coulomb).
    """
    # Dos protones en 1D, con spin/isospin
    spin1 = Spin(0.5, xp.array([0, 0, 1]))
    spin2 = Spin(-0.5, xp.array([0, 0, -1]))
    isospin1 = Isospin(0.5, xp.array([1, 0, 0]))
    isospin2 = Isospin(-0.5, xp.array([-1, 0, 0]))
    charge1 = Charge(1.0)
    charge2 = Charge(1.0)
    qs1 = QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
    qs2 = QuantumState(1, 0, 0, -0.5, 0.5, -0.5)
    base1 = BaseParticleExtended("proton", 938.27, charge1, spin1, isospin1, qs1)
    base2 = BaseParticleExtended("proton", 938.27, charge2, spin2, isospin2, qs2)
    pos1 = xp.array([0.0])
    pos2 = xp.array([2.0])
    vel1 = xp.array([0.0])
    vel2 = xp.array([0.0])
    dp1 = DynamicParticleExtended(base1, pos1, vel1, spin1, isospin1, qs1)
    dp2 = DynamicParticleExtended(base2, pos2, vel2, spin2, isospin2, qs2)
    system = CoupledSystem([dp1, dp2], nuclear_potential=Reid93Potential(channel="1S0"), use_coulomb=True)
    E0 = system.total_energy()
    for _ in range(100):
        system.step(1e-24)
    E1 = system.total_energy()
    assert abs(E1 - E0) < 1e-2, f"No se conserva la energía total: ΔE = {E1-E0} MeV"

def test_coupled_system_force_propagation():
    """
    Testea que las fuerzas nucleares y de Coulomb se propagan correctamente en el sistema acoplado.
    """
    spin1 = Spin(0.5, xp.array([0, 0, 1]))
    spin2 = Spin(-0.5, xp.array([0, 0, -1]))
    isospin1 = Isospin(0.5, xp.array([1, 0, 0]))
    isospin2 = Isospin(-0.5, xp.array([-1, 0, 0]))
    charge1 = Charge(1.0)
    charge2 = Charge(-1.0)
    qs1 = QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
    qs2 = QuantumState(1, 0, 0, -0.5, 0.5, -0.5)
    base1 = BaseParticleExtended("proton", 938.27, charge1, spin1, isospin1, qs1)
    base2 = BaseParticleExtended("electron", 0.511, charge2, spin2, isospin2, qs2)
    pos1 = xp.array([0.0])
    pos2 = xp.array([1.0])
    vel1 = xp.array([0.0])
    vel2 = xp.array([0.0])
    dp1 = DynamicParticleExtended(base1, pos1, vel1, spin1, isospin1, qs1)
    dp2 = DynamicParticleExtended(base2, pos2, vel2, spin2, isospin2, qs2)
    system = CoupledSystem([dp1, dp2], nuclear_potential=Reid93Potential(channel="1S0"), use_coulomb=True)
    forces = system.compute_forces()
    # Debe haber fuerzas opuestas y finitas
    assert xp.all(xp.isfinite(forces[0]))
    assert xp.all(xp.isfinite(forces[1]))
    assert xp.allclose(forces[0], -forces[1], atol=1e-10)

def test_coupled_system_spin_isospin():
    """
    Testea que los grados de libertad de spin/isospin se propagan y pueden ser accedidos en el sistema acoplado.
    """
    spin = Spin(1.0, xp.array([0, 1, 0]))
    isospin = Isospin(1.0, xp.array([0, 1, 0]))
    charge = Charge(0.0)
    qs = QuantumState(2, 1, 1, 1.0, 1.0, 1.0)
    base = BaseParticleExtended("neutron", 939.57, charge, spin, isospin, qs)
    pos = xp.array([0.0])
    vel = xp.array([0.0])
    dp = DynamicParticleExtended(base, pos, vel, spin, isospin, qs)
    system = CoupledSystem([dp], nuclear_potential=None, use_coulomb=False)
    assert system.particles[0].spin.value == 1.0
    assert xp.allclose(system.particles[0].spin.vector, xp.array([0, 1, 0]))
    assert system.particles[0].isospin.value == 1.0
    assert xp.allclose(system.particles[0].isospin.vector, xp.array([0, 1, 0]))
    assert system.particles[0].quantum_state.n == 2
