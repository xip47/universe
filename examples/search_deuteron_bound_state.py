"""
Búsqueda automática del estado ligado realista del deuterón (protón-neutrón) con potencial nuclear modular (GeneralNuclearPotential, canal 3S1).

Automatic search for the realistic bound state of the deuteron (proton-neutron) with modular nuclear potential (GeneralNuclearPotential, 3S1 channel).

Barre la separación inicial y reporta la energía de enlace mínima y la condición óptima.

Usage
-----
Ejecuta este script para encontrar la condición inicial que reproduce la energía de enlace experimental del deuterón.

Requisitos
----------
- numpy
- universe (con CuPy si se desea GPU)
"""
from typing import List, Tuple
import numpy as np
from universe.particles.extended_definitions import BaseParticleExtended, DynamicParticleExtended
from universe.particles.degrees_of_freedom import Spin, Isospin, Charge
from universe.physics.nuclear.potentials_extended import GeneralNuclearPotential
from universe.systems.coupled_system import CoupledSystem
from universe.numerics import backend as backend

# --- Parámetros físicos ---
mass_p: float = 938.27  # MeV/c^2 (protón)
mass_n: float = 939.57  # MeV/c^2 (neutrón)
reid = GeneralNuclearPotential.preset_reid93(channel="3S1")
dt: float = 0.005  # fm/c (más pequeño para mayor precisión)
steps: int = 5000

# --- Barrido de separaciones iniciales ---
sep_range = np.linspace(1.5, 3.0, 16)  # fm
results: List[Tuple[float, float, float]] = []  # (sep, E_min, r_min)

for sep in sep_range:
    # Inicialización de protón y neutrón
    pos_init = [backend.xp.array([0.0, 0.0]), backend.xp.array([sep, 0.0])]
    vel_init = [backend.xp.array([0.0, 0.0]), backend.xp.array([0.0, 0.0])]
    base_particles = [
        BaseParticleExtended("protón", mass_p, Charge(1), Spin(0.5), Isospin(0.5)),
        BaseParticleExtended("neutrón", mass_n, Charge(0), Spin(0.5), Isospin(-0.5)),
    ]
    particles = [
        DynamicParticleExtended(
            static=bp,
            position=pos.copy(),
            velocity=vel.copy(),
            spin=Spin(0.5),
            isospin=bp.isospin
        )
        for bp, pos, vel in zip(base_particles, pos_init, vel_init)
    ]
    system = CoupledSystem(particles, nuclear_potential=reid, use_coulomb=False)
    hist_E = []
    hist_dist = []
    for _ in range(steps):
        system.step(dt)
        E_total = system.total_energy()
        r_pn = float(backend.xp.linalg.norm(system.particles[1].position - system.particles[0].position))
        hist_E.append(E_total)
        hist_dist.append(r_pn)
    E_min = min(hist_E)
    r_min = hist_dist[np.argmin(hist_E)]
    results.append((sep, E_min, r_min))
    print(f"Separación inicial: {sep:.2f} fm | Energía mínima: {E_min:.6f} MeV | r_min: {r_min:.4f} fm")

# --- Reporte final ---
sep_opt, E_min_opt, r_min_opt = min(results, key=lambda x: abs(x[1] + 2.224))  # Más cercano a -2.224 MeV
print("\n--- Resultado óptimo ---")
print(f"Separación inicial óptima: {sep_opt:.3f} fm")
print(f"Energía de enlace mínima simulada: {E_min_opt:.6f} MeV (experimental: -2.224 MeV)")
print(f"Distancia protón-neutrón en el mínimo: {r_min_opt:.4f} fm")
