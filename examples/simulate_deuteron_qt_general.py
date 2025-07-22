"""
Simulación gráfica 2D del deuterón (protón-neutrón) con potencial nuclear modular (GeneralNuclearPotential, canal 3S1).

2D graphical simulation of the deuteron (proton-neutron) with modular nuclear potential (GeneralNuclearPotential, 3S1 channel).

Incluye logs avanzados, búsqueda de energía de enlace y visualización profesional con PyQtGraph.

Usage
-----
Ejecuta este script para validar la física del deuterón y comparar con la energía de enlace experimental.

Requisitos
----------
- pyqtgraph
- PyQt5

Instala con:
    pip install pyqtgraph PyQt5
"""
from typing import List
from universe.particles.extended_definitions import BaseParticleExtended, DynamicParticleExtended
from universe.particles.degrees_of_freedom import Spin, Isospin, Charge
from universe.physics.nuclear.potentials_extended import GeneralNuclearPotential
from universe.systems.coupled_system import CoupledSystem
from universe.numerics import backend as backend

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("PyQtGraph o PyQt5 no están instalados. Instala con 'pip install pyqtgraph PyQt5' para visualización.")
    import sys
    sys.exit(1)

# --- Parámetros físicos ---
mass_p: float = 938.27  # MeV/c^2 (protón)
mass_n: float = 939.57  # MeV/c^2 (neutrón)
reid = GeneralNuclearPotential.preset_reid93(channel="3S1")
dt: float = 0.01  # fm/c (unidades naturales)
steps: int = 3000

# --- Inicialización de protón y neutrón (condiciones para buscar estado ligado) ---
pos_init: List[backend.xp.ndarray] = [backend.xp.array([0.0, 0.0]), backend.xp.array([2.0, 0.0])]
vel_init: List[backend.xp.ndarray] = [backend.xp.array([0.0, 0.0]), backend.xp.array([0.0, 0.0])]

base_particles: List[BaseParticleExtended] = [
    BaseParticleExtended("protón", mass_p, Charge(1), Spin(0.5), Isospin(0.5)),
    BaseParticleExtended("neutrón", mass_n, Charge(0), Spin(0.5), Isospin(-0.5)),
]

particles: List[DynamicParticleExtended] = [
    DynamicParticleExtended(
        static=bp,
        position=pos.copy(),
        velocity=vel.copy(),
        spin=Spin(0.5),
        isospin=bp.isospin
    )
    for bp, pos, vel in zip(base_particles, pos_init, vel_init)
]

# --- Sistema acoplado (solo nuclear, sin Coulomb para deuterón) ---
system = CoupledSystem(particles, nuclear_potential=reid, use_coulomb=False)

# --- Historial para gráficas ---
traj: List[List[backend.xp.ndarray]] = [[p.position.copy()] for p in system.particles]
hist_E: List[float] = []
hist_PE_nuc: List[float] = []
hist_KE: List[float] = []
hist_dist: List[float] = []

# --- Visualización PyQtGraph ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Simulación 2D: Deuterón (GeneralNuclearPotential 3S1)")
win.resize(1200, 700)

# Trayectorias
plot_traj = win.addPlot(title="Trayectorias de nucleones (2D)")
colors = ['r', 'b']
curves = [plot_traj.plot(pen=pg.mkPen(c, width=2), name=f"Nucleón {i+1}") for i, c in enumerate(colors)]
points = [plot_traj.plot([float(p.position[0])], [float(p.position[1])], pen=None, symbol='o', symbolBrush=c, symbolSize=14) for p, c in zip(system.particles, colors)]
plot_traj.setLabel('left', 'y [fm]')
plot_traj.setLabel('bottom', 'x [fm]')

# Energía total
del_E = win.nextRow()
plot_E = win.addPlot(title="Energía total del sistema")
curve_E = plot_E.plot(pen=pg.mkPen('c', width=2), name="E_total")
plot_E.setLabel('left', 'Energía [MeV]')
plot_E.setLabel('bottom', 'Paso')

# Energía potencial nuclear
del_PE_nuc = win.nextRow()
plot_PE_nuc = win.addPlot(title="Energía potencial nuclear")
curve_PE_nuc = plot_PE_nuc.plot(pen=pg.mkPen('y', width=2), name="PE_nuc")
plot_PE_nuc.setLabel('left', 'PE_nuc [MeV]')
plot_PE_nuc.setLabel('bottom', 'Paso')

# Energía cinética
del_KE = win.nextRow()
plot_KE = win.addPlot(title="Energía cinética total")
curve_KE = plot_KE.plot(pen=pg.mkPen('g', width=2), name="KE")
plot_KE.setLabel('left', 'KE [MeV]')
plot_KE.setLabel('bottom', 'Paso')

# Distancia protón-neutrón
win.nextRow()
plot_dist = win.addPlot(title="Distancia protón-neutrón")
curve_dist = plot_dist.plot(pen=pg.mkPen('b', width=2), name="r_pn")
plot_dist.setLabel('left', 'r [fm]')
plot_dist.setLabel('bottom', 'Paso')

# --- Bucle de simulación ---
step_count: int = 0
max_steps: int = steps

def update() -> None:
    """
    Actualiza la simulación un paso, integrando la dinámica y actualizando las gráficas.

    Updates the simulation one step, integrating the dynamics and updating the plots.
    """
    global step_count
    system.step(dt)
    # Guardar trayectorias
    for i, p in enumerate(system.particles):
        traj[i].append(p.position.copy())
    # Energía y distancia
    E_total: float = system.total_energy()
    KE = sum(0.5 * p.static.mass_mev * float(backend.xp.linalg.norm(p.velocity))**2 for p in system.particles)
    PE_nuc = 0.0
    r_pn = float(backend.xp.linalg.norm(system.particles[1].position - system.particles[0].position))
    PE_nuc += float(reid.potential(
        r_pn,
        spin1=system.particles[0].spin, spin2=system.particles[1].spin,
        isospin1=system.particles[0].isospin, isospin2=system.particles[1].isospin,
        quantum_state1=system.particles[0].quantum_state, quantum_state2=system.particles[1].quantum_state
    ))
    hist_E.append(E_total)
    hist_PE_nuc.append(PE_nuc)
    hist_KE.append(KE)
    hist_dist.append(r_pn)
    # Actualizar gráficos
    for i in range(2):
        t = backend.xp.asnumpy(backend.xp.stack(traj[i]))
        curves[i].setData(t[:, 0], t[:, 1])
        points[i].setData([float(system.particles[i].position[0])], [float(system.particles[i].position[1])])
    curve_E.setData(range(len(hist_E)), hist_E)
    curve_PE_nuc.setData(range(len(hist_PE_nuc)), hist_PE_nuc)
    curve_KE.setData(range(len(hist_KE)), hist_KE)
    curve_dist.setData(range(len(hist_dist)), hist_dist)
    # Logs físicos avanzados cada 10 pasos
    if step_count % 10 == 0:
        print(f"Paso {step_count}: E_total = {E_total:.6f} MeV, KE = {KE:.6f} MeV, PE_nuc = {PE_nuc:.6f} MeV, r_pn = {r_pn:.4f} fm")
        for i, p in enumerate(system.particles):
            pos = backend.xp.asnumpy(p.position)
            vel = backend.xp.asnumpy(p.velocity)
            print(f"  Nucleón {i+1}: pos = ({pos[0]:.4f}, {pos[1]:.4f}) fm, vel = ({vel[0]:.3e}, {vel[1]:.3e}) fm/c")
    step_count += 1
    if step_count > max_steps:
        timer.stop()
        print("Simulación finalizada.")
        # Reporte de energía de enlace
        E_min = min(hist_E)
        print(f"\nEnergía de enlace mínima simulada: {E_min:.6f} MeV (experimental: -2.224 MeV)")
        return

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

if __name__ == '__main__':
    app.exec_()
