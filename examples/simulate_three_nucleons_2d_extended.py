"""
Simulación gráfica 2D de tres nucleones extendidos (spin, isospin, carga) con potencial de Reid93 y fuerza de Coulomb.

2D graphical simulation of three extended nucleons (spin, isospin, charge) with Reid93 potential and Coulomb force.

Este ejemplo utiliza la arquitectura extendida: DynamicParticleExtended, CoupledSystem, xp (NumPy/CuPy), y PyQtGraph.

Usage
-----
Ejecuta este script para visualizar la dinámica realista de tres nucleones en 2D, con logs físicos y gráficas de energía y distancias.

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
from universe.physics.nuclear.potentials import Reid93Potential
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
mass_n: float = 938.92  # MeV/c^2
reid = Reid93Potential(channel="1S0")
dt: float = 0.01  # fm/c (unidades naturales)
steps: int = 2000

# --- Inicialización de nucleones extendidos ---
pos_init: List[backend.xp.ndarray] = [backend.xp.array([0.0, 0.0]), backend.xp.array([1.0, 0.0]), backend.xp.array([5.0, 4.33])]
vel_init: List[backend.xp.ndarray] = [backend.xp.array([2.0, 0.0]), backend.xp.array([-2.0, 0.0]), backend.xp.array([0.0, 0.0])]

nucleon_types = [
    ("protón", Charge(1), Isospin(0.5)),
    ("neutrón", Charge(0), Isospin(-0.5)),
    ("protón", Charge(1), Isospin(0.5)),
]

base_particles: List[BaseParticleExtended] = [
    BaseParticleExtended(name, mass_n, charge, Spin(0.5), isospin) for (name, charge, isospin) in nucleon_types
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

# --- Sistema acoplado (nuclear + Coulomb) ---
system = CoupledSystem(particles, nuclear_potential=reid, use_coulomb=True)

# --- Historial para gráficas ---
traj: List[List[backend.xp.ndarray]] = [[p.position.copy()] for p in system.particles]
hist_E: List[float] = []
hist_dist: List[float] = []

# --- Visualización PyQtGraph ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Simulación 2D: Tres nucleones extendidos (Reid93 + Coulomb)")
win.resize(1200, 800)

# Trayectorias
plot_traj = win.addPlot(title="Trayectorias de nucleones (2D)")
colors = ['r', 'g', 'b']
curves = [plot_traj.plot(pen=pg.mkPen(c, width=2), name=f"Nucleón {i+1}") for i, c in enumerate(colors)]
points = [plot_traj.plot([float(p.position[0])], [float(p.position[1])], pen=None, symbol='o', symbolBrush=c, symbolSize=12) for p, c in zip(system.particles, colors)]
plot_traj.setLabel('left', 'y [fm]')
plot_traj.setLabel('bottom', 'x [fm]')

# Energía total
del_E = win.nextRow()
plot_E = win.addPlot(title="Energía total del sistema")
curve_E = plot_E.plot(pen=pg.mkPen('c', width=2), name="E_total")
plot_E.setLabel('left', 'Energía [MeV]')
plot_E.setLabel('bottom', 'Paso')

# Distancias mínimas
win.nextRow()
plot_dist = win.addPlot(title="Distancia mínima entre nucleones")
curve_dist = plot_dist.plot(pen=pg.mkPen('m', width=2), name="r_min")
plot_dist.setLabel('left', 'r_min [fm]')
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
    # Energía y distancias
    E_total: float = system.total_energy()
    hist_E.append(E_total)
    dists = [float(backend.xp.linalg.norm(system.particles[j].position - system.particles[i].position))
             for i in range(3) for j in range(i+1, 3)]
    hist_dist.append(min(dists))
    # Actualizar gráficos
    for i in range(3):
        t = backend.xp.asnumpy(backend.xp.stack(traj[i]))
        curves[i].setData(t[:, 0], t[:, 1])
        points[i].setData([float(system.particles[i].position[0])], [float(system.particles[i].position[1])])
    curve_E.setData(range(len(hist_E)), hist_E)
    curve_dist.setData(range(len(hist_dist)), hist_dist)
    # Logs físicos cada 10 pasos
    if step_count % 10 == 0:
        KE = sum(0.5 * p.static.mass_mev * float(backend.xp.linalg.norm(p.velocity))**2 for p in system.particles)
        PE_nuc = 0.0
        PE_coul = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                r = float(backend.xp.linalg.norm(system.particles[j].position - system.particles[i].position))
                PE_nuc += float(reid.potential(
                    r,
                    spin1=system.particles[i].spin, spin2=system.particles[j].spin,
                    isospin1=system.particles[i].isospin, isospin2=system.particles[j].isospin,
                    quantum_state1=system.particles[i].quantum_state, quantum_state2=system.particles[j].quantum_state
                ))
                q1 = getattr(system.particles[i].static.charge, 'value', 0.0)
                q2 = getattr(system.particles[j].static.charge, 'value', 0.0)
                if q1 != 0.0 and q2 != 0.0:
                    # Potencial de Coulomb en MeV/fm
                    from universe.physics.electromagnetism.coulomb import coulomb_potential_mevfm
                    PE_coul += coulomb_potential_mevfm(q1, q2,
                        tuple(backend.xp.asnumpy(system.particles[i].position)),
                        tuple(backend.xp.asnumpy(system.particles[j].position)))
        print(f"Paso {step_count}: E_total = {E_total:.6f} MeV, KE = {KE:.6f} MeV, PE_nuc = {PE_nuc:.6f} MeV, PE_coul = {PE_coul:.6f} MeV")
        for i, p in enumerate(system.particles):
            pos = backend.xp.asnumpy(p.position)
            vel = backend.xp.asnumpy(p.velocity)
            print(f"  Nucleón {i+1}: pos = ({pos[0]:.4f}, {pos[1]:.4f}) fm, vel = ({vel[0]:.3e}, {vel[1]:.3e}) fm/c")
        print(f"  Distancias: {', '.join(f'{d:.4f}' for d in dists)} fm")
    step_count += 1
    if step_count > max_steps:
        timer.stop()
        print("Simulación finalizada.")
        return

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

if __name__ == '__main__':
    app.exec_()
