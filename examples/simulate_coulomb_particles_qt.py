import sys
from universe.numerics.backend import xp
import numpy as np
from universe.particles.definitions import Quark
from universe.particles.definitions import Lepton
from universe.particles.definitions import DynamicParticle
from universe.particles.definitions import ParticleSystem
from universe.physics.electromagnetism import coulomb_forces_on_particles
from universe.physics.constants import E_CHARGE

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("PyQtGraph o PyQt5 no están instalados. Instala con 'pip install pyqtgraph PyQt5' para visualización.")
    sys.exit(1)

# --- Definición de partículas realistas ---
class Electron(Lepton):
    def __init__(self):
        super().__init__(
            name="electron",
            symbol="e-",
            mass_mev=0.510998950,
            charge=-1.0,
            spin=0.5,
            generation=1,
            lepton_number=1
        )

class Proton(Quark):
    def __init__(self):
        super().__init__(
            name="proton",
            symbol="p+",
            mass_mev=938.27208816,
            charge=+1.0,
            spin=0.5,
            generation=1,
            color_charge=False
        )

# --- Parámetros de simulación física ---
dim = 2
pos_e = xp.array([0.0, 0.0], dtype=xp.float64)
pos_p = xp.array([1e-10, 0.0], dtype=xp.float64)  # 1 Ångström de separación inicial
vel_e = xp.array([0.0, 0.0], dtype=xp.float64)
vel_p = xp.array([0.0, 0.0], dtype=xp.float64)
electron = Electron()
proton = Proton()
p1 = DynamicParticle(static=electron, position=pos_e.copy(), velocity=vel_e.copy())
p2 = DynamicParticle(static=proton, position=pos_p.copy(), velocity=vel_p.copy())
system = ParticleSystem([p1, p2])

# --- Parámetros de integración ---
dt = 1e-18  # Paso temporal (s)
steps = 10000

# --- Visualización con PyQtGraph ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Simulación: Electrón y Protón bajo Fuerza de Coulomb")
plot = win.addPlot(title="Trayectorias (2D)")
plot.setXRange(-2e-10, 2e-10)
plot.setYRange(-2e-10, 2e-10)
plot.setLabel('left', 'y (m)')
plot.setLabel('bottom', 'x (m)')
curve_e = plot.plot(pen=pg.mkPen('b', width=2), name="Electrón")
curve_p = plot.plot(pen=pg.mkPen('r', width=2), name="Protón")
point_e = plot.plot([float(pos_e[0])], [float(pos_e[1])], pen=None, symbol='o', symbolBrush='b', symbolSize=10)
point_p = plot.plot([float(pos_p[0])], [float(pos_p[1])], pen=None, symbol='o', symbolBrush='r', symbolSize=10)

traj_e = [pos_e.copy()]
traj_p = [pos_p.copy()]

# --- Cálculo de energía ---
def energia_cinetica(p):
    m_kg = p.static.mass_mev * 1.78266192e-30
    v = xp.asnumpy(p.velocity) if hasattr(xp, 'asnumpy') else p.velocity
    return 0.5 * m_kg * np.sum(v ** 2)

def energia_potencial_coulomb(p1, p2):
    q1 = p1.static.charge * E_CHARGE
    q2 = p2.static.charge * E_CHARGE
    r = p1.position - p2.position
    r = xp.asnumpy(r) if hasattr(xp, 'asnumpy') else r
    dist = np.linalg.norm(r)
    if dist == 0:
        return 0.0
    return (1 / (4 * np.pi * 8.854187817e-12)) * (q1 * q2) / dist

# --- Bucle de simulación ---
def update():
    global traj_e, traj_p
    forces = coulomb_forces_on_particles(system)
    system.step(dt, lambda sys: forces)
    traj_e.append(system.particles[0].position.copy())
    traj_p.append(system.particles[1].position.copy())
    if len(traj_e) > steps:
        timer.stop()
        print("Simulación finalizada.")
        return
    # Actualizar visualización
    te = xp.stack(traj_e)
    tp = xp.stack(traj_p)
    te_np = xp.asnumpy(te) if hasattr(xp, 'asnumpy') else te
    tp_np = xp.asnumpy(tp) if hasattr(xp, 'asnumpy') else tp
    curve_e.setData(te_np[:, 0], te_np[:, 1])
    curve_p.setData(tp_np[:, 0], tp_np[:, 1])
    pe = system.particles[0].position
    pp = system.particles[1].position
    pe_np = xp.asnumpy(pe) if hasattr(xp, 'asnumpy') else pe
    pp_np = xp.asnumpy(pp) if hasattr(xp, 'asnumpy') else pp
    point_e.setData([float(pe_np[0])], [float(pe_np[1])])
    point_p.setData([float(pp_np[0])], [float(pp_np[1])])
    # Imprimir datos relevantes
    if len(traj_e) % 100 == 0:
        ke = energia_cinetica(system.particles[0]) + energia_cinetica(system.particles[1])
        pe = energia_potencial_coulomb(system.particles[0], system.particles[1])
        r = system.particles[0].position - system.particles[1].position
        r = xp.asnumpy(r) if hasattr(xp, 'asnumpy') else r
        print(f"Paso {len(traj_e)}: r = {np.linalg.norm(r):.3e} m, "
              f"KE = {ke:.3e} J, PE = {pe:.3e} J, E_total = {ke+pe:.3e} J")

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

if __name__ == '__main__':
    app.exec_()
