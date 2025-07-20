import sys
import numpy as np
from universe.numerics.backend import xp
from universe.solvers.maxwell_solver import MaxwellSolver
from universe.physics.constants import EPSILON_0, MU_0
from universe.quantum_states.quantum_numbers import QuantumNumbers
from universe.quantum_states.wavefunctions import full_wavefunction
from universe.particles.definitions import Electron, Proton, DynamicParticle, ParticleSystem
from universe.physics.electromagnetism.coulomb import coulomb_forces_on_particles

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("PyQtGraph o PyQt5 no están instalados. Instala con 'pip install pyqtgraph PyQt5' para visualización.")
    sys.exit(1)

# --- Parámetros EM ---
shape = (100, 100)
dx = 1e-3
def courant_dt(dx):
    c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
    return dx / (c * np.sqrt(2)) * 0.99
dt = courant_dt(dx)
pml_thickness = 15
solver = MaxwellSolver(
    shape, dx, dt,
    pml_thickness=pml_thickness,
    pml_sigma_max=20.0,
    use_pml=True
)
# Pulso gaussiano en el centro
y, x = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing='ij')
y0, x0 = shape[0] // 2, shape[1] // 2
sigma = 5.0
solver.Ez = xp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
solver.Hx = xp.zeros(shape)
solver.Hy = xp.zeros(shape)

# --- Estado cuántico ---
qn = QuantumNumbers(n=1, l=0, m=0, s=0.5, j=0.5)
psi = full_wavefunction(qn)

# --- Partículas clásicas (electrón y protón) ---
electron = Electron()
proton = Proton()
pos_e = xp.array([0.0, 0.0], dtype=xp.float64)
pos_p = xp.array([1e-10, 0.0], dtype=xp.float64)
vel_e = xp.array([0.0, 0.0], dtype=xp.float64)
vel_p = xp.array([0.0, 0.0], dtype=xp.float64)
p1 = DynamicParticle(static=electron, position=pos_e.copy(), velocity=vel_e.copy())
p2 = DynamicParticle(static=proton, position=pos_p.copy(), velocity=vel_p.copy())
system = ParticleSystem([p1, p2])

# --- Visualización PyQtGraph ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Universe Demo: EM + Cuántica + Partículas")
win.resize(1200, 600)

# Campo Ez
plot_em = win.addPlot(title="Campo Ez (EM)")
img_em = pg.ImageItem()
plot_em.addItem(img_em)
plot_em.setLabel('left', 'y')
plot_em.setLabel('bottom', 'x')

# Trayectorias de partículas
win.nextRow()
plot_particles = win.addPlot(title="Trayectorias de partículas")
curve_e = plot_particles.plot(pen=pg.mkPen('b', width=2), name="Electrón")
curve_p = plot_particles.plot(pen=pg.mkPen('r', width=2), name="Protón")
point_e = plot_particles.plot([float(pos_e[0])], [float(pos_e[1])], pen=None, symbol='o', symbolBrush='b', symbolSize=10)
point_p = plot_particles.plot([float(pos_p[0])], [float(pos_p[1])], pen=None, symbol='o', symbolBrush='r', symbolSize=10)
traj_e = [pos_e.copy()]
traj_p = [pos_p.copy()]

# --- Bucle de simulación ---
step_count = 0
max_steps = 500

def energia_em():
    return float(xp.sum(0.5 * EPSILON_0 * solver.Ez**2 + 0.5 * MU_0 * (solver.Hx**2 + solver.Hy**2)) * dx**2)

def update():
    global step_count, traj_e, traj_p
    # --- EM ---
    solver.step()
    em_np = xp.asnumpy(solver.Ez)
    img_em.setImage(em_np.T, autoLevels=True)
    # --- Partículas clásicas ---
    forces = coulomb_forces_on_particles(system)
    system.step(dt, lambda sys: forces)
    traj_e.append(system.particles[0].position.copy())
    traj_p.append(system.particles[1].position.copy())
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
    # --- Cuántica: <r> y probabilidad en [0, 1 Å] ---
    r = xp.linspace(1e-12, 2e-9, 1000)
    psi_r = psi(r, 0.0, 0.0)
    prob_density = xp.abs(psi_r) ** 2
    dr = r[1] - r[0]
    norm = xp.sum(prob_density * r ** 2) * dr
    expected_r = xp.sum(prob_density * r * r ** 2) * dr / norm
    mask = (r >= 0) & (r <= 1e-10)
    prob_region = xp.sum(prob_density[mask] * r[mask] ** 2) * dr / norm
    # Si xp es cupy, pasar a numpy para imprimir
    if hasattr(xp, 'asnumpy'):
        expected_r = float(xp.asnumpy(expected_r))
        prob_region = float(xp.asnumpy(prob_region))
    # --- Imprimir valores físicos clave ---
    if step_count % 10 == 0:
        print(f"Paso {step_count}: Energía EM = {energia_em():.3e} J, <r>_cuántico = {expected_r:.3e} m, Prob[0,1Å] = {prob_region:.3f}")
        print(f"  Electrón: pos = {pe_np}, Protón: pos = {pp_np}")
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
