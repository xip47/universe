import sys
from universe.numerics.backend import xp
from universe.physics.nuclear.potentials import Reid93Potential
from universe.particles.definitions import DynamicParticle, ParticleSystem

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("PyQtGraph o PyQt5 no están instalados. Instala con 'pip install pyqtgraph PyQt5' para visualización.")
    sys.exit(1)

# --- Parámetros físicos ---
mass_n = 938.92  # MeV/c^2
reid = Reid93Potential(channel="1S0")
dt = 1e-24  # s
steps = 2000

# --- Inicialización de partículas (dinámica rica: colisión) ---
pos_init = [xp.array([0.0, 0.0]), xp.array([1.0, 0.0]), xp.array([5.0, 4.33])]
vel_init = [xp.array([2.0, 0.0]), xp.array([-2.0, 0.0]), xp.array([0.0, 0.0])]
ps = [DynamicParticle(static=None, position=pos.copy(), velocity=vel.copy()) for pos, vel in zip(pos_init, vel_init)]
for p in ps:
    p.static = type('Dummy', (), {'mass_mev': mass_n})()
system = ParticleSystem(ps)

# --- Fuerza total sobre cada partícula ---
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

# --- Inicialización para Verlet ---
forces = force_fn(system)
for p, f in zip(system.particles, forces):
    a = f / mass_n
    p.position -= p.velocity * dt - 0.5 * a * dt**2

# --- Historial para gráficas ---
traj = [[p.position.copy()] for p in system.particles]
hist_E = []
hist_dist = []

# --- Visualización PyQtGraph ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Simulación 2D: Tres nucleones con Reid93 (Verlet)")
win.resize(1200, 800)

# Trayectorias
plot_traj = win.addPlot(title="Trayectorias de nucleones (2D)")
colors = ['r', 'g', 'b']
curves = [plot_traj.plot(pen=pg.mkPen(c, width=2), name=f"Nucleón {i+1}") for i, c in enumerate(colors)]
points = [plot_traj.plot([float(p.position[0])], [float(p.position[1])], pen=None, symbol='o', symbolBrush=c, symbolSize=12) for p, c in zip(system.particles, colors)]
plot_traj.setLabel('left', 'y [fm]')
plot_traj.setLabel('bottom', 'x [fm]')

# Energía total
win.nextRow()
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
step_count = 0
max_steps = steps

def update():
    global step_count
    # Verlet
    forces = force_fn(system)
    for i, p in enumerate(system.particles):
        a = forces[i] / mass_n
        new_pos = 2 * p.position - (p.position - p.velocity * dt + 0.5 * a * dt**2) + a * dt**2
        p.velocity = (new_pos - p.position) / dt
        p.position = new_pos
        traj[i].append(p.position.copy())
    # Energía y distancias
    KE = sum(0.5 * mass_n * float(xp.linalg.norm(p.velocity))**2 for p in system.particles)
    PE = 0.0
    dists = []
    for i in range(3):
        for j in range(i+1, 3):
            d = float(xp.linalg.norm(system.particles[j].position - system.particles[i].position))
            PE += float(reid.potential(d))
            dists.append(d)
    E_total = KE + PE
    hist_E.append(E_total)
    hist_dist.append(min(dists))
    # Actualizar gráficos
    for i in range(3):
        t = xp.asnumpy(xp.stack(traj[i]))
        curves[i].setData(t[:, 0], t[:, 1])
        points[i].setData([float(system.particles[i].position[0])], [float(system.particles[i].position[1])])
    curve_E.setData(range(len(hist_E)), hist_E)
    curve_dist.setData(range(len(hist_dist)), hist_dist)
    # Logs físicos cada 10 pasos
    if step_count % 10 == 0:
        print(f"Paso {step_count}: Energía total = {E_total:.6f} MeV, Energía cinética = {KE:.6f} MeV, Energía potencial = {PE:.6f} MeV")
        for i, p in enumerate(system.particles):
            pos = xp.asnumpy(p.position)
            vel = xp.asnumpy(p.velocity)
            print(f"  Nucleón {i+1}: pos = ({pos[0]:.4f}, {pos[1]:.4f}) fm, vel = ({vel[0]:.3e}, {vel[1]:.3e}) fm/s")
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
