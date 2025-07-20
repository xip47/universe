import sys
import numpy as np
from universe.numerics.backend import xp
from universe.physics.nuclear.potentials import Reid93Potential

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("PyQtGraph o PyQt5 no están instalados. Instala con 'pip install pyqtgraph PyQt5' para visualización.")
    sys.exit(1)

# --- Parámetros físicos ---
reid = Reid93Potential(channel="1S0")
r = xp.linspace(0.5, 5.0, 200)  # 0.5-5 fm
V = reid.potential(r)
F = reid.force(r)

# --- Inicialización de partículas (simplificado: dos nucleones en 1D) ---
pos_n1 = xp.array([0.0], dtype=xp.float64)  # fm (más separado)
pos_n2 = xp.array([5.0], dtype=xp.float64)  # fm (más separado)
vel_n1 = xp.array([0.0], dtype=xp.float64)
vel_n2 = xp.array([0.0], dtype=xp.float64)
mass_n = 938.92  # MeV/c^2 (valor físico real)
dt = 1e-22  # s (valor físico realista)
traj_n1 = [pos_n1.copy()]
traj_n2 = [pos_n2.copy()]

# --- Visualización PyQtGraph ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Simulación Deuterón con Reid93Potential")
win.resize(900, 600)

# Potencial
plot_V = win.addPlot(title="Potencial Reid93 (1S0)")
curve_V = plot_V.plot(xp.asnumpy(r), xp.asnumpy(V), pen=pg.mkPen('g', width=2), name="V(r)")
plot_V.setLabel('left', 'V(r) [MeV]')
plot_V.setLabel('bottom', 'r [fm]')

# Fuerza
win.nextRow()
plot_F = win.addPlot(title="Fuerza Reid93 (1S0)")
curve_F = plot_F.plot(xp.asnumpy(r), xp.asnumpy(F), pen=pg.mkPen('m', width=2), name="F(r)")
plot_F.setLabel('left', 'F(r) [MeV/fm]')
plot_F.setLabel('bottom', 'r [fm]')

# Trayectorias
win.nextRow()
plot_traj = win.addPlot(title="Trayectorias nucleones (1D)")
curve_n1 = plot_traj.plot(pen=pg.mkPen('b', width=2), name="Nucleón 1")
curve_n2 = plot_traj.plot(pen=pg.mkPen('r', width=2), name="Nucleón 2")
point_n1 = plot_traj.plot([float(pos_n1[0])], [0], pen=None, symbol='o', symbolBrush='b', symbolSize=12)
point_n2 = plot_traj.plot([float(pos_n2[0])], [0], pen=None, symbol='o', symbolBrush='r', symbolSize=12)
traj_n1_y = [0]
traj_n2_y = [0]

# --- Gráfica de energía total ---
win.nextRow()
plot_E = win.addPlot(title="Energía total del sistema")
curve_E = plot_E.plot(pen=pg.mkPen('c', width=2), name="E_total")
plot_E.setLabel('left', 'Energía [MeV]')
plot_E.setLabel('bottom', 'Paso')

# --- Gráfica de distancia r12 ---
win.nextRow()
plot_r12 = win.addPlot(title="Distancia entre nucleones r12")
curve_r12 = plot_r12.plot(pen=pg.mkPen('y', width=2), name="r12")
plot_r12.setLabel('left', 'r12 [fm]')
plot_r12.setLabel('bottom', 'Paso')

# --- Historial de energías y distancias ---
hist_E = []
hist_r12 = []

# --- Utilidad para compatibilidad NumPy/CuPy ---
def to_numpy(arr):
    # Convierte lista de arrays (NumPy o CuPy) a NumPy plano, evitando conversiones implícitas prohibidas
    out = []
    for x in arr:
        if hasattr(x, 'get'):
            out.append(float(x.get()) if x.shape == () else x.get())
        else:
            out.append(float(x) if hasattr(x, 'shape') and x.shape == () else x)
    return np.array(out)

# --- Bucle de simulación ---
step_count = 0
max_steps = 300


def update():
    global step_count, pos_n1, pos_n2, vel_n1, vel_n2, traj_n1, traj_n2
    # Calcular distancia y fuerza
    r12 = xp.abs(pos_n2 - pos_n1)
    f = reid.force(r12)
    # Acción-reacción
    acc_n1 = f / mass_n
    acc_n2 = -f / mass_n
    # Integrar posiciones (Euler simple)
    vel_n1 += acc_n1 * dt
    vel_n2 += acc_n2 * dt
    pos_n1 += vel_n1 * dt
    pos_n2 += vel_n2 * dt
    traj_n1.append(pos_n1.copy())
    traj_n2.append(pos_n2.copy())
    # Actualizar visualización
    t1 = xp.asnumpy(xp.array(traj_n1).flatten())
    t2 = xp.asnumpy(xp.array(traj_n2).flatten())
    curve_n1.setData(t1, np.zeros_like(t1))
    curve_n2.setData(t2, np.zeros_like(t2))
    point_n1.setData([float(pos_n1[0])], [0])
    point_n2.setData([float(pos_n2[0])], [0])
    # Calcular energía cinética, potencial y total
    v1 = vel_n1[0]
    v2 = vel_n2[0]
    KE = 0.5 * mass_n * (v1**2 + v2**2)  # MeV (si v en fm/s y mass_n en MeV·fm⁻²·s²)
    PE = float(reid.potential(r12)[0])    # MeV
    E_total = KE + PE
    hist_E.append(E_total)
    hist_r12.append(float(r12[0]))
    # Actualizar gráficos de energía y distancia
    curve_E.setData(np.arange(len(hist_E)), to_numpy(hist_E))
    curve_r12.setData(np.arange(len(hist_r12)), to_numpy(hist_r12))
    # Imprimir valores clave cada 10 pasos
    if step_count % 10 == 0:
        print(f"Paso {step_count}: r12 = {float(r12[0]):.6e} fm, V = {PE:.6f} MeV, F = {float(f[0]):.6f} MeV/fm")
        print(f"  Nucleón 1: pos = {float(pos_n1[0]):.6e} fm, vel = {float(vel_n1[0]):.3e} fm/s, acc = {float(acc_n1[0]):.3e} fm/s²")
        print(f"  Nucleón 2: pos = {float(pos_n2[0]):.6e} fm, vel = {float(vel_n2[0]):.3e} fm/s, acc = {float(acc_n2[0]):.3e} fm/s²")
        print(f"  Energía cinética = {KE:.3e} MeV, Energía potencial = {PE:.3e} MeV, Energía total = {E_total:.3e} MeV")
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
