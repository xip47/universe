"""
Simulación visual acoplada de tres nucleones en 2D bajo fuerzas nucleares (Reid93) y de Coulomb.
Cumple estándares: NumPy doc español, tipado Python 3, imports individuales, GPU (xp), arquitectura modular.
"""

import sys
import logging
from typing import List
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication
from universe.numerics.backend import xp
from universe.systems.coupled_system import CoupledSystem
from universe.particles.extended_definitions import BaseParticleExtended, DynamicParticleExtended
from universe.particles.degrees_of_freedom import Spin, Isospin, Charge, QuantumState
from universe.physics.nuclear.potentials import Reid93Potential


def crear_nucleon(nombre: str, masa: float, carga: float, spin_val: float, spin_vec, isospin_val: float, isospin_vec, qs) -> DynamicParticleExtended:
    """
    Crea un nucleón extendido con todos los grados de libertad.

    Parameters
    ----------
    nombre : str
        Nombre del nucleón.
    masa : float
        Masa en MeV/c².
    carga : float
        Carga eléctrica (e).
    spin_val : float
        Valor del spin.
    spin_vec : List[float]
        Vector de spin (3D).
    isospin_val : float
        Valor del isospin.
    isospin_vec : List[float]
        Vector de isospin (3D).
    qs : QuantumState
        Estado cuántico.

    Returns
    -------
    DynamicParticleExtended
        Nucleón extendido listo para simulación.
    """
    spin = Spin(spin_val, xp.array(spin_vec, dtype=xp.float64))
    isospin = Isospin(isospin_val, xp.array(isospin_vec, dtype=xp.float64))
    charge = Charge(carga)
    base = BaseParticleExtended(nombre, masa, charge, spin, isospin, qs)
    # Posición y velocidad se asignan fuera
    return base


def condiciones_iniciales(tipo: str = "triangular"):
    """
    Devuelve posiciones y velocidades iniciales para diferentes escenarios.

    Parameters
    ----------
    tipo : str
        Tipo de condición inicial: "colision", "triangular", "lineal".
        En 'colision' se rompe la simetría para forzar dinámica.

    Returns
    -------
    posiciones : List[xp.ndarray]
        Lista de posiciones iniciales (fm).
    velocidades : List[xp.ndarray]
        Lista de velocidades iniciales (fm/c).
    """
    if tipo == "colision":
        # Dos nucleones se aproximan a un tercero ligeramente desplazado y con velocidad
        posiciones = [xp.array([-1.0, 0.0]), xp.array([1.0, 0.0]), xp.array([0.1, 0.2])]
        velocidades = [xp.array([0.02, 0.0]), xp.array([-0.01, 0.0]), xp.array([0.0, 0.01])]
    elif tipo == "lineal":
        posiciones = [xp.array([-1.0, 0.0]), xp.array([0.0, 0.0]), xp.array([1.0, 0.0])]
        velocidades = [xp.array([0.01, 0.0]), xp.array([0.0, 0.0]), xp.array([-0.01, 0.0])]
    else:  # "triangular" por defecto
        posiciones = [xp.array([0.0, 0.0]), xp.array([2.0, 0.0]), xp.array([1.0, 1.5])]
        velocidades = [xp.array([0.0, 0.01]), xp.array([0.0, -0.01]), xp.array([0.01, 0.0])]
    return posiciones, velocidades


def main() -> None:
    """
    Simulación visual acoplada de tres nucleones en 2D con logs físicos.
    Permite elegir condiciones iniciales: colisión, triangular, lineal.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Simulación acoplada de nucleones en 2D")
    parser.add_argument('--init', type=str, default='triangular', choices=['colision', 'triangular', 'lineal'], help='Condición inicial')
    args = parser.parse_args()
    # Configuración de logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("simulacion")

    # --- Definición de nucleones extendidos ---
    nucleones = [
        crear_nucleon(
            nombre="proton1", masa=938.27, carga=1.0,
            spin_val=0.5, spin_vec=[0, 0, 1],
            isospin_val=0.5, isospin_vec=[1, 0, 0],
            qs=QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
        ),
        crear_nucleon(
            nombre="proton2", masa=938.27, carga=1.0,
            spin_val=-0.5, spin_vec=[0, 0, -1],
            isospin_val=-0.5, isospin_vec=[-1, 0, 0],
            qs=QuantumState(1, 0, 0, -0.5, 0.5, -0.5)
        ),
        crear_nucleon(
            nombre="neutron", masa=939.57, carga=0.0,
            spin_val=0.5, spin_vec=[0, 1, 0],
            isospin_val=0.5, isospin_vec=[0, 1, 0],
            qs=QuantumState(1, 0, 0, 0.5, 0.5, 0.5)
        )
    ]
    logger.info(f"Condición inicial seleccionada: {args.init}")
    posiciones, velocidades = condiciones_iniciales(args.init)
    particulas = [
        DynamicParticleExtended(
            static=nucleones[i],
            position=posiciones[i],
            velocity=velocidades[i],
            spin=nucleones[i].spin,
            isospin=nucleones[i].isospin,
            quantum_state=nucleones[i].quantum_state
        ) for i in range(3)
    ]

    # --- Sistema acoplado ---
    sistema = CoupledSystem(
        particles=particulas,
        nuclear_potential=Reid93Potential(channel="1S0"),
        use_coulomb=True
    )

    # --- Visualización con PyQtGraph ---
    app = QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Simulación acoplada: Tres nucleones (2D)")
    plot = win.addPlot(title="Trayectorias de nucleones (fm)")
    plot.setXRange(-2, 4)
    plot.setYRange(-2, 4)
    colores = [(200, 0, 0), (0, 0, 200), (0, 150, 0)]
    curvas = [plot.plot(pen=pg.mkPen(color=colores[i], width=2)) for i in range(3)]
    puntos = [plot.plot(symbol='o', symbolBrush=colores[i], symbolSize=14) for i in range(3)]
    trayectorias = [[], [], []]

    # --- Logs de energía y distancias ---
    energia_log = []
    dist_log = []
    steps = 0
    dt = 1e-24
    max_steps = 2000

    def update():
        nonlocal steps
        sistema.step(dt)
        for i, p in enumerate(sistema.particles):
            trayectorias[i].append(np.array(xp.asnumpy(p.position)))
        for i in range(3):
            arr = np.array(trayectorias[i])
            if arr.shape[0] > 1:
                curvas[i].setData(arr[:, 0], arr[:, 1])
                puntos[i].setData([arr[-1, 0]], [arr[-1, 1]])
        energia = sistema.total_energy()
        energia_log.append(energia)
        dist12 = float(xp.linalg.norm(sistema.particles[0].position - sistema.particles[1].position))
        dist13 = float(xp.linalg.norm(sistema.particles[0].position - sistema.particles[2].position))
        dist23 = float(xp.linalg.norm(sistema.particles[1].position - sistema.particles[2].position))
        dist_log.append((dist12, dist13, dist23))
        if steps % 100 == 0:
            logger.info(f"Paso {steps}: Energía total = {energia:.4f} MeV, d12 = {dist12:.3f} fm, d13 = {dist13:.3f} fm, d23 = {dist23:.3f} fm")
            for i, p in enumerate(sistema.particles):
                pos = xp.asnumpy(p.position)
                vel = xp.asnumpy(p.velocity)
                logger.info(f"  Nucleón {i+1}: pos = ({pos[0]:.4f}, {pos[1]:.4f}) fm, vel = ({vel[0]:.3e}, {vel[1]:.3e}) fm/c, spin = {p.spin.value}, isospin = {p.isospin.value}")
            # Logs de fuerzas y aceleraciones
            fuerzas = sistema.compute_forces()
            for i, f in enumerate(fuerzas):
                a = f / sistema.particles[i].static.mass_mev
                logger.info(f"    Fuerza neta Nucleón {i+1}: {xp.asnumpy(f)} MeV/fm, aceleración: {xp.asnumpy(a)} 1/fm")
        steps += 1
        if steps > max_steps:
            timer.stop()
            logger.info("Simulación finalizada.")
            app.quit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)
    app.exec_()


if __name__ == "__main__":
    main()
