"""
Visualización de niveles de energía hidrogenoides usando Qt5.

Este script grafica los niveles de energía del átomo de hidrógeno y sus valores en eV.
"""

from __future__ import annotations

import sys
from typing import Any

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from universe.quantum_states.energy_levels import hydrogenic_energy


class EnergyLevelPlot(QWidget):
    """
    Widget que muestra los niveles de energía del hidrógeno.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("Niveles de energía del hidrógeno")
        self.ax.set_xlabel("n")
        self.ax.set_ylabel("Energía [eV]")
        self.ax.invert_yaxis()

        self.plot_levels()

    def plot_levels(self) -> None:
        """
        Dibuja los niveles energéticos para n = 1 a 7.
        """
        n_max: int = 7
        Z: int = 1

        for n in range(1, n_max + 1):
            energy = hydrogenic_energy(n=n, Z=Z)
            print(f"[Simulación] n = {n}, Energía = {energy:.6f} eV")

            # Dibujar la línea de nivel
            self.ax.hlines(energy, xmin=0.1, xmax=0.9, color="blue", linewidth=2)

            # Colocar texto alineado correctamente
            align = "bottom" if n % 2 == 0 else "top"
            self.ax.text(
                0.95,
                energy,
                f"n = {n}",
                va=align,
                fontsize=9,
                ha="left",
                color="black"
            )

        self.ax.set_xlim(0, 1.3)
        self.ax.grid(True)


class MainWindow(QMainWindow):
    """
    Ventana principal que contiene el gráfico.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Visualización de niveles energéticos")
        self.setCentralWidget(EnergyLevelPlot())


def main() -> None:
    """
    Punto de entrada para lanzar la interfaz Qt5.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
