from __future__ import annotations

from universe.config.simulation_config import CONFIG
from universe.hadrons.instances import HADRONS
from universe.numerics.backend import xp


def print_hadron_summary() -> None:
    """
    Imprime un resumen físico de los hadrones definidos, usando backend en GPU si está activado.

    Mostrará: nombre, símbolo, carga total, masa estimada (MeV/c²), generación máxima.
    """

    print(f"\n[Simulación] Backend numérico activo: {CONFIG.backend_name.upper()}")
    print("[Simulación] Listado de hadrones definidos:")
    print("-" * 70)
    print(f"{'Nombre':<12} | {'Símbolo':<6} | {'Carga':>6} | {'Masa [MeV/c²]':>14} | {'Generación':>10}")
    print("-" * 70)

    for h in HADRONS:
        charge = h.charge
        mass = h.mass_mev
        gen = h.generation

        print(f"{h.name:<12} | {h.symbol:<6} | {charge:6.2f} | {mass:14.2f} | {gen:10}")

    print("-" * 70)
    total_mass = xp.sum(xp.array([h.mass_mev for h in HADRONS]))
    print(f"[Simulación] Masa total estimada del conjunto: {float(total_mass):.2f} MeV/c²\n")


def main() -> None:
    """
    Función principal para ejecución desde la CLI oficial (`python -m universe simulate --name ...`)
    """
    print_hadron_summary()


# Exposición de main() como atributo del módulo
if __name__ == "__main__":
    main()
