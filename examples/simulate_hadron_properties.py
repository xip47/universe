from universe.config.simulation_config import CONFIG
from universe.hadrons.instances import HADRONS
from universe.numerics.backend import xp


def print_hadron_summary() -> None:
    """
    Imprime un resumen físico de los hadrones definidos.
    """
    print(f"\n[Simulación] Backend numérico activo: {CONFIG.backend_name.upper()}")
    print("[Simulación] Listado de hadrones definidos:")
    print("-" * 70)
    print(f"{'Nombre':<12} | {'Símbolo':<6} | {'Carga':>6} | {'Masa [MeV/c²]':>14} | {'Generación':>10}")
    print("-" * 70)

    for h in HADRONS:
        print(f"{h.name:<12} | {h.symbol:<6} | {h.charge:6.2f} | {h.mass_mev:14.2f} | {h.generation:10}")

    print("-" * 70)
    total_mass = xp.sum(xp.array([h.mass_mev for h in HADRONS]))
    print(f"[Simulación] Masa total estimada del conjunto: {float(total_mass):.2f} MeV/c²\n")


def main() -> None:
    """
    Punto de entrada principal para la simulación.
    """
    print_hadron_summary()


# ✅ Exposición explícita para el CLI (importlib)
main = main

# ✅ También ejecutable directamente
if __name__ == "__main__":
    main()
