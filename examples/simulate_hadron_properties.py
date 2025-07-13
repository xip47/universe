from universe.config.simulation_config import CONFIG
from universe.hadrons.instances import HADRONS
from universe.numerics.backend import xp


def print_hadron_summary() -> None:
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
    print_hadron_summary()


# ✅ Exponer main() explícitamente para importlib
main = main

# También ejecutable como script independiente
if __name__ == "__main__":
    main()
