from __future__ import annotations

import argparse
import importlib
import os
import sys

from universe.config.simulation_config import CONFIG


def list_available_simulations(sim_dir: str = "universe/examples") -> list[str]:
    """
    Retorna una lista con los scripts disponibles en el directorio de simulaciones.

    Parameters
    ----------
    sim_dir : str
        Carpeta donde se almacenan los scripts de simulación.

    Returns
    -------
    list of str
        Lista de nombres de scripts sin extensión.
    """
    return sorted([
        f[:-3]
        for f in os.listdir(sim_dir)
        if f.endswith(".py") and not f.startswith("__")
    ])


def run_simulation(name: str) -> None:
    """
    Ejecuta una simulación desde el directorio 'examples/' por nombre.

    Parameters
    ----------
    name : str
        Nombre del script sin extensión.
    """
    print(f"[CLI] Backend activo: {CONFIG.backend_name.upper()}")
    sim_path = os.path.join("examples", f"{name}.py")

    if not os.path.exists(sim_path):
        print(f"[ERROR] No se encontró el script '{name}' en examples/")
        sys.exit(1)

    try:
        print(f"[CLI] Ejecutando script: {sim_path}")
        with open(sim_path, "r", encoding="utf-8") as f:
            code = compile(f.read(), sim_path, "exec")
            exec(code, {"__name__": "__main__"})
    except Exception as e:
        print(f"[ERROR] Falló la ejecución de '{name}': {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador físico del universo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Ejecutar una simulación")
    simulate_parser.add_argument("--name", required=True, help="Nombre del script en 'examples/'")

    subparsers.add_parser("list", help="Listar simulaciones disponibles")
    subparsers.add_parser("test", help="Ejecutar todos los tests del proyecto")

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args.name)
    elif args.command == "list":
        print("[CLI] Simulaciones disponibles:")
        for sim in list_available_simulations():
            print(f"  - {sim}")
    elif args.command == "test":
        print("[CLI] Ejecutando suite de tests...")
        os.system("pytest --cov=universe tests/")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
