"""CLI principal del framework Universe."""

from __future__ import annotations

import argparse
import os
import sys
import subprocess

from universe.config.simulation_config import CONFIG
from universe.particles.mass_model import MassModel


def list_available_simulations(sim_dir: str = "examples") -> list[str]:
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
    if not os.path.exists(sim_dir):
        return []

    return sorted([
        f[:-3]
        for f in os.listdir(sim_dir)
        if f.endswith(".py") and not f.startswith("__")
    ])


def run_simulation(name: str, extra_args: list[str] = None) -> None:
    """
    Ejecuta una simulación desde el directorio 'examples/' por nombre, reenviando argumentos adicionales.

    Parameters
    ----------
    name : str
        Nombre del script sin extensión.
    extra_args : list[str], opcional
        Argumentos adicionales para el script de simulación.
    """
    print(f"[CLI] Backend activo: {CONFIG.backend_name.upper()}")
    sim_path = os.path.join("examples", f"{name}.py")

    if not os.path.exists(sim_path):
        print(f"[ERROR] No se encontró el script '{name}' en examples/")
        sys.exit(1)

    # Filtrar el primer '--' si está presente
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = [sys.executable, sim_path]
    if extra_args:
        cmd.extend(extra_args)
    try:
        print(f"[CLI] Ejecutando script: {sim_path} {' '.join(extra_args or [])}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[ERROR] Falló la ejecución de '{name}': {e}")
        sys.exit(1)


def handle_query_mass(args: argparse.Namespace) -> None:
    """
    Maneja la consulta de masas mediante el subcomando 'query mass'.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos del parser.
    """
    if args.list:
        _print_mass_table()
        return

    if args.particle:
        particle: str = args.particle.lower()
        if particle not in MassModel._mass_table:
            print(f"[ERROR] Partícula no reconocida: '{particle}'")
            return

        mass_mev: float = MassModel.get_mass(particle, "MeV")
        mass_kg: float = MassModel.get_mass(particle, "kg")

        print(f"[{particle.upper()}]")
        print(f"  Masa: {mass_mev:.8f} MeV/c²")
        print(f"  Masa: {mass_kg:.3e} kg")
    else:
        print("[ERROR] Debes especificar --particle <nombre> o --list")


def _print_mass_table() -> None:
    """
    Imprime una tabla con las masas de todas las partículas del Modelo Estándar.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("[ERROR] Falta la librería 'tabulate'. Instálala con: pip install tabulate")
        return

    headers: list[str] = ["Partícula", "Masa (MeV/c²)", "Masa (kg)"]
    rows: list[list[str]] = []

    for name, mass in MassModel._mass_table.items():
        rows.append([name, f"{mass.mass_MeV:.8f}", f"{mass.mass_kg:.3e}"])

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def main() -> None:
    """
    Punto de entrada principal del CLI del framework Universe.
    Uso:
      universe simulate --name <script> [-- <args extra para el script>]
    Ejemplo:
      universe simulate --name simulate_coupled_nucleons_qt -- --init colision
    """
    parser = argparse.ArgumentParser(description="Simulador físico del universo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # simulate
    simulate_parser = subparsers.add_parser("simulate", help="Ejecutar una simulación")
    simulate_parser.add_argument("--name", type=str, required=True, help="Nombre del script en 'examples/'")
    simulate_parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Argumentos adicionales para el script de simulación")

    # list
    subparsers.add_parser("list", help="Listar simulaciones disponibles")

    # test
    subparsers.add_parser("test", help="Ejecutar todos los tests del proyecto")

    # query
    query_parser = subparsers.add_parser("query", help="Consultar propiedades del universo")
    query_subparsers = query_parser.add_subparsers(dest="query_type", required=True)

    # query mass
    mass_parser = query_subparsers.add_parser("mass", help="Consultar masa de partículas")
    mass_parser.add_argument("--particle", type=str, help="Nombre de la partícula (ej. electron)")
    mass_parser.add_argument("--list", action="store_true", help="Listar todas las partículas disponibles")

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args.name, args.extra_args)
    elif args.command == "list":
        print("[CLI] Simulaciones disponibles:")
        for sim in list_available_simulations():
            print(f"  - {sim}")
    elif args.command == "test":
        print("[CLI] Ejecutando suite de tests...")
        os.system("pytest --cov=universe tests/")
    elif args.command == "query":
        if args.query_type == "mass":
            handle_query_mass(args)
        else:
            print(f"[ERROR] Subconsulta no reconocida: '{args.query_type}'")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
