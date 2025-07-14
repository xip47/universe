# universe/particles/mass_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Optional


ParticleName = Literal[
    "electron", "muon", "tau",
    "up", "down", "charm", "strange", "top", "bottom",
    "photon", "gluon", "z_boson", "w_boson", "higgs",
    "neutrino_e", "neutrino_mu", "neutrino_tau"
]

MassUnit = Literal["MeV", "eV", "kg"]


@dataclass(frozen=True)
class ParticleMass:
    """
    Representación de la masa de una partícula fundamental.

    Attributes
    ----------
    name : ParticleName
        Nombre de la partícula.
    mass_MeV : float
        Masa en MeV/c².
    mass_kg : float
        Masa en kilogramos.
    uncertainty_MeV : Optional[float]
        Incertidumbre en MeV/c², si está disponible.
    uncertainty_kg : Optional[float]
        Incertidumbre en kilogramos, si está disponible.
    """
    name: ParticleName
    mass_MeV: float
    mass_kg: float
    uncertainty_MeV: Optional[float] = None
    uncertainty_kg: Optional[float] = None


class MassModel:
    """
    Modelo centralizado para masas del Modelo Estándar, basado en PDG 2024.
    """

    _mass_table: Final[dict[ParticleName, ParticleMass]] = {
        "electron": ParticleMass("electron", 0.510998950, 9.1093837015e-31, 2.3e-11, 2.8e-41),
        "muon": ParticleMass("muon", 105.6583755, 1.883531627e-28, 0.0000023, 4.1e-36),
        "tau": ParticleMass("tau", 1776.86, 3.167540e-27, 0.12, 2.1e-29),

        "up": ParticleMass("up", 2.16, 3.85e-30, 0.49, 8.8e-31),
        "down": ParticleMass("down", 4.67, 8.32e-30, 0.48, 8.5e-31),
        "charm": ParticleMass("charm", 1270.0, 2.26e-27, 30.0, 5.3e-29),
        "strange": ParticleMass("strange", 93.4, 1.66e-28, 8.6, 1.5e-29),
        "top": ParticleMass("top", 172760.0, 3.08e-25, 300.0, 5.3e-28),
        "bottom": ParticleMass("bottom", 4180.0, 7.48e-27, 30.0, 5.3e-29),

        "photon": ParticleMass("photon", 0.0, 0.0),
        "gluon": ParticleMass("gluon", 0.0, 0.0),
        "z_boson": ParticleMass("z_boson", 91187.6, 1.62782e-25, 2.1, 3.7e-29),
        "w_boson": ParticleMass("w_boson", 80379.0, 1.435e-25, 12.0, 2.1e-28),
        "higgs": ParticleMass("higgs", 125090.0, 2.2275e-25, 24.0, 4.3e-28),

        "neutrino_e": ParticleMass("neutrino_e", 1e-6, 1.78266192e-39),       # upper bound < 0.8 eV
        "neutrino_mu": ParticleMass("neutrino_mu", 0.19, 3.38705765e-34),     # upper bound < 0.19 MeV
        "neutrino_tau": ParticleMass("neutrino_tau", 18.2, 3.2432447e-32),   # upper bound < 18.2 MeV
    }

    @classmethod
    def get_mass(cls, particle: ParticleName, unit: MassUnit = "MeV") -> float:
        """
        Retorna la masa de una partícula en la unidad deseada.

        Parameters
        ----------
        particle : ParticleName
            Nombre de la partícula (según el Modelo Estándar).
        unit : {"MeV", "eV", "kg"}, optional
            Unidad deseada de la masa. Default es "MeV".

        Returns
        -------
        float
            Masa en la unidad especificada.

        Raises
        ------
        KeyError
            Si el nombre de partícula no es válido.
        ValueError
            Si la unidad no es soportada.
        """
        entry = cls._mass_table[particle]
        if unit == "MeV":
            return entry.mass_MeV
        elif unit == "eV":
            return entry.mass_MeV * 1e6
        elif unit == "kg":
            return entry.mass_kg
        raise ValueError(f"Unidad de masa no soportada: {unit}")

    @classmethod
    def list_particles(cls) -> list[str]:
        """
        Lista todos los nombres de partículas disponibles.

        Returns
        -------
        list of str
            Nombres de partículas definidos en el modelo.
        """
        return list(cls._mass_table.keys())
