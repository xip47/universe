"""Modelo de masas para partículas fundamentales del Modelo Estándar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Final


ParticleName = Literal[
    "electron", "muon", "tau",
    "up", "down", "charm", "strange", "top", "bottom",
    "photon", "gluon", "z_boson", "w_boson", "higgs",
    "neutrino_e", "neutrino_mu", "neutrino_tau"
]


@dataclass(frozen=True)
class ParticleMass:
    """
    Representación de la masa de una partícula fundamental.

    Attributes
    ----------
    name : ParticleName
        Nombre de la partícula.
    mass_MeV : float
        Masa en unidades de MeV/c².
    mass_kg : float
        Masa en kilogramos (kg).
    """
    name: ParticleName
    mass_MeV: float
    mass_kg: float


class MassModel:
    """
    Modelo centralizado para obtener masas de partículas del Modelo Estándar.
    """

    _mass_table: Final[dict[ParticleName, ParticleMass]] = {
        "electron": ParticleMass("electron", 0.51099895, 9.10938356e-31),
        "muon": ParticleMass("muon", 105.6583755, 1.8835316e-28),
        "tau": ParticleMass("tau", 1776.86, 3.16747e-27),

        "up": ParticleMass("up", 2.2, 3.92e-30),
        "down": ParticleMass("down", 4.7, 8.38e-30),
        "charm": ParticleMass("charm", 1270.0, 2.26e-27),
        "strange": ParticleMass("strange", 96.0, 1.71e-28),
        "top": ParticleMass("top", 172760.0, 3.09e-25),
        "bottom": ParticleMass("bottom", 4180.0, 7.49e-27),

        "photon": ParticleMass("photon", 0.0, 0.0),
        "gluon": ParticleMass("gluon", 0.0, 0.0),
        "z_boson": ParticleMass("z_boson", 91187.6, 1.63e-25),
        "w_boson": ParticleMass("w_boson", 80379.0, 1.43e-25),
        "higgs": ParticleMass("higgs", 125100.0, 2.23e-25),

        "neutrino_e": ParticleMass("neutrino_e", 1e-6, 1e-36),
        "neutrino_mu": ParticleMass("neutrino_mu", 2e-4, 3.6e-34),
        "neutrino_tau": ParticleMass("neutrino_tau", 0.0182, 3.24e-32),
    }

    @classmethod
    def get_mass(cls, particle: ParticleName, unit: Literal["MeV", "kg"] = "MeV") -> float:
        """
        Retorna la masa de una partícula en la unidad especificada.

        Parameters
        ----------
        particle : ParticleName
            Nombre de la partícula del Modelo Estándar.
        unit : {"MeV", "kg"}, optional
            Unidad deseada: MeV/c² o kilogramos (default = "MeV").

        Returns
        -------
        float
            Masa de la partícula en la unidad seleccionada.

        Raises
        ------
        KeyError
            Si la partícula no se encuentra definida en el modelo.
        """
        entry: ParticleMass = cls._mass_table[particle]
        return entry.mass_MeV if unit == "MeV" else entry.mass_kg
