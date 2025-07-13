from __future__ import annotations

from typing import Final

from universe.particles.constants import BOSONS as B_DATA
from universe.particles.constants import LEPTONS as L_DATA
from universe.particles.constants import QUARKS as Q_DATA
from universe.particles.definitions import Boson
from universe.particles.definitions import Lepton
from universe.particles.definitions import Quark


# ----------------------------
# Instanciación de Quarks
# ----------------------------
QUARKS: Final[list[Quark]] = [
    Quark(
        name=q["name"],
        symbol=q["symbol"],
        mass_mev=q["mass_mev"],
        charge=q["charge"],
        spin=q["spin"],
        generation=q["generation"]
    ) for q in Q_DATA.values()
]


# ----------------------------
# Instanciación de Leptones
# ----------------------------
LEPTONS: Final[list[Lepton]] = [
    Lepton(
        name=l["name"],
        symbol=l["symbol"],
        mass_mev=l["mass_mev"],
        charge=l["charge"],
        spin=l["spin"],
        generation=l["generation"],
        lepton_number=l["lepton_number"]
    ) for l in L_DATA.values()
]


# ----------------------------
# Instanciación de Bosones
# ----------------------------
BOSONS: Final[list[Boson]] = [
    Boson(
        name=b["name"],
        symbol=b["symbol"],
        mass_mev=b["mass_mev"],
        charge=b["charge"],
        spin=b["spin"],
        generation=b["generation"],
        force_carrier=b["force_carrier"]
    ) for b in B_DATA.values()
]


# ----------------------------
# Agrupación del Modelo Estándar
# ----------------------------
STANDARD_MODEL: Final[dict[str, list]] = {
    "quarks": QUARKS,
    "leptons": LEPTONS,
    "bosons": BOSONS
}
