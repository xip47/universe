from __future__ import annotations

from typing import Final

from universe.hadrons.composite import Hadron
from universe.particles.standard_model import QUARKS


# Helper para acceder por símbolo
QUARK_DICT: Final[dict[str, object]] = {q.symbol: q for q in QUARKS}

# Obtener quarks necesarios
u = QUARK_DICT["u"]     # up
d = QUARK_DICT["d"]     # down


# -------------------------------
# BARIONES ESTABLES
# -------------------------------

proton: Final[Hadron] = Hadron(
    name="proton",
    symbol="p⁺",
    quarks=(u, u, d),
    type="baryon"
)

neutron: Final[Hadron] = Hadron(
    name="neutron",
    symbol="n⁰",
    quarks=(u, d, d),
    type="baryon"
)


# -------------------------------
# CONJUNTO EXPORTABLE
# -------------------------------

HADRONS: Final[list[Hadron]] = [
    proton,
    neutron
]
