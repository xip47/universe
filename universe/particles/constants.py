from typing import Final, Literal

# Tipado general
ParticleDict = dict[str, dict[str, float | str | int | bool]]

# ---------------------------
# QUARKS: (color_charge = True)
# ---------------------------
QUARKS: Final[ParticleDict] = {
    "up": {
        "name": "up quark",
        "symbol": "u",
        "mass_mev": 2.16,
        "charge": +2/3,
        "spin": 0.5,
        "generation": 1
    },
    "down": {
        "name": "down quark",
        "symbol": "d",
        "mass_mev": 4.67,
        "charge": -1/3,
        "spin": 0.5,
        "generation": 1
    },
    "charm": {
        "name": "charm quark",
        "symbol": "c",
        "mass_mev": 1270.0,
        "charge": +2/3,
        "spin": 0.5,
        "generation": 2
    },
    "strange": {
        "name": "strange quark",
        "symbol": "s",
        "mass_mev": 93.0,
        "charge": -1/3,
        "spin": 0.5,
        "generation": 2
    },
    "top": {
        "name": "top quark",
        "symbol": "t",
        "mass_mev": 172760.0,
        "charge": +2/3,
        "spin": 0.5,
        "generation": 3
    },
    "bottom": {
        "name": "bottom quark",
        "symbol": "b",
        "mass_mev": 4180.0,
        "charge": -1/3,
        "spin": 0.5,
        "generation": 3
    },
}

# ---------------------------
# LEPTONES: (lepton_number = ±1)
# ---------------------------
LEPTONS: Final[ParticleDict] = {
    "electron": {
        "name": "electron",
        "symbol": "e⁻",
        "mass_mev": 0.51099895,
        "charge": -1.0,
        "spin": 0.5,
        "generation": 1,
        "lepton_number": +1
    },
    "electron_neutrino": {
        "name": "electron neutrino",
        "symbol": "νₑ",
        "mass_mev": 0.0000022,  # upper limit < 2.2 eV
        "charge": 0.0,
        "spin": 0.5,
        "generation": 1,
        "lepton_number": +1
    },
    "muon": {
        "name": "muon",
        "symbol": "μ⁻",
        "mass_mev": 105.6583755,
        "charge": -1.0,
        "spin": 0.5,
        "generation": 2,
        "lepton_number": +1
    },
    "muon_neutrino": {
        "name": "muon neutrino",
        "symbol": "ν_μ",
        "mass_mev": 0.00017,  # upper limit
        "charge": 0.0,
        "spin": 0.5,
        "generation": 2,
        "lepton_number": +1
    },
    "tau": {
        "name": "tau",
        "symbol": "τ⁻",
        "mass_mev": 1776.86,
        "charge": -1.0,
        "spin": 0.5,
        "generation": 3,
        "lepton_number": +1
    },
    "tau_neutrino": {
        "name": "tau neutrino",
        "symbol": "ν_τ",
        "mass_mev": 0.00018,  # upper limit
        "charge": 0.0,
        "spin": 0.5,
        "generation": 3,
        "lepton_number": +1
    },
}

# ---------------------------
# BOSONES: (mediadores de fuerza)
# ---------------------------
BOSONS: Final[ParticleDict] = {
    "photon": {
        "name": "photon",
        "symbol": "γ",
        "mass_mev": 0.0,
        "charge": 0.0,
        "spin": 1.0,
        "generation": 0,
        "force_carrier": "electroweak"
    },
    "gluon": {
        "name": "gluon",
        "symbol": "g",
        "mass_mev": 0.0,
        "charge": 0.0,
        "spin": 1.0,
        "generation": 0,
        "force_carrier": "strong"
    },
    "z_boson": {
        "name": "Z boson",
        "symbol": "Z⁰",
        "mass_mev": 91187.6,
        "charge": 0.0,
        "spin": 1.0,
        "generation": 0,
        "force_carrier": "electroweak"
    },
    "w_boson": {
        "name": "W boson",
        "symbol": "W±",
        "mass_mev": 80379.0,
        "charge": +1.0,  # ±1.0, simplificamos como W⁺
        "spin": 1.0,
        "generation": 0,
        "force_carrier": "electroweak"
    },
    "higgs": {
        "name": "Higgs boson",
        "symbol": "H⁰",
        "mass_mev": 125090.0,
        "charge": 0.0,
        "spin": 0.0,
        "generation": 0,
        "force_carrier": "electroweak"
    },
    "graviton": {
        "name": "graviton",
        "symbol": "G",
        "mass_mev": 0.0,
        "charge": 0.0,
        "spin": 2.0,
        "generation": 0,
        "force_carrier": "gravitational"
    }
}
