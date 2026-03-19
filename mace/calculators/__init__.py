"""Lazy calculator exports.

Keep package import light so embedded runtimes that only need the LAMMPS
wrappers do not eagerly import foundation-model helpers and their optional
training/image dependencies.
"""

from importlib import import_module

__all__ = [
    "MACECalculator",
    "LAMMPS_MACE",
    "mace_mp",
    "mace_off",
    "mace_anicc",
    "mace_omol",
    "mace_polar",
]


def __getattr__(name):
    if name == "MACECalculator":
        return import_module(".mace", __name__).MACECalculator
    if name == "LAMMPS_MACE":
        return import_module(".lammps_mace", __name__).LAMMPS_MACE
    if name in {"mace_mp", "mace_off", "mace_anicc", "mace_omol", "mace_polar"}:
        return getattr(import_module(".foundations_models", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
