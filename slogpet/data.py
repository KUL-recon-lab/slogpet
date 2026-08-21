"""Loading the published parameters from the bundled data files.

The numbers behind Table 2 and Figure 7 are data, not code: they live in
``slogpet/data/systems.json`` and ``slogpet/data/detectors.json``, and adding a
scanner needs no code change and no new release of the package.

JSON rather than YAML deliberately: it needs nothing outside the standard
library, which keeps the browser build small, and it forces the provenance to be
structured -- ``reference``, ``note`` and ``assumed`` are fields one can query,
not comments one can only read.

    from slogpet.data import load_systems, load_references, load_detector_groups

    for sc in load_systems():
        print(sc.name, sc.efficiency())
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import Scanner

__all__ = ["DATA_DIR", "load_systems", "load_references", "load_conventions",
           "load_detector_groups", "DetectorGroup", "load_json"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SYSTEMS_FILE = os.path.join(DATA_DIR, "systems.json")
DETECTORS_FILE = os.path.join(DATA_DIR, "detectors.json")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _scanner_from(d: dict, **override) -> Scanner:
    """One JSON record -> a Scanner.  ``null`` for ``L_mrd`` means unlimited."""
    kw = dict(
        name=d["name"], L_pet=d["L_pet"], D_pet=d["D_pet"],
        F_y=d.get("F_y"), F_z=d.get("F_z"),
        F_t=d.get("F_t"), ctr=d.get("ctr"),
        L_mrd=np.inf if d.get("L_mrd") is None else d["L_mrd"],
        epsilon=d.get("epsilon"), S_nema=d.get("S_nema"),
        crystal=d.get("crystal", ""), crystal_size=d.get("crystal_size"),
        energy_resolution=d.get("energy_resolution"),
        energy_window=d.get("energy_window"),
        reference=d.get("reference", ""),
        assumed=tuple(d.get("assumed", ())), note=d.get("note", ""),
    )
    kw.update(override)
    return Scanner(**kw)


def load_systems(path: str = SYSTEMS_FILE) -> List[Scanner]:
    """The published systems, in the order the table prints them."""
    return [_scanner_from(d) for d in load_json(path)["systems"]]


def load_references(path: str = SYSTEMS_FILE) -> Dict[str, str]:
    """Source of every number, keyed by the superscript used in the table."""
    return load_json(path)["references"]


def load_conventions(path: str = SYSTEMS_FILE) -> Dict[str, str]:
    """What each quantity means, e.g. how F_y is defined."""
    return load_json(path)["conventions"]


@dataclass(frozen=True)
class DetectorGroup:
    """One generic detector design and the axial lengths it is offered in.

    The design fixes everything except ``L_pet`` and the ring-difference limit,
    so ``configurations`` is a list of ``Scanner`` objects that differ only in
    those, paired with a plotting style.
    """
    id: str
    colour: str
    crystal: str
    crystal_size: Tuple[float, ...]
    epsilon: float
    F_t: Optional[float]
    F_y: float
    F_z: float
    D_pet: float
    based_on: Tuple[str, ...]
    configurations: Tuple[Tuple[Scanner, str], ...]

    @property
    def has_tof(self) -> bool:
        return self.F_t is not None

    @property
    def scanners(self) -> Tuple[Scanner, ...]:
        return tuple(sc for sc, _ in self.configurations)


def load_detector_groups(path: str = DETECTORS_FILE) -> List[DetectorGroup]:
    """The generic detector designs of Figure 7."""
    doc = load_json(path)
    out = []
    for g in doc["groups"]:
        cfgs = []
        for c in g["configurations"]:
            sc = Scanner(
                name=c.get("name", ""),
                L_pet=c["L_pet"], D_pet=g["D_pet"],
                F_y=g["F_y"], F_z=g["F_z"], F_t=g["F_t"],
                L_mrd=np.inf if c.get("L_mrd") is None else c["L_mrd"],
                epsilon=g["epsilon"], crystal=g["crystal"],
                crystal_size=tuple(g["crystal_size"]), note=c.get("note", ""))
            cfgs.append((sc, c.get("style", "long")))
        out.append(DetectorGroup(
            id=g["id"], colour=g["colour"], crystal=g["crystal"],
            crystal_size=tuple(g["crystal_size"]), epsilon=g["epsilon"],
            F_t=g["F_t"], F_y=g["F_y"], F_z=g["F_z"], D_pet=g["D_pet"],
            based_on=tuple(g.get("based_on", ())), configurations=tuple(cfgs)))
    return out


def load_styles(path: str = DETECTORS_FILE):
    """Plotting styles, as matplotlib dash specifications."""
    raw = load_json(path)["styles"]
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = v if isinstance(v, str) else (v[0], tuple(v[1]))
    return out


__all__.append("load_styles")


def verify(systems_file: str = SYSTEMS_FILE, detectors_file: str = DETECTORS_FILE):
    """Referential integrity of the data files.

    Catches the mistakes one actually makes when adding a system by hand: a
    reference number with no source, a detector design pointing at a system that
    is not in the table, a style that is never defined, a system with neither an
    efficiency nor a NEMA sensitivity.  Raises ``ValueError`` listing every
    problem found, so one edit round fixes them all.
    """
    problems = []
    systems = load_systems(systems_file)
    refs = load_references(systems_file)
    names = {sc.name for sc in systems}
    for sc in systems:
        if str(sc.reference) not in refs:
            problems.append(f"{sc.name}: reference {sc.reference!r} has no source")
        if sc.efficiency() is None:
            problems.append(f"{sc.name}: neither epsilon nor S_nema")
        for f in sc.assumed:
            if getattr(sc, f, "missing") == "missing":
                problems.append(f"{sc.name}: 'assumed' names unknown field {f!r}")
        if sc.F_y is None or sc.F_z is None:
            problems.append(f"{sc.name}: spatial resolution missing")
    styles = set(load_styles(detectors_file))
    for g in load_detector_groups(detectors_file):
        for name in g.based_on:
            if name not in names:
                problems.append(f"group {g.id}: based_on {name!r} is not a listed system")
        for sc, sty in g.configurations:
            if sty not in styles:
                problems.append(f"group {g.id}: style {sty!r} is not defined")
    if problems:
        raise ValueError("data files inconsistent:\n  " + "\n  ".join(problems))
    return len(systems), len(refs)


__all__.append("verify")
