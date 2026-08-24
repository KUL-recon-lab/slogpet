"""JSON adapter between the browser and the slogpet package.

The frontend must not contain physics, and the package must not contain user
interface.  This module is the only place the two meet: it turns JSON-friendly
dictionaries into ``Scanner`` and ``Task`` objects, calls the same functions the
paper calls, and hands back plain lists.

It runs unmodified in CPython, which is how it is tested.
"""
import json

import numpy as np

import slogpet as sp
from slogpet.data import (load_systems, load_detector_groups, load_references,
                          load_conventions)

__all__ = ["catalogue", "scanner_from", "summary", "sweep", "profile"]


def _family(crystal, size):
    """Systems sharing a crystal share a colour, as in Figure 7 of the paper."""
    if not crystal or not size:
        return "custom"
    return "%s %s" % (crystal, "x".join("%g" % v for v in size))


def _spec_of(sc, kind, label, family, extra=None):
    d = {
        "kind": kind, "label": label, "family": family,
        "name": sc.name, "L_pet": sc.L_pet, "D_pet": sc.D_pet,
        "L_mrd": None if not np.isfinite(sc.L_mrd) else sc.L_mrd,
        "F_y": sc.F_y, "F_z": sc.F_z, "F_t": sc.F_t, "ctr": sc.ctr,
        "epsilon": sc.efficiency(), "S_nema": sc.S_nema,
        "crystal": sc.crystal,
        "crystal_size": list(sc.crystal_size) if sc.crystal_size else None,
        "energy_resolution": sc.energy_resolution,
        "energy_window": list(sc.energy_window) if sc.energy_window else None,
        "reference": sc.reference, "assumed": list(sc.assumed), "note": sc.note,
        "S_ideal": sc.S_ideal(),
    }
    if extra:
        d.update(extra)
    return d


def catalogue():
    """Everything the picker offers, plus the sources behind it."""
    systems = []
    for sc in load_systems():
        mrd = "" if not np.isfinite(sc.L_mrd) else ", MRD %.0f cm" % (sc.L_mrd / 10)
        label = "%s (%.0f cm%s)" % (sc.name, sc.L_pet / 10, mrd)
        systems.append(_spec_of(sc, "system", label,
                                _family(sc.crystal, sc.crystal_size)))
    designs = []
    for g in load_detector_groups():
        for sc, style in g.configurations:
            mrd = "" if not np.isfinite(sc.L_mrd) else ", MRD %.0f cm" % (sc.L_mrd / 10)
            label = "%s -- %.0f cm%s" % (g.id, sc.L_pet / 10, mrd)
            designs.append(_spec_of(sc, "design", label,
                                    _family(g.crystal, g.crystal_size),
                                    {"group": g.id, "style": style,
                                     "based_on": list(g.based_on)}))
    return json.dumps({
        "systems": systems,
        "designs": designs,
        "references": load_references(),
        "conventions": load_conventions(),
        "families": sorted({s["family"] for s in systems + designs}),
    })


def scanner_from(spec):
    """A dictionary from the browser -> a Scanner.

    ``F_t`` wins over ``ctr`` and ``epsilon`` over ``S_nema``, exactly as in the
    package, so a custom system can be specified either way.
    """
    if isinstance(spec, str):
        spec = json.loads(spec)
    L_mrd = spec.get("L_mrd")
    return sp.Scanner(
        name=spec.get("label") or spec.get("name") or "custom",
        L_pet=float(spec["L_pet"]), D_pet=float(spec["D_pet"]),
        F_y=float(spec["F_y"]), F_z=float(spec["F_z"]),
        F_t=None if spec.get("F_t") in (None, "") else float(spec["F_t"]),
        ctr=None if spec.get("ctr") in (None, "") else float(spec["ctr"]),
        L_mrd=np.inf if L_mrd in (None, "") else float(L_mrd),
        epsilon=None if spec.get("epsilon") in (None, "") else float(spec["epsilon"]),
        S_nema=None if spec.get("S_nema") in (None, "") else float(spec["S_nema"]),
        crystal=spec.get("crystal", "") or "",
        crystal_size=tuple(spec["crystal_size"]) if spec.get("crystal_size") else None,
    )


def summary(spec, F_o, D_cyl, S):
    """Derived quantities for one configuration at one scan length."""
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    res = sp.snr2(sc, task, float(S))
    return json.dumps({
        "S_ideal": sc.S_ideal(),
        "epsilon": res.epsilon,
        "r": res.r,
        "n_beds": res.n_beds,
        "overlap": res.overlap,
        "spacing": res.protocol.spacing,
        "min_eta": res.protocol.min_eta,
        "mean_eta": res.protocol.mean_eta,
        "max_eta": res.protocol.max_eta,
        "snr2_min": res.snr2_min,
        "F_t": sc.F_t,
        "has_tof": sc.has_tof,
    })


def sweep(spec, F_o, D_cyl, S_list):
    """SNR^2_min against scan length.  The axial profile is built once."""
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    Ss = [float(s) for s in S_list]
    prof = sp.axial_profile(sc, task.D_cyl, task.mu)
    out = {"S": Ss, "snr2": [], "n_beds": [], "overlap": []}
    for S in Ss:
        p = sp.optimal_protocol(prof, S)
        out["snr2"].append(sp.snr2_value(sc.efficiency(), p.min_eta,
                                         sc.r(task), task.F_o))
        out["n_beds"].append(p.n_beds)
        out["overlap"].append(p.overlap)
    return json.dumps(out)


def profile(spec, F_o, D_cyl, S, nz=401):
    """The axial SNR^2 profile over the scan range, showing the bed structure."""
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    res = sp.snr2(sc, task, float(S), nz=int(nz))
    return json.dumps({
        "z": [float(v) for v in res.z],
        "snr2": [float(v) for v in res.snr2],
        "eta_N": [float(v) for v in res.eta_N],
        "snr2_min": res.snr2_min,
        "n_beds": res.n_beds,
    })
