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
from slogpet.resolution import C_OVER_2

__all__ = ["catalogue", "scanner_from", "summary", "sweep", "profile", "derived",
           "single_bed"]


def _number(v):
    """A form field the browser may have left empty -> a float or None.

    An empty field can reach here as None, as "" or -- when JavaScript passes
    its own null through Pyodide -- as a ``JsNull`` object, which is neither of
    those and raises on ``float()``.  All three mean the same thing.
    """
    if v is None or v == "":
        return None
    try:
        v = float(v)
    except TypeError:
        return None
    return None if v <= 0 else v


def _ctr_ps(sc):
    """Coincidence time resolution in ps, or None for a system without TOF."""
    if sc.ctr is not None:
        return float(sc.ctr)
    return None if sc.F_t is None else float(sc.F_t) / C_OVER_2


def _nema_of(sc):
    """The NEMA sensitivity: as published, or as implied by epsilon.

    Returns ``(cps_per_kBq, measured)`` so that the frontend can mark the ones it
    worked out itself.
    """
    if sc.S_nema is not None:
        return float(sc.S_nema), True
    eps = sc.efficiency()
    if eps is None:
        return None, False
    return 1000.0 * sc.S_ideal() * eps, False


def _plan(sc, task, scan_length, max_peak_to_trough):
    """The axial profile and the bed protocol, honouring a ripple limit.

    ``slogpet`` refuses a limit that no arrangement can meet, rather than
    quietly breaking it.  Here that refusal is data, not an error: the point is
    left out of the curve and the reason is passed on to the page.
    """
    prof = sp.axial_profile(sc, task.D_cyl, task.mu)
    try:
        protocol = sp.optimal_protocol(prof, float(scan_length),
                                       max_peak_to_trough=max_peak_to_trough)
    except ValueError as exc:
        return prof, None, str(exc)
    return prof, protocol, None


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


def summary(spec, F_o, D_cyl, S, max_peak_to_trough=None):
    """Derived quantities for one configuration at one scan length."""
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    nema, measured = _nema_of(sc)
    out = {
        "S_ideal": sc.S_ideal(),
        "epsilon": sc.efficiency(),
        "r": sc.r(task),
        "ctr": _ctr_ps(sc),
        "F_t": sc.F_t,
        "has_tof": sc.has_tof,
        "S_nema": nema,
        "S_nema_measured": measured,
    }
    prof, protocol, refused = _plan(sc, task, S, _number(max_peak_to_trough))
    if protocol is None:
        out["infeasible"] = refused
        return json.dumps(out)
    res = sp.snr2(sc, task, float(S), profile=prof, protocol=protocol)
    out.update({
        "n_beds": res.n_beds,
        "overlap": res.overlap,
        "spacing": res.protocol.spacing,
        "min_eta": res.protocol.min_eta,
        "mean_eta": res.protocol.mean_eta,
        "max_eta": res.protocol.max_eta,
        "peak_to_trough": res.protocol.coverage.peak_to_trough,
        "snr2_min": res.snr2_min,
    })
    return json.dumps(out)


def derived(spec):
    """What epsilon and the NEMA sensitivity imply about each other.

    The two are the same number in different clothes -- ``S_nema = 1000 x
    S_ideal x epsilon`` -- but ``S_ideal`` depends on the geometry, so the form
    cannot convert between them on its own.
    """
    sc = scanner_from(spec)
    nema, measured = _nema_of(sc)
    return json.dumps({"S_ideal": sc.S_ideal(), "epsilon": sc.efficiency(),
                       "S_nema": nema, "S_nema_measured": measured})


def sweep(spec, F_o, D_cyl, S_list, max_peak_to_trough=None):
    """SNR^2_min against scan length.  The axial profile is built once.

    A scan length at which the ripple limit cannot be met comes back as null
    rather than as an error, so that the rest of the curve still draws.
    """
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    limit = _number(max_peak_to_trough)
    Ss = [float(s) for s in S_list]
    prof = sp.axial_profile(sc, task.D_cyl, task.mu)
    out = {"S": Ss, "snr2": [], "n_beds": [], "overlap": []}
    for S in Ss:
        try:
            p = sp.optimal_protocol(prof, S, max_peak_to_trough=limit)
        except ValueError:
            out["snr2"].append(None)
            out["n_beds"].append(None)
            out["overlap"].append(None)
            continue
        out["snr2"].append(sp.snr2_value(sc.efficiency(), p.min_eta,
                                         sc.r(task), task.F_o))
        out["n_beds"].append(p.n_beds)
        out["overlap"].append(p.overlap)
    return json.dumps(out)


def single_bed(spec, F_o, D_cyl, nz=401):
    """``eta(z)`` for ONE bed position, with the two factors that multiply it.

    This is the profile every multi-bed acquisition is built from, and it does
    not depend on the scan length or on the protocol.  ``epsilon`` and ``r`` come
    back alongside rather than applied, so that the page can show what each
    contributes without asking for the curve again: the system part of the SNR^2
    at one bed position is ``epsilon x eta(z) x r x sigma_o^3``.
    """
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    prof = sp.axial_profile(sc, task.D_cyl, task.mu)
    reach = 0.55 * sc.L_pet                      # eta is zero beyond the detector
    z = np.linspace(-reach, reach, int(nz))
    return json.dumps({
        "z": [float(v) for v in z],
        "eta": [float(v) for v in prof.samples(z)],
        "epsilon": sc.efficiency(),
        "r": sc.r(task),
        "sigma_o3": (float(F_o) / sp.FWHM) ** 3,
        "eta_peak": float(prof.samples.values.max()),
    })


def profile(spec, F_o, D_cyl, S, nz=401, max_peak_to_trough=None):
    """The axial SNR^2 profile over the scan range, showing the bed structure."""
    sc = scanner_from(spec)
    task = sp.Task(float(F_o), float(D_cyl))
    prof, protocol, refused = _plan(sc, task, S, _number(max_peak_to_trough))
    if protocol is None:
        return json.dumps({"infeasible": refused, "z": [], "snr2": [],
                           "eta_N": [], "snr2_min": None, "n_beds": None})
    res = sp.snr2(sc, task, float(S), profile=prof, protocol=protocol,
                  nz=int(nz))
    return json.dumps({
        "z": [float(v) for v in res.z],
        "snr2": [float(v) for v in res.snr2],
        "eta_N": [float(v) for v in res.eta_N],
        "snr2_min": res.snr2_min,
        "n_beds": res.n_beds,
    })
