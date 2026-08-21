"""Pinned values: the numbers the paper prints.

Unlike the rest of the suite these assert nothing about correctness -- they
assert only that the answers have not moved.  A failure here is not necessarily
a bug, but it is always something that has to be looked at and explained before
`tests/golden.json` is regenerated.

The tolerance is 1e-10 relative: pure refactoring passes, and anything that
actually changes a number does not.
"""
import json
import os

import pytest

import slogpet as sp
from slogpet.data import load_systems, load_detector_groups

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")
RTOL = 1e-10

with open(GOLDEN) as _fh:
    G = json.load(_fh)


def _key(sc):
    return f"{sc.name} L={sc.L_pet:.0f} MRD={sc.L_mrd:.0f}"


@pytest.mark.parametrize("sc", load_systems(), ids=_key)
def test_published_systems(sc):
    ref = G["systems"][_key(sc)]
    assert sc.S_ideal() == pytest.approx(ref["S_ideal"], rel=RTOL)
    assert sc.efficiency() == pytest.approx(ref["epsilon"], rel=RTOL)
    for F_o, want in ref["r"].items():
        assert sc.r(sp.Task(float(F_o), 300.0)) == pytest.approx(want, rel=RTOL)


@pytest.mark.parametrize("g", load_detector_groups(), ids=lambda g: g.id)
def test_generic_designs(g):
    for key, want in G["designs"][g.id].items():
        _, fo, d = key.split("_")
        got = g.scanners[0].r(sp.Task(float(fo[2:]), float(d[1:])))
        assert got == pytest.approx(want, rel=RTOL)


@pytest.mark.parametrize("key", sorted(G["protocols"]))
def test_protocols(key):
    L_pet, D_cyl, S = (float(part.split("=")[1]) for part in key.split())
    sc = sp.Scanner("generic", L_pet, 740.0, 3.4, 3.8, F_t=32.0, epsilon=1.0)
    p = sp.optimal_protocol(sp.axial_profile(sc, D_cyl), S)
    ref = G["protocols"][key]
    assert p.n_beds == ref["n_beds"]
    assert p.spacing == pytest.approx(ref["spacing"], rel=RTOL, abs=1e-12)
    assert p.min_eta == pytest.approx(ref["min_eta"], rel=RTOL)


def _snr_cases():
    cases = []
    for g in load_detector_groups():
        for sc, _ in g.configurations:
            for F_o in (5.0, 10.0):
                for D_cyl in (200.0, 300.0):
                    for S in (200.0, 500.0, 1000.0, 1800.0):
                        cases.append((g.id, sc, F_o, D_cyl, S))
    return cases


@pytest.mark.parametrize("gid,sc,F_o,D_cyl,S", _snr_cases(),
                         ids=lambda x: str(x) if not hasattr(x, "L_pet") else
                         f"L{x.L_pet:.0f}")
def test_snr2_of_every_configuration(gid, sc, F_o, D_cyl, S):
    key = (f"{gid} L={sc.L_pet:.0f} MRD={sc.L_mrd:.0f} "
           f"Fo={F_o:g} D={D_cyl:g} S={S:g}")
    got = sp.snr2(sc, sp.Task(F_o, D_cyl), S).snr2_min
    assert got == pytest.approx(G["snr2"][key], rel=RTOL)


def test_the_golden_file_covers_everything_the_paper_prints():
    assert len(G["systems"]) == 10
    assert len(G["designs"]) == 4
    assert len(G["snr2"]) == 10 * 2 * 2 * 4
