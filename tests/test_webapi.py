"""The browser adapter, exercised in CPython.

``web/api.py`` is the only code the frontend calls.  Testing it here means the
page cannot drift from the package without a test failing, and it is the reason
the numbers in the browser can be trusted: they come through this.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))

import api                                                    # noqa: E402
import slogpet as sp                                          # noqa: E402


@pytest.fixture(scope="module")
def cat():
    return json.loads(api.catalogue())


def test_the_catalogue_offers_everything(cat):
    assert len(cat["systems"]) == 10
    assert len(cat["designs"]) == 10
    assert len(cat["families"]) == 4          # four crystals, four colours
    assert len(cat["references"]) >= 8
    assert cat["conventions"]["F_t"].startswith("F_t")


def test_every_entry_is_json_serialisable_and_complete(cat):
    for sp_ in cat["systems"] + cat["designs"]:
        json.dumps(sp_)                       # no numpy scalars, no inf
        for k in ("label", "family", "L_pet", "D_pet", "F_y", "F_z", "epsilon"):
            assert sp_[k] is not None, (sp_["label"], k)
        assert sp_["L_mrd"] is None or np.isfinite(sp_["L_mrd"])


def test_infinity_survives_the_round_trip(cat):
    """JSON has no infinity, so an unlimited ring difference travels as null and
    must come back as inf -- otherwise every unrestricted system would silently
    acquire a ring-difference limit of zero."""
    unlimited = next(s for s in cat["systems"] if s["L_mrd"] is None)
    assert np.isinf(api.scanner_from(unlimited).L_mrd)
    limited = next(s for s in cat["systems"] if s["L_mrd"] is not None)
    assert api.scanner_from(limited).L_mrd == 282.0


def test_a_round_trip_reproduces_the_package_object(cat):
    from slogpet.data import load_systems
    for spec, orig in zip(cat["systems"], load_systems()):
        sc = api.scanner_from(spec)
        assert sc.L_pet == orig.L_pet and sc.D_pet == orig.D_pet
        assert sc.F_y == orig.F_y and sc.F_z == orig.F_z
        assert sc.F_t == orig.F_t
        assert sc.efficiency() == pytest.approx(orig.efficiency(), rel=1e-15)
        assert (sc.r(sp.Task(5.0, 300.0))
                == pytest.approx(orig.r(sp.Task(5.0, 300.0)), rel=1e-15))


def test_a_custom_specification_works_without_any_metadata():
    """What the browser form sends: geometry, resolutions, efficiency, nothing else."""
    spec = {"label": "mine", "L_pet": 2000.0, "D_pet": 820.0, "F_y": 3.4, "F_z": 3.8,
            "ctr": 230.0, "F_t": None, "L_mrd": None, "epsilon": 0.276, "S_nema": None}
    sc = api.scanner_from(spec)
    assert sc.F_t == pytest.approx(34.5, rel=1e-15)
    assert sc.efficiency() == 0.276
    assert np.isinf(sc.L_mrd)
    assert json.loads(api.summary(spec, 5.0, 300.0, 1000.0))["n_beds"] == 1


def test_blank_fields_are_treated_as_absent():
    spec = {"label": "no tof", "L_pet": 320.0, "D_pet": 725.0, "F_y": 4.3, "F_z": 4.4,
            "ctr": "", "F_t": "", "L_mrd": "", "epsilon": 0.5, "S_nema": ""}
    sc = api.scanner_from(spec)
    assert not sc.has_tof and np.isinf(sc.L_mrd)
    out = json.loads(api.summary(spec, 5.0, 300.0, 700.0))
    assert out["has_tof"] is False and out["F_t"] is None


def test_the_adapter_agrees_with_the_package(cat):
    """The point of the whole exercise: the browser gets the paper's numbers."""
    for spec in cat["systems"][:4] + cat["designs"][:4]:
        sc = api.scanner_from(spec)
        for F_o, D_cyl, S in ((5.0, 300.0, 1000.0), (10.0, 200.0, 1800.0)):
            task = sp.Task(F_o, D_cyl)
            want = sp.snr2(sc, task, S)
            got = json.loads(api.summary(spec, F_o, D_cyl, S))
            assert got["snr2_min"] == pytest.approx(want.snr2_min, rel=1e-15)
            assert got["r"] == pytest.approx(want.r, rel=1e-15)
            assert got["n_beds"] == want.n_beds


def test_a_sweep_is_the_same_as_the_points_it_contains(cat):
    spec = cat["systems"][2]
    Ss = [200.0, 700.0, 1500.0]
    swept = json.loads(api.sweep(spec, 5.0, 300.0, Ss))
    for S, v in zip(Ss, swept["snr2"]):
        assert v == pytest.approx(
            json.loads(api.summary(spec, 5.0, 300.0, S))["snr2_min"], rel=1e-15)


def test_the_axial_profile_bottoms_out_at_the_reported_minimum(cat):
    spec = cat["designs"][1]
    S = 1400.0
    prof = json.loads(api.profile(spec, 5.0, 300.0, S))
    assert len(prof["z"]) == len(prof["snr2"]) == len(prof["eta_N"])
    assert min(prof["z"]) == pytest.approx(-S / 2) and max(prof["z"]) == pytest.approx(S / 2)
    # the optimiser searches 241 points across the scan range, so its reported
    # minimum can sit slightly above the true one -- never below, and by under
    # 1 per cent across every configuration in the data files
    assert min(prof["snr2"]) <= prof["snr2_min"] * (1 + 1e-9)
    assert min(prof["snr2"]) == pytest.approx(prof["snr2_min"], rel=1.5e-2)


def test_families_group_by_crystal(cat):
    """The frontend colours by family, so a family must mean one detector."""
    from collections import defaultdict
    fam = defaultdict(set)
    for s in cat["systems"] + cat["designs"]:
        fam[s["family"]].add((s["crystal"], tuple(s["crystal_size"] or ())))
    for f, members in fam.items():
        assert len(members) == 1, (f, members)
