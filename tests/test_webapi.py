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


# ---------------------------------------------------- the ripple limit, and
# sensitivity given as a NEMA figure instead of an efficiency
def test_a_ripple_limit_changes_the_protocol_not_the_physics(cat):
    """The limit reaches the browser through the same argument the paper uses."""
    quadra = next(s for s in cat["systems"] if s["name"] == "Biograph Vision Quadra")
    free = json.loads(api.summary(quadra, 5.0, 200.0, 1000.0))
    held = json.loads(api.summary(quadra, 5.0, 200.0, 1000.0, 1.1))
    assert free["peak_to_trough"] > 1.1                  # the limit binds
    assert held["peak_to_trough"] <= 1.1 + 1e-12
    assert held["n_beds"] > free["n_beds"]               # paid for in beds
    assert held["snr2_min"] < free["snr2_min"]           # and in sensitivity
    for k in ("epsilon", "r", "S_nema", "ctr"):          # the system is unchanged
        assert held[k] == free[k]


def test_an_unmeetable_limit_is_reported_not_raised(cat):
    """slogpet refuses such a limit with an exception.  The page must survive it,
    so the adapter turns the refusal into data: nulls in the sweep, a message in
    the summary."""
    quadra = next(s for s in cat["systems"] if s["name"] == "Biograph Vision Quadra")
    swept = json.loads(api.sweep(quadra, 5.0, 200.0, [500.0, 1000.0], 1.001))
    assert swept["snr2"] == [None, None]
    assert swept["n_beds"] == [None, None]
    told = json.loads(api.summary(quadra, 5.0, 200.0, 1000.0, 1.001))
    assert "max/min" in told["infeasible"]
    assert "snr2_min" not in told
    assert told["epsilon"] is not None                   # still describes the system
    drawn = json.loads(api.profile(quadra, 5.0, 200.0, 1000.0, 401, 1.001))
    assert drawn["z"] == [] and drawn["snr2_min"] is None


def test_no_limit_is_the_default_in_every_entry_point(cat):
    """Blank, absent and zero all mean the same thing: no limit."""
    one = cat["systems"][0]
    plain = api.summary(one, 5.0, 200.0, 1000.0)
    for blank in (None, "", 0):
        assert api.summary(one, 5.0, 200.0, 1000.0, blank) == plain


def test_a_nema_sensitivity_is_as_good_as_an_efficiency(cat):
    """A custom system may quote either; they must describe the same scanner."""
    base = next(s for s in cat["systems"] if s["name"] == "Omni 128")
    by_eps = dict(base, epsilon=base["epsilon"], S_nema=None)
    d = json.loads(api.derived(by_eps))
    by_nema = dict(base, epsilon=None, S_nema=d["S_nema"])
    assert api.scanner_from(by_nema).efficiency() == pytest.approx(
        base["epsilon"], rel=1e-12)
    assert json.loads(api.summary(by_nema, 5.0, 200.0, 1000.0))["snr2_min"] == (
        pytest.approx(json.loads(api.summary(by_eps, 5.0, 200.0, 1000.0))["snr2_min"],
                      rel=1e-12))


def test_the_form_can_convert_between_the_two(cat):
    """What the hint under the field says, and the identity behind it."""
    one = next(s for s in cat["systems"] if s["name"] == "uMI Panorama GS")
    d = json.loads(api.derived(one))
    assert d["S_nema_measured"] is True
    assert d["S_nema"] == pytest.approx(1000.0 * d["S_ideal"] * d["epsilon"], rel=1e-12)
    implied = json.loads(api.derived(dict(one, S_nema=None, epsilon=0.25)))
    assert implied["S_nema_measured"] is False
    assert implied["S_nema"] == pytest.approx(1000.0 * d["S_ideal"] * 0.25, rel=1e-12)


def test_the_table_columns_are_all_supplied(cat):
    """CTR and the NEMA sensitivity are new columns; a system without time of
    flight must say so rather than showing a number."""
    tof = json.loads(api.summary(
        next(s for s in cat["systems"] if s["name"] == "Biograph Vision Quadra"),
        5.0, 200.0, 1000.0))
    assert tof["ctr"] == pytest.approx(230.0, abs=0.5)
    assert tof["S_nema"] == pytest.approx(175.3) and tof["S_nema_measured"]
    plain = json.loads(api.summary(
        next(s for s in cat["systems"] if s["name"] == "Omni 128"), 5.0, 200.0, 1000.0))
    assert plain["ctr"] is None and plain["has_tof"] is False


def test_an_empty_field_means_no_limit_however_it_arrives():
    """JavaScript's null does not reach Python as None: Pyodide hands over a
    JsNull object, which raises on float().  Every spelling of "blank" has to
    end up as no limit, or the page dies on its first call."""
    class JsNull:                       # what Pyodide passes for a JS null
        def __float__(self):
            raise TypeError("float() argument must be a real number, not 'JsNull'")

    for blank in (None, "", 0, 0.0, JsNull()):
        assert api._number(blank) is None
    assert api._number("1.2") == 1.2 and api._number(1.2) == 1.2


def test_the_single_bed_profile_is_what_a_one_bed_scan_would_give(cat):
    """The panel showing one bed position must be the same physics as the rest of
    the page, so that ε x eta x r x sigma_o^3 really is the SNR^2 of that bed."""
    import slogpet as sp_pkg
    for name in ("Biograph Vision Quadra", "Omni 128"):
        spec = next(s for s in cat["systems"] if s["name"] == name)
        bed = json.loads(api.single_bed(spec, 5.0, 200.0, 201))
        sc, task = api.scanner_from(spec), sp.Task(5.0, 200.0)
        one = sp_pkg.snr2(sc, task, 200.0, n_beds=1)
        peak = max(bed["eta"]) * bed["epsilon"] * bed["r"] * bed["sigma_o3"]
        assert peak == pytest.approx(one.snr2.max(), rel=1e-9), name
        assert bed["epsilon"] == pytest.approx(sc.efficiency(), rel=1e-15)
        assert bed["r"] == pytest.approx(sc.r(task), rel=1e-15)
        assert bed["eta"][0] == 0.0 and bed["eta"][-1] == 0.0     # past the detector
        assert len(bed["z"]) == len(bed["eta"]) == 201


def test_the_single_bed_profile_needs_no_protocol(cat):
    """It is the same curve whatever the scan length, which is why the panel has
    no scan-length control and the ripple limit does not touch it."""
    spec = next(s for s in cat["systems"] if s["name"] == "uMI Panorama GS")
    assert api.single_bed(spec, 5.0, 200.0) == api.single_bed(spec, 5.0, 200.0)
    other = json.loads(api.single_bed(spec, 5.0, 300.0))
    same = json.loads(api.single_bed(spec, 5.0, 200.0))
    # the object does change it: more attenuation, and a non-TOF r if applicable
    assert max(other["eta"]) < max(same["eta"])
