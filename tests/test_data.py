"""The bundled parameters, and the claims the paper makes about them."""
import json

import numpy as np
import pytest

import slogpet as sp
from slogpet.data import (SYSTEMS_FILE, DETECTORS_FILE, load_json, load_systems,
                          load_references, load_conventions, load_detector_groups,
                          load_styles, verify)


def test_the_data_files_are_consistent():
    n_systems, n_refs = verify()
    assert n_systems == 10 and n_refs >= 8


def test_every_system_has_a_source(systems):
    refs = load_references()
    for sc in systems:
        assert str(sc.reference) in refs
        assert len(refs[str(sc.reference)]) > 20


def test_every_convention_is_documented():
    conv = load_conventions()
    for key in ("F_t", "F_y", "F_z", "D_pet", "S_nema", "energy_resolution"):
        assert key in conv


def test_efficiencies_are_physical(systems):
    for sc in systems:
        eps = sc.efficiency()
        assert 0.0 < eps < 1.0, sc.name
        assert 0.15 < eps < 0.60, f"{sc.name}: epsilon {eps:.3f} is implausible"


def test_geometry_is_physical(systems):
    for sc in systems:
        assert 150.0 < sc.L_pet < 2000.0
        assert 600.0 < sc.D_pet < 900.0
        assert 1.0 < sc.F_y < 8.0 and 1.0 < sc.F_z < 8.0
        if sc.has_tof:
            assert 100.0 < sc.ctr < 600.0
        if np.isfinite(sc.L_mrd):
            assert 0.0 < sc.L_mrd <= sc.L_pet


def test_the_efficiency_is_a_property_of_the_detector_not_the_length(systems):
    """The paper's central empirical claim about epsilon: systems sharing a
    crystal agree, across a fourfold range of axial length.  If a future edit to
    systems.json breaks this, the argument of section 6 changes.
    """
    by_crystal = {}
    for sc in systems:
        key = (sc.crystal, tuple(sc.crystal_size))
        by_crystal.setdefault(key, []).append((sc.name, sc.L_pet, sc.efficiency()))
    checked = 0
    for key, group in by_crystal.items():
        if len(group) < 2:
            continue
        eps = [e for _n, _L, e in group]
        lengths = [L for _n, L, _e in group]
        spread = max(eps) / min(eps) - 1.0
        assert spread < 0.20, f"{key}: epsilon spread {spread:.0%} over {group}"
        if max(lengths) / min(lengths) > 3.0:
            checked += 1
    assert checked >= 2, "no crystal family spans a large enough length range"


def test_the_assumed_fields_are_real_fields(systems):
    for sc in systems:
        for f in sc.assumed:
            assert hasattr(sc, f)
            assert getattr(sc, f) is not None
        if sc.assumed:
            assert sc.note, f"{sc.name}: assumed values need a note saying why"


def test_the_only_assumed_values_are_the_omni_128(systems):
    assumed = {sc.name for sc in systems if sc.assumed}
    assert assumed == {"Omni 128"}


def test_the_generic_designs_resemble_the_systems_they_come_from(systems, groups):
    """detectors.json rounds; make sure it does not misrepresent.  Every design
    parameter must lie within 25 per cent of at least one of the systems it says
    it is based on."""
    by_name = {}
    for sc in systems:
        by_name.setdefault(sc.name, []).append(sc)
    for g in groups:
        parents = [s for n in g.based_on for s in by_name[n]]
        assert parents, g.id
        for field, value in (("F_y", g.F_y), ("F_z", g.F_z), ("D_pet", g.D_pet)):
            assert any(abs(getattr(p, field) / value - 1) < 0.25 for p in parents), \
                f"design {g.id}: {field}={value} unlike {g.based_on}"
        assert any(abs(p.efficiency() / g.epsilon - 1) < 0.25 for p in parents), \
            f"design {g.id}: epsilon={g.epsilon} unlike {g.based_on}"
        if g.has_tof:
            assert any(p.has_tof and abs(p.F_t / g.F_t - 1) < 0.25 for p in parents)
        else:
            assert all(not p.has_tof for p in parents)
        assert all(tuple(p.crystal_size) == tuple(g.crystal_size) for p in parents)


def test_every_design_offers_more_than_one_length(groups):
    for g in groups:
        assert len(g.configurations) >= 2
        lengths = [sc.L_pet for sc in g.scanners]
        assert max(lengths) / min(lengths) > 1.4, g.id


def test_styles_distinguish_long_from_short(groups):
    """Solid means longer than a metre; every broken style is a short system,
    except the dash-dot, which is a long one with a ring-difference limit."""
    for g in groups:
        for sc, style in g.configurations:
            if style == "long":
                assert sc.L_pet > 1000.0, (g.id, sc.L_pet)
            elif style == "mrd":
                assert np.isfinite(sc.L_mrd)
            else:
                assert sc.L_pet <= 1000.0, (g.id, sc.L_pet)


def test_the_hypothetical_configurations_say_so(groups):
    """Design D at 70 cm does not exist; it must be marked, or a reader will
    take it for a product."""
    for g in groups:
        for sc in g.scanners:
            exists = any(abs(sc.L_pet - s.L_pet) < 5.0 for s in load_systems()
                         if tuple(s.crystal_size) == tuple(g.crystal_size))
            if not exists:
                assert sc.note, f"design {g.id} at {sc.L_pet:.0f} mm needs a note"


def test_the_files_are_valid_json_and_carry_their_own_documentation():
    for path in (SYSTEMS_FILE, DETECTORS_FILE):
        doc = load_json(path)
        assert "_comment" in doc
        assert any("mm" in line for line in doc["_comment"])
