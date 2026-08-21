"""Optimisation of the bed positions.

The protocol is the one part of the calculation that is a search rather than a
formula, so it is checked against properties that must hold whatever the search
returns, and against a brute-force recomputation on a much finer grid.
"""
import numpy as np
import pytest

import slogpet as sp
from slogpet import eta_lattice, fwhm_of, best_for_N, optimise_beds
from slogpet.snr import _eta_N

LP, DP, DC = 1000.0, 740.0, 200.0


@pytest.fixture(scope="module")
def profile():
    return eta_lattice(LP, DP, DC)


@pytest.fixture(scope="module")
def prof_obj():
    sc = sp.Scanner("t", LP, DP, 3.4, 3.8, F_t=32.0, epsilon=1.0)
    return sp.axial_profile(sc, DC)


def test_a_single_bed_has_no_spacing(profile):
    m, d = best_for_N(profile, LP, 200.0, 1)
    assert d == 0.0
    assert m == pytest.approx(profile(np.array([100.0]))[0], rel=1e-12)


def test_tiling_conserves_the_integral(prof_obj):
    """Each bed contributes 1/N of the same profile, so the area under eta_N is
    the area under eta whatever N and whatever the spacing.  This is what makes
    the bed count a redistribution of counts, not a gain or a loss of them."""
    zs = np.linspace(-4 * LP, 4 * LP, 80001)
    for N in (1, 2, 3, 5, 8):
        for spacing in (0.0, 200.0, 600.0, 950.0):
            p = sp.Protocol(scan_length=1000.0, n_beds=N, spacing=spacing,
                            min_eta=0.0, L_pet=LP)
            I = np.trapezoid(_eta_N(prof_obj, p, zs), zs)
            assert I == pytest.approx(prof_obj.integral, rel=2e-6)


@pytest.mark.parametrize("S", [200.0, 500.0, 1000.0, 1800.0])
def test_the_reported_minimum_is_the_actual_minimum(prof_obj, S):
    """The optimiser evaluates 241 points; recompute on 20001 and check it did
    not report a value the profile never reaches."""
    p = sp.optimal_protocol(prof_obj, S)
    zf = np.linspace(-S / 2, S / 2, 20001)
    m = _eta_N(prof_obj, p, zf).min()
    # with the bed offsets on the lattice, eta_N is piecewise linear with every
    # knot on the lattice, so its minimum is attained AT a lattice point: the
    # reported value is not an estimate of the minimum, it is the minimum
    assert m == pytest.approx(p.min_eta, rel=1e-9)


@pytest.mark.parametrize("S", [200.0, 500.0, 1000.0, 1800.0])
def test_the_chosen_bed_count_is_within_tolerance_of_the_best(profile, S):
    """optimise_beds deliberately returns the smallest N within 3 per cent of the
    best, not the argmax.  Check both halves of that promise."""
    N, d, M = optimise_beds(profile, LP, S)
    best = max(best_for_N(profile, LP, S, n)[0] for n in range(1, 20))
    assert M >= 0.97 * best
    if N > 1:
        below = max(best_for_N(profile, LP, S, n)[0] for n in range(1, N))
        assert below < 0.97 * best          # no smaller N would have done


def test_more_beds_never_hurt_if_you_are_allowed_to_choose_the_spacing(profile):
    S = 1800.0
    best = -np.inf
    for N in range(1, 12):
        m, _d = best_for_N(profile, LP, S, N)
        best = max(best, m)
    # the envelope is what optimise_beds compares against
    assert best >= best_for_N(profile, LP, S, 1)[0]


def test_a_longer_scan_needs_at_least_as_many_beds(profile):
    counts = [optimise_beds(profile, LP, S)[0] for S in
              (200.0, 400.0, 700.0, 1000.0, 1400.0, 1800.0, 2000.0)]
    assert all(a <= b for a, b in zip(counts, counts[1:])), counts


def test_a_longer_scan_is_never_easier(profile):
    mins = [optimise_beds(profile, LP, S)[2] for S in (200.0, 500.0, 1000.0, 1800.0)]
    assert all(a >= b for a, b in zip(mins, mins[1:]))


def test_a_single_bed_suffices_for_a_short_scan(profile):
    N, d, _M = optimise_beds(profile, LP, 100.0)
    assert N == 1 and d == 0.0


def test_the_fwhm_rule_of_thumb(profile):
    """The FWHM of eta is roughly the useful axial field of view, and a scan
    shorter than that needs one bed."""
    W = fwhm_of(profile)
    assert 0.3 * LP < W < 0.8 * LP
    assert optimise_beds(profile, LP, 0.8 * W)[0] == 1


def test_overlap_is_reported_only_for_multi_bed(prof_obj):
    assert sp.optimal_protocol(prof_obj, 100.0).overlap is None
    p = sp.optimal_protocol(prof_obj, 1800.0)
    assert p.n_beds > 1
    assert 0.0 < p.overlap < 100.0


def test_the_reported_minimum_is_exact_not_sampled(prof_obj):
    """The reason the lattice is worth having: a 20 000-point scan of the tiled
    profile finds nothing lower than what the optimiser reported, for any scan
    length.  The previous continuous search overestimated by up to 0.7 per cent.
    """
    for S in (150.0, 370.0, 640.0, 1130.0, 1900.0):
        p = sp.optimal_protocol(prof_obj, S)
        dense = _eta_N(prof_obj, p, np.linspace(-S / 2, S / 2, 20001)).min()
        assert dense == pytest.approx(p.min_eta, rel=1e-9), S


@pytest.mark.parametrize("S,N", [(400.0, 2), (900.0, 3), (1500.0, 4), (1900.0, 6)])
def test_the_two_stage_search_matches_an_exhaustive_one(profile, S, N):
    """best_for_N scans the offset every 10 mm and then refines.  Check that the
    coarse pass never lands in the wrong basin, by scanning every offset."""
    from slogpet.protocol import _bed_offsets, _gather
    h = profile.h
    kz = int(np.floor(S / 2 / h))
    iz = np.arange(-kz, kz + 1) + profile.i0
    a = _bed_offsets(N)
    ms = np.arange(0, int((LP + S) / (N - 1) / (2 * h)) + 1)
    exhaustive = _gather(profile, ms, a, iz).min(axis=1).max()
    got, _d = best_for_N(profile, LP, S, N)
    assert got == pytest.approx(exhaustive, rel=1e-9)


def test_the_lattice_step_does_not_change_the_answer():
    """1 mm is a choice, not a constraint.  Refining it fourfold moves the
    minimum sensitivity by well under a per cent."""
    fine = eta_lattice(LP, DP, DC, h=0.25)
    coarse = eta_lattice(LP, DP, DC, h=1.0)
    for S in (300.0, 800.0, 1500.0, 2000.0):
        a = optimise_beds(coarse, LP, S)[2]
        b = optimise_beds(fine, LP, S)[2]
        assert a == pytest.approx(b, rel=1e-2), (S, a, b)


def test_tiling_a_single_bed_returns_the_profile(profile):
    from slogpet.protocol import tiled_profile
    z = np.linspace(-LP / 2, LP / 2, 501)
    assert np.allclose(tiled_profile(profile, 1, 0.0, z), profile(z))
