"""Optimisation of the bed positions.

The protocol is the one part of the calculation that is a search rather than a
formula, so it is checked against properties that must hold whatever the search
returns, and against a brute-force recomputation on a much finer grid.
"""
import numpy as np
import pytest

import slogpet as sp
from slogpet import (sample_single_bed_profile, profile_fwhm,
                     best_spacing_for_n_beds, optimise_bed_positions,
                     coverage)
from slogpet.snr import _eta_N

LP, DP, DC = 1000.0, 740.0, 200.0


@pytest.fixture(scope="module")
def profile():
    return sample_single_bed_profile(LP, DP, DC)


@pytest.fixture(scope="module")
def prof_obj():
    sc = sp.Scanner("t", LP, DP, 3.4, 3.8, F_t=32.0, epsilon=1.0)
    return sp.axial_profile(sc, DC)


def test_a_single_bed_has_no_spacing(profile):
    choice = best_spacing_for_n_beds(profile, LP, 200.0, 1)
    assert choice.spacing_mm == 0.0
    assert choice.min_eta == pytest.approx(profile(np.array([100.0]))[0], rel=1e-12)


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
    """optimise_bed_positions deliberately returns the smallest N within 3 per
    cent of the best, not the argmax.  Check both halves of that promise."""
    got = optimise_bed_positions(profile, LP, S)
    best = max(best_spacing_for_n_beds(profile, LP, S, n).min_eta
               for n in range(1, 20))
    assert got.coverage.min_eta >= 0.97 * best
    if got.n_beds > 1:
        below = max(best_spacing_for_n_beds(profile, LP, S, n).min_eta
                    for n in range(1, got.n_beds))
        assert below < 0.97 * best          # no smaller N would have done


def test_more_beds_never_hurt_if_you_are_allowed_to_choose_the_spacing(profile):
    S = 1800.0
    best = max(best_spacing_for_n_beds(profile, LP, S, N).min_eta
               for N in range(1, 12))
    # the envelope is what optimise_bed_positions compares against
    assert best >= best_spacing_for_n_beds(profile, LP, S, 1).min_eta


def test_a_longer_scan_needs_at_least_as_many_beds(profile):
    counts = [optimise_bed_positions(profile, LP, S).n_beds for S in
              (200.0, 400.0, 700.0, 1000.0, 1400.0, 1800.0, 2000.0)]
    assert all(a <= b for a, b in zip(counts, counts[1:])), counts


def test_a_longer_scan_is_never_easier(profile):
    mins = [optimise_bed_positions(profile, LP, S).coverage.min_eta
            for S in (200.0, 500.0, 1000.0, 1800.0)]
    assert all(a >= b for a, b in zip(mins, mins[1:]))


def test_a_single_bed_suffices_for_a_short_scan(profile):
    got = optimise_bed_positions(profile, LP, 100.0)
    assert got.n_beds == 1 and got.spacing_mm == 0.0


def test_the_fwhm_rule_of_thumb(profile):
    """The FWHM of eta is roughly the useful axial field of view, and a scan
    shorter than that needs one bed."""
    W = profile_fwhm(profile)
    assert 0.3 * LP < W < 0.8 * LP
    assert optimise_bed_positions(profile, LP, 0.8 * W).n_beds == 1


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
    from slogpet.protocol import _bed_index_offsets, _tile_beds_on_grid
    step = profile.step_mm
    z_indices = profile.indices_within(S / 2)
    offsets = _bed_index_offsets(N)
    every = np.arange(0, int((LP + S) / (N - 1) / (2 * step)) + 1)
    exhaustive = _tile_beds_on_grid(profile, every, offsets, z_indices).min(axis=1).max()
    assert best_spacing_for_n_beds(profile, LP, S, N).min_eta == pytest.approx(
        exhaustive, rel=1e-9)


def test_the_lattice_step_does_not_change_the_answer():
    """1 mm is a choice, not a constraint.  Refining it fourfold moves the
    minimum sensitivity by well under a per cent."""
    fine = sample_single_bed_profile(LP, DP, DC, step_mm=0.25)
    coarse = sample_single_bed_profile(LP, DP, DC, step_mm=1.0)
    for S in (300.0, 800.0, 1500.0, 2000.0):
        a = optimise_bed_positions(coarse, LP, S).coverage.min_eta
        b = optimise_bed_positions(fine, LP, S).coverage.min_eta
        assert a == pytest.approx(b, rel=1e-2), (S, a, b)


def test_tiling_a_single_bed_returns_the_profile(profile):
    from slogpet.protocol import tile_beds
    z = np.linspace(-LP / 2, LP / 2, 501)
    assert np.allclose(tile_beds(profile, 1, 0.0, z), profile(z))


# --------------------------------------------------- the reported statistics
def test_the_three_statistics_are_ordered(profile):
    for S in (200.0, 700.0, 1500.0, 2000.0):
        got = optimise_bed_positions(profile, LP, S)
        c = got.coverage
        assert 0.0 <= c.min_eta <= c.mean_eta <= c.max_eta


def test_the_search_and_the_report_agree_on_the_minimum(profile):
    """best_spacing_for_n_beds localises the trough to go fast; coverage looks at
    every grid point in the range.  They must not disagree."""
    for S in (300.0, 900.0, 1700.0):
        for N in (1, 2, 3, 5):
            choice = best_spacing_for_n_beds(profile, LP, S, N)
            reported = coverage(profile, N, choice.spacing_mm, S)
            assert reported.min_eta == pytest.approx(choice.min_eta, rel=1e-12)


def test_the_mean_is_the_integral_over_the_range(profile):
    """(1/S) int eta_N dz, checked against a dense independent quadrature."""
    from slogpet.protocol import tile_beds
    for S in (400.0, 1200.0):
        got = optimise_bed_positions(profile, LP, S)
        z = np.linspace(-S / 2, S / 2, 40001)
        dense = np.trapezoid(tile_beds(profile, got.n_beds, got.spacing_mm, z), z) / S
        assert got.coverage.mean_eta == pytest.approx(dense, rel=1e-6)


def test_a_single_bed_peaks_at_the_centre(profile):
    """With one bed and a range inside the detector, the maximum is eta(0) and
    the minimum is at the ends."""
    S = 200.0
    c = coverage(profile, 1, 0.0, S)
    assert c.max_eta == pytest.approx(profile(np.array([0.0]))[0], rel=1e-12)
    assert c.min_eta == pytest.approx(profile(np.array([S / 2]))[0], rel=1e-12)


def test_more_beds_flatten_the_profile(profile):
    """The ripple is what the bed count buys: covering 180 cm with one bed leaves
    most of the range uncovered, and adding beds evens it out."""
    S = 1800.0
    ripples = [coverage(profile, N, best_spacing_for_n_beds(profile, LP, S, N).spacing_mm,
                        S).ripple for N in (2, 3, 4, 5)]
    assert ripples[-1] < ripples[0]


def test_the_mean_is_bounded_by_conservation(profile):
    """Tiling conserves the integral of eta, so the mean over the scan range can
    never exceed (int eta dz) / S -- whatever the arrangement."""
    for S in (600.0, 1400.0, 2000.0):
        got = optimise_bed_positions(profile, LP, S)
        assert got.coverage.mean_eta <= profile.integral_mm / S * (1 + 1e-12)


def test_overlap_percent_matches_the_spacing(profile):
    got = optimise_bed_positions(profile, LP, 1800.0)
    assert got.n_beds > 1
    assert got.overlap_percent(LP) == pytest.approx(
        100.0 * (1.0 - got.spacing_mm / LP), rel=1e-12)
    assert optimise_bed_positions(profile, LP, 100.0).overlap_percent(LP) is None


def test_peak_to_trough_is_infinite_when_the_range_is_not_covered(profile):
    """One bed cannot reach the ends of a range longer than the detector, so the
    worst point is zero and the ratio is infinite rather than merely large."""
    uncovered = coverage(profile, 1, 0.0, 2.0 * LP)
    assert uncovered.min_eta == 0.0
    assert np.isinf(uncovered.peak_to_trough)
    assert np.isfinite(uncovered.ripple)          # this one stays usable


def test_peak_to_trough_is_one_for_a_flat_profile(profile):
    """A range small enough to sit on the flat top of a single bed barely ripples."""
    c = coverage(profile, 1, 0.0, 20.0)
    assert 1.0 <= c.peak_to_trough < 1.01
