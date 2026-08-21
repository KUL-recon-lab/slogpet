"""The assembly, and the parameter objects it is built on."""
import numpy as np
import pytest

import slogpet as sp
from slogpet import FWHM


# --------------------------------------------------------------- Scanner
def test_ctr_is_converted_to_a_length():
    sc = sp.Scanner("x", 1000.0, 740.0, 3.4, 3.8, ctr=200.0)
    assert sc.F_t == pytest.approx(30.0, rel=1e-15)
    assert sc.has_tof


def test_an_explicit_width_wins_over_the_ctr():
    sc = sp.Scanner("x", 1000.0, 740.0, 3.4, 3.8, F_t=32.0, ctr=200.0)
    assert sc.F_t == 32.0


def test_no_ctr_means_no_time_of_flight():
    sc = sp.Scanner("x", 1000.0, 740.0, 4.3, 4.4)
    assert sc.F_t is None and not sc.has_tof


def test_efficiency_is_derived_from_the_nema_sensitivity():
    sc = sp.Scanner("x", 1060.0, 820.0, 3.4, 3.8, ctr=230.0, S_nema=175.3)
    assert sc.efficiency() == pytest.approx(175.3 / (1000.0 * sc.S_ideal()), rel=1e-15)
    assert 0.0 < sc.efficiency() < 1.0


def test_an_explicit_efficiency_wins():
    sc = sp.Scanner("x", 1060.0, 820.0, 3.4, 3.8, ctr=230.0, S_nema=175.3, epsilon=0.3)
    assert sc.efficiency() == 0.3


def test_a_scanner_is_hashable_so_the_profile_can_be_cached():
    sc = sp.Scanner("x", 1000.0, 740.0, 3.4, 3.8, ctr=200.0,
                    crystal_size=[3.2, 3.2, 20.0], energy_window=[435, 585],
                    assumed=["F_y"])
    assert hash(sc) == hash(sp.Scanner("x", 1000.0, 740.0, 3.4, 3.8, ctr=200.0,
                                       crystal_size=(3.2, 3.2, 20.0),
                                       energy_window=(435, 585), assumed=("F_y",)))


def test_a_scanner_is_immutable():
    sc = sp.Scanner("x", 1000.0, 740.0, 3.4, 3.8)
    with pytest.raises(Exception):
        sc.L_pet = 1.0


# ------------------------------------------------------------------ Task
def test_sigma_from_fwhm():
    assert sp.Task(F_o=5.0, D_cyl=300.0).sigma_o == pytest.approx(5.0 / FWHM, rel=1e-15)


# ---------------------------------------------------------- axial profile
def test_the_profile_is_cached(quadra):
    sp.axial_profile.cache_clear()
    a = sp.axial_profile(quadra, 300.0)
    b = sp.axial_profile(quadra, 300.0)
    assert a is b
    assert sp.axial_profile.cache_info().hits == 1


def test_the_profile_does_not_depend_on_the_slog_size(quadra):
    """Which is the whole reason it is worth caching: the protocol is fixed by
    the geometry alone."""
    a = sp.snr2(quadra, sp.Task(5.0, 300.0), 1000.0)
    b = sp.snr2(quadra, sp.Task(10.0, 300.0), 1000.0)
    assert a.protocol == b.protocol


def test_the_profile_integrates_to_something_sensible(quadra):
    p = sp.axial_profile(quadra, 300.0)
    assert 0.0 < p.integral < quadra.L_pet
    assert p(np.array([0.0]))[0] > p(np.array([quadra.L_pet / 2 * 0.9]))[0]


# -------------------------------------------------------------- assembly
def test_the_minimum_agrees_with_the_reported_curve(quadra, task):
    r = sp.snr2(quadra, task, 1000.0)
    assert r.snr2.min() == pytest.approx(r.snr2_min, rel=5e-3)


def test_snr2_is_linear_in_the_efficiency(quadra, task):
    import dataclasses
    a = sp.snr2(dataclasses.replace(quadra, epsilon=0.2, S_nema=None), task, 800.0)
    b = sp.snr2(dataclasses.replace(quadra, epsilon=0.4, S_nema=None), task, 800.0)
    assert b.snr2_min == pytest.approx(2.0 * a.snr2_min, rel=1e-12)


def test_snr2_factorises_as_the_paper_says(quadra, task):
    """SNR^2 = epsilon x min eta_N x r x sigma_o^3, and nothing else."""
    res = sp.snr2(quadra, task, 1200.0)
    expected = (res.epsilon * res.protocol.min_eta * res.r
                * (task.F_o / FWHM) ** 3)
    assert res.snr2_min == pytest.approx(expected, rel=1e-14)


def test_the_scale_factor_passes_straight_through(quadra, task):
    a = sp.snr2(quadra, task, 800.0)
    b = sp.snr2(quadra, task, 800.0, scale=7.0)
    assert b.snr2_min == pytest.approx(7.0 * a.snr2_min, rel=1e-14)


def test_forcing_the_bed_count_is_never_better_than_optimising(quadra, task):
    best = sp.snr2(quadra, task, 1800.0)
    for N in (1, 2, 3, 4, 5):
        assert sp.snr2(quadra, task, 1800.0, n_beds=N).snr2_min <= best.snr2_min * 1.031


def test_a_curve_matches_the_points_it_is_made_of(quadra, task):
    Ss = [200.0, 600.0, 1400.0]
    curve = sp.snr2_curve(quadra, task, Ss)
    for S, res in zip(Ss, curve):
        assert res.snr2_min == pytest.approx(sp.snr2(quadra, task, S).snr2_min, rel=1e-15)


def test_a_scanner_with_no_efficiency_is_refused(task):
    sc = sp.Scanner("nameless", 1000.0, 740.0, 3.4, 3.8, ctr=200.0)
    with pytest.raises(ValueError, match="epsilon"):
        sp.snr2(sc, task, 500.0)


def test_a_longer_scanner_wins_at_every_scan_length(task):
    """Same detector, four times the length: the long system is ahead
    everywhere, and the gap widens with the scan length."""
    short = sp.Scanner("short", 261.0, 820.0, 3.6, 3.6, F_t=32.0, epsilon=0.28)
    long_ = sp.Scanner("long", 1060.0, 820.0, 3.6, 3.6, F_t=32.0, epsilon=0.28)
    gaps = []
    for S in (200.0, 500.0, 1000.0, 1800.0):
        a = sp.snr2(short, task, S).snr2_min
        b = sp.snr2(long_, task, S).snr2_min
        assert b > a
        gaps.append(b / a)
    assert gaps[-1] > gaps[0]


def test_a_ring_difference_limit_costs_sensitivity(task):
    full = sp.Scanner("full", 1060.0, 820.0, 3.6, 3.6, F_t=32.0, epsilon=0.28)
    import dataclasses
    limited = dataclasses.replace(full, L_mrd=282.0)
    assert (sp.snr2(limited, task, 1000.0).snr2_min
            < sp.snr2(full, task, 1000.0).snr2_min)


def test_the_task_can_reverse_the_ranking():
    """The point of the paper: which system wins depends on what one is looking
    for.  A long, coarse, non-TOF BGO system loses to a short, fine, TOF LYSO one
    on a 3 mm SLoG and beats it on a 15 mm SLoG -- with no parameter changed but
    the size of the thing being detected.
    """
    bgo_long = sp.Scanner("BGO 128 cm", 1280.0, 725.0, 4.3, 4.4, epsilon=0.50)
    lyso_short = sp.Scanner("LYSO 35 cm", 351.0, 788.0, 2.9, 2.8, F_t=28.0,
                            epsilon=0.20)
    fine, coarse = sp.Task(3.0, 300.0), sp.Task(15.0, 300.0)
    S = 700.0
    assert (sp.snr2(bgo_long, fine, S).snr2_min
            < sp.snr2(lyso_short, fine, S).snr2_min)
    assert (sp.snr2(bgo_long, coarse, S).snr2_min
            > sp.snr2(lyso_short, coarse, S).snr2_min)


def test_efficiency_and_resolution_trade_off_monotonically():
    """The crossing above is not an accident of two particular SLoG sizes: the
    advantage of the efficient, coarse detector grows steadily with the size of
    the SLoG."""
    bgo = sp.Scanner("BGO", 320.0, 725.0, 4.3, 4.4, epsilon=0.50)
    lyso = sp.Scanner("LYSO", 351.0, 788.0, 2.9, 2.8, F_t=28.0, epsilon=0.20)
    ratios = [sp.snr2(bgo, sp.Task(F_o, 300.0), 700.0).snr2_min
              / sp.snr2(lyso, sp.Task(F_o, 300.0), 700.0).snr2_min
              for F_o in (2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0)]
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
