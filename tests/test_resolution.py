"""The resolution factor r.

r is where the task enters: it is the only place the SLoG size and the three
detector resolutions appear, and it is what makes the comparison between systems
depend on what one is trying to see.
"""
import itertools

import numpy as np
import pytest

from slogpet import FWHM, C_OVER_2, w, r_tof, r_nontof, r_of, F_t_equiv

R_MAX = 15.0 / 2.0


def test_the_weight_is_the_same_in_sigma_and_in_fwhm():
    """w = F_o^2/(F_i^2+F_o^2) is scale invariant, which is why the note can be
    written in FWHM without changing a single number."""
    for F_i, F_o in itertools.product([1.0, 3.6, 32.0], [2.0, 5.0, 10.0]):
        assert w(F_i, F_o) == pytest.approx(w(F_i / FWHM, F_o / FWHM), rel=1e-15)


def test_the_weight_is_bounded():
    for F_i in (0.0, 1e-6, 3.6, 57.0, 1e6):
        for F_o in (1.0, 5.0, 20.0):
            assert 0.0 < w(F_i, F_o) <= 1.0
    assert w(0.0, 5.0) == 1.0


def test_perfect_resolution_saturates_r_at_fifteen_halves():
    assert r_tof(1e-12, 1e-12, 1e-12, 1.0) == pytest.approx(R_MAX, rel=1e-12)


def test_r_never_exceeds_its_bound():
    for F_t, F_y, F_z, F_o in itertools.product(
            [1.0, 28.0, 57.0], [1.0, 2.9, 4.3], [1.0, 2.8, 5.0], [2.0, 5.0, 10.0, 20.0]):
        assert 0.0 < r_tof(F_t, F_y, F_z, F_o) <= R_MAX


def test_r_is_symmetric_in_its_three_resolutions():
    """The three directions enter identically -- a consequence of the SLoG being
    a single derivative of an isotropic Gaussian."""
    a, b, c, F_o = 3.0, 7.0, 20.0, 6.0
    ref = r_tof(a, b, c, F_o)
    for p in itertools.permutations((a, b, c)):
        assert r_tof(*p, F_o) == pytest.approx(ref, rel=1e-14)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_r_degrades_monotonically_as_any_resolution_worsens(i):
    base = [3.0, 3.0, 3.0]
    prev = np.inf
    for val in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        args = list(base)
        args[i] = val
        cur = r_tof(*args, 6.0)
        assert cur < prev
        prev = cur


def test_a_bigger_slog_is_always_easier():
    prev = 0.0
    for F_o in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        cur = r_tof(32.0, 3.6, 3.6, F_o)
        assert cur > prev
        prev = cur


def test_tomitani_constant():
    """The equivalent TOF width of a non-TOF system in a cylinder of diameter D
    is sigma_t = D/(2 sqrt(pi)), i.e. FWHM = 0.664 D."""
    for D in (100.0, 200.0, 300.0):
        assert F_t_equiv(D) / D == pytest.approx(0.664, abs=5e-4)
        assert F_t_equiv(D) == pytest.approx(FWHM * D / (2 * np.sqrt(np.pi)), rel=1e-15)


def test_the_non_tof_factor_is_the_tof_one_with_the_timing_term_dropped():
    """r_nonTOF is r_TOF evaluated at the equivalent width with w_t removed from
    the additive bracket -- not simply r_TOF at that width, which is what the
    identity below quantifies."""
    F_y, F_z, F_o, D = 3.4, 3.8, 6.0, 200.0
    wy, wz = w(F_y, F_o), w(F_z, F_o)
    expected = ((F_o / F_t_equiv(D)) * np.sqrt(wy * wz)
                * (0.5 * (wy + wz)**2 + wy**2 + wz**2))
    assert r_nontof(F_y, F_z, F_o, D) == pytest.approx(expected, rel=1e-15)


def test_substituting_the_equivalent_width_slightly_overestimates():
    """Feeding the equivalent width into r_TOF is not the same as r_nonTOF: it
    keeps a w_t term that the limit drops.  The error is small, always positive,
    and shrinks as the object grows relative to the SLoG -- worth pinning, since
    the two are easy to confuse."""
    ratios = []
    for D in (100.0, 200.0, 300.0, 400.0):
        a = r_nontof(3.4, 3.8, 6.0, D)
        b = r_tof(F_t_equiv(D), 3.4, 3.8, 6.0)
        ratios.append(b / a)
        assert 1.0 < b / a < 1.01
    assert all(x > y for x, y in zip(ratios, ratios[1:]))


def test_a_bigger_object_makes_a_non_tof_system_worse():
    v = [r_nontof(3.4, 3.8, 6.0, D) for D in (100.0, 200.0, 300.0, 400.0)]
    assert all(a > b for a, b in zip(v, v[1:]))


def test_r_of_dispatches_on_the_presence_of_tof():
    assert r_of(32.0, 3.6, 3.6, 5.0, 300.0) == r_tof(32.0, 3.6, 3.6, 5.0)
    assert r_of(None, 4.3, 4.4, 5.0, 300.0) == r_nontof(4.3, 4.4, 5.0, 300.0)


def test_ctr_conversion():
    """F_t = (c/2) CTR: 200 ps is 30 mm."""
    assert 200.0 * C_OVER_2 == pytest.approx(30.0, rel=1e-15)


@pytest.mark.slow
@pytest.mark.parametrize("D", [100.0, 200.0, 300.0])
def test_non_tof_limit_against_equation_53(D):
    """The closed-form non-TOF limit against a direct numerical evaluation of
    Eq. (53) of Nuyts et al. with the TOF width taken to 30 times the object.

    This is the check that the non-TOF branch of the paper rests on.
    """
    from scipy.special import erfc
    F_o, F_y, F_z = 6.0, 3.4, 3.8
    so, sy, sz = F_o / FWHM, F_y / FWHM, F_z / FWHM
    K = 1.0 / (16 * np.pi * np.sqrt(np.pi))

    def master(a, b, st, n=400001):
        sa, sb = np.sqrt(st**2 + a**2), np.sqrt(st**2 + b**2)
        x = np.linspace(-(6 * st + D), 6 * st + D, n)
        r2 = np.sqrt(2) * st
        Bt = 0.5 * (erfc(-(D / 2 - x) / r2) - erfc(-(-D / 2 - x) / r2))
        lognum = -0.5 * (x / sa)**2 - 0.5 * (x / sb)**2 - np.log(2 * np.pi * sa * sb)
        integ = np.where(Bt > 1e-290,
                         np.exp(lognum - np.log(np.maximum(Bt, 1e-300))), 0.0)
        return (np.trapezoid(integ, x)
                / np.sqrt(2 * np.pi * (2 * sy**2 + a**2 + b**2))
                / np.sqrt(2 * np.pi * (2 * sz**2 + a**2 + b**2)))

    h, st = 1e-3, 30 * D
    d2 = (master(so + h, so + h, st) - master(so + h, so - h, st)
          - master(so - h, so + h, st) + master(so - h, so - h, st)) / (4 * h * h)
    exact = d2 * so**5 / K
    assert exact == pytest.approx(r_nontof(F_y, F_z, F_o, D), rel=2e-3)
