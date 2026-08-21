"""The axial geometry: eta(z) and S_ideal.

These are the two places where a sign or a factor of two would silently change
every number in the paper, so they are checked against an independent Monte
Carlo and against direct quadrature, not only against themselves.
"""
import numpy as np
import pytest

from slogpet import MU_WATER, d1, u1, eta, S_ideal_closed, S_ideal_quad

L_PETS = [200.0, 300.0, 700.0, 1060.0, 1500.0]
D_PETS = [725.0, 744.0, 788.0, 820.0]


def test_d1_is_clipped_by_the_ring_difference():
    # deep inside a long scanner the ring-difference limit is what binds
    assert d1(0.0, 1500.0, 300.0) == 150.0
    # near the end of the detector the detector length is what binds
    assert d1(700.0, 1500.0, np.inf) == 50.0
    # beyond the detector nothing is seen
    assert d1(800.0, 1500.0) == 0.0


def test_d1_never_negative():
    for z in np.linspace(-2000.0, 2000.0, 401):
        assert d1(z, 1000.0) >= 0.0


@pytest.mark.parametrize("L_pet", L_PETS)
@pytest.mark.parametrize("D_pet", D_PETS)
def test_eta_without_attenuation_is_u1(L_pet, D_pet):
    for z in np.linspace(-L_pet / 2, L_pet / 2, 21):
        assert eta(z, L_pet, D_pet, 0.0) == pytest.approx(u1(z, L_pet, D_pet), rel=1e-14)


@pytest.mark.parametrize("L_pet", L_PETS)
def test_eta_peaks_at_the_centre_and_falls_monotonically(L_pet):
    zs = np.linspace(0.0, L_pet / 2, 51)
    v = np.array([eta(z, L_pet, 740.0, 200.0) for z in zs])
    assert np.all(np.diff(v) <= 1e-15)
    assert v[0] == max(v)


def test_eta_is_even_in_z():
    for z in (37.0, 150.0, 400.0):
        a = eta(z, 1000.0, 740.0, 200.0)
        b = eta(-z, 1000.0, 740.0, 200.0)
        assert a == pytest.approx(b, rel=1e-14)


def test_eta_vanishes_outside_the_detector():
    assert eta(600.0, 1000.0, 740.0, 200.0) == 0.0
    assert eta(500.0, 1000.0, 740.0, 200.0) == 0.0


def test_attenuation_is_monotone_in_the_object_diameter():
    v = [eta(0.0, 1000.0, 740.0, D) for D in (0.0, 100.0, 200.0, 300.0, 400.0)]
    assert all(a > b for a, b in zip(v, v[1:]))


def test_a_larger_ring_is_less_sensitive():
    v = [eta(0.0, 1000.0, D, 200.0) for D in (600.0, 740.0, 900.0)]
    assert all(a > b for a, b in zip(v, v[1:]))


def test_attenuation_factor_is_bounded_by_the_shortest_and_longest_chord():
    """Every pair crosses at least D_cyl of water, and eta is an average over
    chords D_cyl/sin(theta) >= D_cyl, so eta/u1 must lie in (0, exp(-mu D))."""
    L_pet, D_pet, D_cyl = 1000.0, 740.0, 200.0
    for z in (0.0, 200.0, 400.0):
        ratio = eta(z, L_pet, D_pet, D_cyl) / u1(z, L_pet, D_pet)
        assert 0.0 < ratio < np.exp(-MU_WATER * D_cyl)


@pytest.mark.parametrize("L_pet", L_PETS)
@pytest.mark.parametrize("L_mrd", [np.inf, 282.0, 500.0])
@pytest.mark.parametrize("L_s", [200.0, 700.0, 1800.0])
def test_closed_form_matches_quadrature(L_pet, L_mrd, L_s):
    a = S_ideal_closed(L_pet, 820.0, L_s, L_mrd)
    b = S_ideal_quad(L_pet, 820.0, L_s, L_mrd)
    assert a == pytest.approx(b, rel=1e-12)


def test_the_three_branches_are_all_exercised():
    """The closed form has three regimes; make sure each one is reached and that
    the formula is continuous where they meet."""
    D_pet, L_s = 820.0, 700.0
    h = lambda x: np.sqrt(x * x + D_pet * D_pet)

    # (a) the ring-difference limit bites over the whole source: S is constant
    s1 = S_ideal_closed(1500.0, D_pet, L_s, 300.0)
    s2 = S_ideal_closed(2000.0, D_pet, L_s, 300.0)
    assert s1 == pytest.approx(s2, rel=1e-14)
    assert s1 == pytest.approx(300.0 / h(300.0), rel=1e-14)

    # (b) the source sticks out of the detector
    assert S_ideal_closed(300.0, D_pet, L_s) == pytest.approx(
        h(300.0) / L_s - h(0.0) / L_s, rel=1e-12)

    # (c) no jump where the branches meet.  Sweep the ring-difference limit
    # across L_pet - L_s = 800 mm and across L_pet = 1500 mm in 0.1 mm steps: a
    # discontinuity would show up as a step of order the function value itself,
    # while the true slope contributes only ~5e-5 per step.
    M = np.linspace(600.0, 1600.0, 10001)
    v = np.array([S_ideal_closed(1500.0, D_pet, L_s, m) for m in M])
    assert np.abs(np.diff(v)).max() < 1e-3
    assert np.all(np.diff(v) >= -1e-15)          # and it is monotone in L_mrd


def test_an_infinitely_long_scanner_sees_everything():
    assert S_ideal_closed(1e9, 820.0, 700.0) == pytest.approx(1.0, rel=1e-9)


def test_a_longer_scanner_is_never_less_sensitive():
    v = [S_ideal_closed(L, 820.0, 700.0) for L in (200.0, 400.0, 800.0, 1600.0)]
    assert all(a < b for a, b in zip(v, v[1:]))


@pytest.mark.slow
@pytest.mark.parametrize("L_pet,D_cyl,z", [
    (300.0, 0.0, 0.0), (300.0, 200.0, 75.0),
    (1000.0, 0.0, 250.0), (1000.0, 200.0, 450.0),
    (1500.0, 200.0, 0.0),
])
def test_eta_against_monte_carlo(L_pet, D_cyl, z):
    """Sample directions isotropically, require both photons to land on the
    detector, weight by the pair survival probability."""
    from slogpet.validation import monte_carlo_profile
    a = eta(z, L_pet, 740.0, D_cyl)
    m, s = monte_carlo_profile(z, L_pet, 740.0, D_cyl)
    assert abs(a - m) < 4 * s + 1e-9, f"analytic {a}, Monte Carlo {m} +/- {s}"


@pytest.mark.slow
@pytest.mark.parametrize("L_pet,L_mrd", [(300.0, np.inf), (1060.0, np.inf), (1060.0, 282.0)])
def test_S_ideal_against_monte_carlo(L_pet, L_mrd):
    from slogpet.validation import monte_carlo_S
    a = S_ideal_closed(L_pet, 820.0, 700.0, L_mrd)
    m, s = monte_carlo_S(L_pet, 820.0, 700.0, L_mrd)
    assert abs(a - m) < 4 * s + 1e-9, f"closed form {a}, Monte Carlo {m} +/- {s}"
