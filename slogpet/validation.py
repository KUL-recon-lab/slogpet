"""Independent checks of the analytic results.

These are the tests of whether the physics is right, as opposed to the pinned
values in ``tests/golden.json``, which only test that it has not changed.  Three
checks:

* ``eta(z)`` and ``S_ideal`` against Monte Carlo sampling of the geometry;
* the closed form of ``S_ideal`` against quadrature;
* the non-TOF resolution factor against a direct numerical evaluation of
  Eq. (53) of Nuyts et al. (2025), which keeps the finite object instead of
  assuming a narrow TOF kernel.

They print a report; the numbers they compare are computed by the pure functions
in the other modules.
"""
import numpy as np

from .geometry import MU_WATER, u1, eta, S_ideal_quad, S_ideal_closed
from .resolution import FWHM, C_OVER_2, r_tof, r_nontof, F_t_equiv

__all__ = ["monte_carlo_profile", "monte_carlo_S", "run_checks",
           "check_resolution_factor", "check_nontof"]

D_PET_DEFAULT = 740.0
LENGTHS_DEFAULT = [1500.0, 1000.0, 600.0, 300.0]


def monte_carlo_profile(z, L_pet, D_pet, D_cyl, mu=MU_WATER, L_mrd=np.inf,
                        n=4_000_000, seed=0):
    """Check of ``eta(z)``: sample directions isotropically, require both photons
    to land on the detector, weight by the pair survival probability."""
    R = D_pet / 2.0
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, n)                 # isotropic: u = cos(theta) uniform
    delta = R * np.abs(u) / np.sqrt(1.0 - u * u)  # axial half-separation of the ends
    ok = (z + delta <= L_pet / 2) & (z - delta >= -L_pet / 2) & (2 * delta <= L_mrd)
    if D_cyl == 0.0 or mu == 0.0:
        wgt = np.where(ok, 1.0, 0.0)
    else:
        wgt = np.where(ok, np.exp(-mu * D_cyl / np.sqrt(1.0 - u * u)), 0.0)
    return wgt.mean(), wgt.std() / np.sqrt(n)


def monte_carlo_S(L_pet, D_pet, L_s, L_mrd=np.inf, n=20_000_000, seed=12345):
    """Check of ``S_ideal``: sample a point uniformly along the source and a
    direction isotropically, count the fraction giving a valid coincidence."""
    R = D_pet / 2.0
    rng = np.random.default_rng(seed)
    z = rng.uniform(-L_s / 2, L_s / 2, n)
    u = rng.uniform(-1.0, 1.0, n)
    d = R * np.abs(u) / np.sqrt(1.0 - u * u)
    ok = (z + d <= L_pet / 2) & (z - d >= -L_pet / 2) & (2 * d <= L_mrd)
    p = ok.mean()
    return p, np.sqrt(p * (1 - p) / n)


def run_checks(D_pet=D_PET_DEFAULT, lengths=None):
    lengths = LENGTHS_DEFAULT if lengths is None else lengths
    print("eta(z): analytic vs Monte Carlo   (D_PET = %.0f mm, mu = %g /mm)\n"
          % (D_pet, MU_WATER))
    print(f"{'L_PET':>7} {'D_cyl':>6} {'z':>7} {'analytic':>11} {'Monte Carlo':>20}")
    for L_pet in (300.0, 1000.0, 1500.0):
        for D_cyl in (0.0, 200.0):
            for z in (0.0, L_pet / 4.0, 0.45 * L_pet):
                a = eta(z, L_pet, D_pet, D_cyl)
                m, s = monte_carlo_profile(z, L_pet, D_pet, D_cyl)
                flag = "" if abs(a - m) < 4 * s + 1e-9 else "   <-- MISMATCH"
                print(f"{L_pet:7.0f} {D_cyl:6.0f} {z:7.1f} {a:11.6f} "
                      f"{m:11.6f} +/- {s:.6f}{flag}")
    print("\nHow far the D_cyl = 0 profile is from a triangle.")
    print("A perfect triangle would give eta(L_PET/4)/eta(0) = 0.5 exactly.\n")
    print(f"{'L_PET':>7} {'d_1(0)/R':>9} {'ratio':>8} {'excess over triangle':>22}")
    for L_pet in lengths[::-1]:
        r = u1(L_pet / 4.0, L_pet, D_pet) / u1(0.0, L_pet, D_pet)
        print(f"{L_pet:7.0f} {(L_pet/2)/(D_pet/2):9.2f} {r:8.4f} "
              f"{100*(r/0.5-1):21.1f}%")
    print("\nS_ideal, quadrature vs closed form (L_s = 700 mm):")
    for L_pet in lengths[::-1]:
        for L_mrd in (np.inf, 300.0):
            q = S_ideal_quad(L_pet, D_pet, 700.0, L_mrd)
            c = S_ideal_closed(L_pet, D_pet, 700.0, L_mrd)
            print(f"  L_PET={L_pet:6.0f}  L_MRD={L_mrd:8.1f}  {q:.10f}  {c:.10f}"
                  f"  diff {abs(q-c):.1e}")


def check_resolution_factor():
    print("\nResolution factor F (F_max = 15/2 = %.1f)\n" % 7.5)
    print("Effect of improving the detector resolution from 4.5 to 1.5 mm FWHM:")
    D_cyl = 200.0
    Ft_nontof = F_t_equiv(D_cyl)
    print("  (a non-TOF system in a %.0f mm cylinder is modelled with F_t = %.1f mm,"
          " i.e. CTR = %.0f ps)" % (D_cyl, Ft_nontof, Ft_nontof / C_OVER_2))
    for Fo in (1.5, 6.0):
        for ctr_ps in (214.0, None):
            Ft = Ft_nontof if ctr_ps is None else ctr_ps * C_OVER_2
            a = r_tof(Ft, 4.5, 4.5, Fo)
            b = r_tof(Ft, 1.5, 1.5, Fo)
            tof = "no TOF" if ctr_ps is None else f"{ctr_ps:.0f} ps"
            print(f"  SLoG {Fo:4.1f} mm FWHM, {tof:>7}: "
                  f"F = {a:.4e} -> {b:.4e},  ratio {b/a:7.2f}")
    print("\n  perfect resolution in all three directions: F = %.4f" %
          r_tof(1e-9, 1e-9, 1e-9, 1.0))


def check_nontof(Fo=6.0, Fy=3.4, Fz=3.8):
    """Non-TOF limit of the resolution factor.

    Eq. (11) of Nuyts et al. (2025) assumes the TOF kernel is narrow compared with
    the object, so letting sigma_t -> infinity sends r -> 0, which is unphysical.
    The correct treatment is their Eq. (53), which keeps the finite object.  This
    routine evaluates Eq. (53) numerically and checks it against ``r_nontof``.
    """
    from scipy.special import erfc
    so, sy, sz = Fo / FWHM, Fy / FWHM, Fz / FWHM     # Eq. (53) is derived in sigma
    K = 1.0 / (16 * np.pi * np.sqrt(np.pi))

    def master(a, b, st, D, n=400001):
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

    def F_exact(st, D, h=1e-3):
        d2 = (master(so + h, so + h, st, D) - master(so + h, so - h, st, D)
              - master(so - h, so + h, st, D) + master(so - h, so - h, st, D)) / (4 * h * h)
        return d2 * so**5 / K

    print("non-TOF limit of F  (F_o=%.1f, F_y=%.1f, F_z=%.1f mm FWHM)" % (Fo, Fy, Fz))
    for D in (100.0, 200.0, 300.0):
        F_lim = r_nontof(Fy, Fz, Fo, D)
        F_sub = r_tof(F_t_equiv(D), Fy, Fz, Fo)
        fe = [F_exact(st, D) for st in (3 * D, 10 * D, 30 * D)]
        print("  D=%3.0f mm  Eq.(53) at sigma_t=3D,10D,30D: %.6e %.6e %.6e" % (D, *fe))
        print("            closed-form limit             : %.6e  (rel. err %.1e)"
              % (F_lim, abs(fe[-1] / F_lim - 1)))
        print("            sigma_t = D/(2 sqrt(pi)) in F : %.6e  (%+.2f %%)"
              % (F_sub, 100 * (F_sub / F_lim - 1)))


def check_protocol(L_pet=1000.0, D_pet=740.0, D_cyl=200.0):
    """Independent checks of the bed-position machinery.

    Two things must hold whatever the optimiser does:

    * tiling conserves the integral -- ``int eta_N dz = int eta dz`` for any N and
      any spacing, because each bed contributes ``1/N`` of the same profile;
    * the minimum reported by the optimiser is the actual minimum of the tiled
      profile over the scan range, recomputed here on a finer grid.
    """
    from .snr import axial_profile, optimal_protocol, protocol_for_N, _eta_N
    from .types import Scanner

    sc = Scanner("check", L_pet, D_pet, 3.4, 3.8, F_t=32.0, epsilon=1.0)
    prof = axial_profile(sc, D_cyl)
    print("\nProtocol checks (L_PET = %.0f mm, D_cyl = %.0f mm)\n" % (L_pet, D_cyl))
    print(f"{'S':>7} {'N*':>3} {'overlap':>8} {'min eta_N':>11} "
          f"{'recomputed':>11} {'int eta_N / int eta':>21}")
    zs = np.linspace(-3 * L_pet, 3 * L_pet, 60001)
    for S in (200.0, 500.0, 1000.0, 1800.0):
        p = optimal_protocol(prof, S)
        zf = np.linspace(-S / 2, S / 2, 20001)
        m = _eta_N(prof, p, zf).min()
        I = np.trapezoid(_eta_N(prof, p, zs), zs)
        ov = "---" if p.n_beds == 1 else "%.1f%%" % p.overlap
        flag = "" if abs(m / p.min_eta - 1) < 5e-3 and abs(I / prof.integral - 1) < 1e-4 \
            else "   <-- MISMATCH"
        print(f"{S:7.0f} {p.n_beds:3d} {ov:>8} {p.min_eta:11.6f} {m:11.6f} "
              f"{I/prof.integral:21.8f}{flag}")
    print("\n  (the recomputed minimum uses 20001 points, the optimiser 241, so a")
    print("   difference of a few parts per thousand is expected and harmless)")
