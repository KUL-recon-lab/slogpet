"""The dimensionless resolution factor of the SLoG detection task.

For a system whose point response has FWHM ``F_t`` along the time-of-flight
direction and ``F_y``, ``F_z`` transaxially and axially, detecting a SLoG of size
``F_o`` gives a Hotelling SNR^2 proportional to ``r``.  Each direction enters only
through the weight ``w_i = F_o^2 / (F_i^2 + F_o^2)``, which is unchanged if all
widths are given as standard deviations instead; the paper uses FWHM throughout
because that is how PET resolutions are reported.

Without time of flight the derivation of Nuyts et al. (their Eq. 53) keeps the
finite object instead, and the result is ``r_nontof``: the same expression with
an equivalent TOF width ``sigma_t = D_cyl / (2 sqrt(pi))`` substituted and the
``w_t`` term dropped from the additive bracket.

Pure functions; all widths in mm FWHM, all coincidence timings in ps.
"""
import numpy as np

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # sigma -> FWHM
C_OVER_2 = 0.15                           # mm per ps: CTR [ps] -> F_t [mm]

__all__ = ["FWHM", "C_OVER_2", "w", "ctr_to_mm", "r_tof", "r_nontof",
           "F_t_equiv", "R_MAX"]

R_MAX = 15.0 / 2.0        # r at perfect resolution in all three directions


def ctr_to_mm(ctr_ps):
    """Coincidence time resolution in ps FWHM -> positioning FWHM in mm."""
    return C_OVER_2 * np.asarray(ctr_ps, float)


def w(F_i, F_o):
    """Resolution weight ``w_i = F_o^2 / (F_i^2 + F_o^2)``, in (0, 1]."""
    return F_o**2 / (F_i**2 + F_o**2)


def r_tof(F_t, F_y, F_z, F_o):
    """Resolution factor of a TOF system; ``0 < r <= 15/2``."""
    wt, wy, wz = (w(f, F_o) for f in (F_t, F_y, F_z))
    return np.sqrt(wt * wy * wz) * (0.5 * (wt + wy + wz)**2
                                    + wt**2 + wy**2 + wz**2)


def F_t_equiv(D_cyl):
    """Equivalent TOF resolution (FWHM, mm) of a system without time of flight:
    ``sqrt(8 ln 2) / (2 sqrt(pi)) * D_cyl = 0.664 D_cyl`` (Tomitani's constant)."""
    return FWHM / (2.0 * np.sqrt(np.pi)) * D_cyl


def r_nontof(F_y, F_z, F_o, D_cyl):
    """Resolution factor of a non-TOF system in a cylinder of diameter ``D_cyl``."""
    wy, wz = w(F_y, F_o), w(F_z, F_o)
    return (F_o / F_t_equiv(D_cyl)) * np.sqrt(wy * wz) * (0.5 * (wy + wz)**2
                                                          + wy**2 + wz**2)


def r_of(F_t, F_y, F_z, F_o, D_cyl):
    """Dispatch on whether the system has time of flight (``F_t is None``)."""
    return (r_nontof(F_y, F_z, F_o, D_cyl) if F_t is None
            else r_tof(F_t, F_y, F_z, F_o))


__all__.append("r_of")
