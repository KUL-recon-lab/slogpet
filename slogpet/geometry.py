"""Axial geometry of a cylindrical PET system.

The detector is a cylinder of diameter ``D_pet`` and axial length ``L_pet``.  A
point source at axial position ``z`` sees a band of polar angles; the fraction of
the full sphere it subtends, weighted by the survival probability of the photon
pair in a water cylinder of diameter ``D_cyl``, is ``eta(z)``.  Averaging the
unattenuated version over a line source of length ``L_s`` gives ``S_ideal``.

Every function here is pure: all geometry is passed in explicitly, nothing is
cached, nothing is printed, and there is no module-level mutable state.  Lengths
are in mm throughout, attenuation coefficients in mm^-1.
"""
import numpy as np
from scipy.integrate import quad

MU_WATER = 0.0096          # mm^-1, water at 511 keV

__all__ = ["MU_WATER", "d1", "u1", "eta", "S_ideal_quad", "S_ideal_closed"]


def d1(z, L_pet, L_mrd=np.inf):
    """Axial half-reach at ``z``: how far a photon may travel axially and still
    land on the detector, also respecting the maximum ring difference."""
    return max(min(L_pet / 2.0 - abs(z), L_mrd / 2.0), 0.0)


def u1(z, L_pet, D_pet, L_mrd=np.inf):
    """cos(theta_1): the fraction of 4 pi covered at ``z``, without attenuation."""
    d = d1(z, L_pet, L_mrd)
    R = D_pet / 2.0
    return d / np.sqrt(d * d + R * R)


def eta(z, L_pet, D_pet, D_cyl, mu=MU_WATER, L_mrd=np.inf):
    """Fraction of the pairs emitted at ``z`` that reach the detector and survive
    a water cylinder of diameter ``D_cyl``."""
    U = u1(z, L_pet, D_pet, L_mrd)
    if D_cyl == 0.0 or mu == 0.0:
        return U
    return quad(lambda u: np.exp(-mu * D_cyl / np.sqrt(1.0 - u * u)),
                0.0, U, limit=200)[0]


def S_ideal_quad(L_pet, D_pet, L_s, L_mrd=np.inf):
    """``S_ideal`` by quadrature: the mean of ``u1`` over the NEMA line source.

    This exists to check ``S_ideal_closed``, so it has to be more accurate than
    the thing it checks.  The integrand has corners -- where the ring-difference
    limit starts to bind, and where the detector ends -- and adaptive quadrature
    is markedly more accurate when told where they are.
    """
    z1 = min(L_s / 2.0, L_pet / 2.0)
    pts = [z for z in ((L_pet - L_mrd) / 2.0, L_pet / 2.0) if 0.0 < z < z1]
    return 2.0 * quad(lambda z: u1(z, L_pet, D_pet, L_mrd), 0.0, z1,
                      limit=400, points=pts or None)[0] / L_s


def S_ideal_closed(L_pet, D_pet, L_s, L_mrd=np.inf):
    """``S_ideal`` in closed form.

    Three branches, according to whether the ring-difference limit bites over the
    whole source, part of it, or not at all."""
    h = lambda x: np.sqrt(x * x + D_pet * D_pet)
    Lp = max(L_pet - L_s, 0.0)
    M = min(L_mrd, L_pet)
    pM = M / h(M)
    if M <= Lp:
        return pM
    return (h(M) - h(Lp)) / L_s + (L_pet - M) / L_s * pM
