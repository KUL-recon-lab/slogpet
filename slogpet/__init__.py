"""slogpet -- SLoG detectability of cylindrical PET systems.

Pure, side-effect-free implementations of the model in

    J. Nuyts, S. A. Zamanpour, G. Schramm,
    "Comparing short and long PET systems on a resolution-dependent task".

Layout
------
``slogpet.geometry``    axial coverage ``eta(z)`` and ``S_ideal``
``slogpet.resolution``  the resolution factor ``r`` with and without TOF
``slogpet.protocol``    optimisation of the multi-bed acquisition
``slogpet.types``       Scanner / Task / Protocol / AxialProfile / SNRResult
``slogpet.snr``         the assembly: (scanner, task, scan length) -> SNR^2
``slogpet.validation``  Monte Carlo and numerical cross-checks

Nothing here imports matplotlib, writes files or holds mutable module state:
every geometry is passed in explicitly.  Lengths are in mm, coincidence timings
in ps, resolutions as FWHM.
"""
from .geometry import MU_WATER, d1, u1, eta, S_ideal_quad, S_ideal_closed
from .resolution import (FWHM, C_OVER_2, R_MAX, w, ctr_to_mm, r_tof, r_nontof,
                         r_of, F_t_equiv)
from .protocol import (H_LATTICE, H_SEARCH, Lattice, eta_lattice, fwhm_of,
                       lattice_integral, tiled_profile, best_for_N, optimise_beds)
from .types import Scanner, Task, Protocol, AxialProfile, SNRResult, L_S_NEMA
from .snr import (axial_profile, optimal_protocol, protocol_for_N, snr2,
                  snr2_curve, snr2_value)

__version__ = "0.1.0"

__all__ = [
    "MU_WATER", "d1", "u1", "eta", "S_ideal_quad", "S_ideal_closed",
    "FWHM", "C_OVER_2", "R_MAX", "w", "ctr_to_mm", "r_tof", "r_nontof", "r_of",
    "F_t_equiv", "H_LATTICE", "H_SEARCH", "Lattice", "eta_lattice", "fwhm_of",
    "lattice_integral", "tiled_profile", "best_for_N", "optimise_beds",
    "Scanner", "Task", "Protocol", "AxialProfile", "SNRResult", "L_S_NEMA",
    "axial_profile", "optimal_protocol", "protocol_for_N", "snr2", "snr2_curve",
    "snr2_value",
]
