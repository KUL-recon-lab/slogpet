"""Assembly: from a scanner, a task and a scan length to an SNR^2 profile.

The Hotelling SNR^2 for detecting a SLoG factorises as

    SNR^2(z) = T x [ Sdot^2 sigma_o^3 / (16 pi sqrt(pi) Bdot) ] x epsilon eta_N(z) x r

The first two factors are properties of the patient and of the acquisition time,
identical for every system compared; the last two are the system.  This module
computes the system part,

    snr2(z) = epsilon x eta_N(z) x r x sigma_o^3

which is what the paper plots, in units of ``T Sdot^2 / (16 pi sqrt(pi) Bdot)``.
Pass ``scale`` to put an absolute prefactor back in.

The entry points are

    axial_profile(scanner, D_cyl)            -> AxialProfile   (cached)
    optimal_protocol(profile, scan_length)   -> Protocol
    snr2(scanner, task, scan_length)         -> SNRResult
    snr2_curve(scanner, task, scan_lengths)  -> list[SNRResult]

``axial_profile`` is the expensive step and is memoised on its arguments, which
is what keeps an interactive frontend responsive: changing the SLoG size, the
scan length or the acquisition time reuses it.
"""
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .geometry import MU_WATER
from .protocol import (PROFILE_STEP_MM, best_spacing_for_n_beds, coverage,
                       optimise_bed_positions, sample_single_bed_profile,
                       tile_beds)
from .resolution import FWHM
from .types import AxialProfile, Protocol, SNRResult, Scanner, Task

__all__ = ["axial_profile", "optimal_protocol", "protocol_for_N", "snr2",
           "snr2_curve", "snr2_value"]

N_Z = 401               # points at which the multi-bed profile is reported


@lru_cache(maxsize=256)
def axial_profile(scanner: Scanner, D_cyl: float, mu: float = MU_WATER,
                  step_mm: float = PROFILE_STEP_MM) -> AxialProfile:
    """``eta(z)`` for one scanner in a water cylinder of diameter ``D_cyl``.

    Memoised: the result depends on neither the SLoG size nor the scan length,
    so a whole family of tasks and protocols reuses one profile.
    """
    samples = sample_single_bed_profile(scanner.L_pet, scanner.D_pet, D_cyl,
                                        mu_per_mm=mu, L_mrd_mm=scanner.L_mrd,
                                        step_mm=step_mm)
    return AxialProfile(scanner=scanner, D_cyl=D_cyl, mu=mu, samples=samples,
                        integral=samples.integral_mm)


@lru_cache(maxsize=4096)
def optimal_protocol(profile: AxialProfile, scan_length: float,
                     Nmax: Optional[int] = None, tol: float = 3e-2) -> Protocol:
    """Bed positions maximising the minimum of ``eta_N`` over ``|z| <= S/2``.

    Memoised as well: this is the expensive step once the profile exists, and an
    interactive frontend revisits the same scan lengths constantly."""
    L = profile.scanner.L_pet
    best = optimise_bed_positions(profile.samples, L, scan_length,
                                  max_beds=Nmax, tolerance=tol)
    return Protocol(scan_length=scan_length, n_beds=best.n_beds,
                    spacing=best.spacing_mm, min_eta=best.coverage.min_eta,
                    mean_eta=best.coverage.mean_eta,
                    max_eta=best.coverage.max_eta, L_pet=L)


def protocol_for_N(profile: AxialProfile, scan_length: float, N: int) -> Protocol:
    """The best protocol with exactly ``N`` bed positions."""
    L = profile.scanner.L_pet
    choice = best_spacing_for_n_beds(profile.samples, L, scan_length, N)
    spacing = choice.spacing_mm
    stats = coverage(profile.samples, N, spacing, scan_length)
    return Protocol(scan_length=scan_length, n_beds=N, spacing=spacing,
                    min_eta=stats.min_eta, mean_eta=stats.mean_eta,
                    max_eta=stats.max_eta, L_pet=L)


def snr2_value(epsilon: float, eta_value, r: float, F_o: float, scale: float = 1.0):
    """The system part of the SNR^2: ``epsilon x eta x r x sigma_o^3``."""
    return scale * epsilon * eta_value * r * (F_o / FWHM) ** 3


def _eta_N(profile: AxialProfile, protocol: Protocol, z) -> np.ndarray:
    """The tiled profile ``eta_N(z) = (1/N) sum_n eta(z - z_n)``."""
    return tile_beds(profile.samples, protocol.n_beds, protocol.spacing, z)


def snr2(scanner: Scanner, task: Task, scan_length: float,
         profile: Optional[AxialProfile] = None,
         protocol: Optional[Protocol] = None,
         n_beds: Optional[int] = None,
         L_s: float = 700.0, scale: float = 1.0, nz: int = N_Z) -> SNRResult:
    """SNR^2 over the scan range for one system, one task and one scan length.

    ``profile`` and ``protocol`` may be supplied to avoid recomputing them.  If
    ``n_beds`` is given the protocol is forced to that many bed positions instead
    of being optimised.
    """
    eps = scanner.efficiency(L_s)
    if eps is None:
        raise ValueError(f"{scanner.name}: neither epsilon nor S_nema is known")
    if profile is None:
        profile = axial_profile(scanner, task.D_cyl, task.mu)
    if protocol is None:
        protocol = (protocol_for_N(profile, scan_length, n_beds) if n_beds
                    else optimal_protocol(profile, scan_length))
    r = scanner.r(task)
    z = np.linspace(-scan_length / 2.0, scan_length / 2.0, nz)
    e_N = _eta_N(profile, protocol, z)
    return SNRResult(
        scanner=scanner, task=task, protocol=protocol, epsilon=eps, r=r,
        snr2_min=snr2_value(eps, protocol.min_eta, r, task.F_o, scale),
        z=z, eta_N=e_N,
        snr2=snr2_value(eps, e_N, r, task.F_o, scale))


def snr2_curve(scanner: Scanner, task: Task, scan_lengths: Sequence[float],
               **kw) -> List[SNRResult]:
    """``snr2`` for a series of scan lengths, building the profile only once."""
    profile = axial_profile(scanner, task.D_cyl, task.mu)
    return [snr2(scanner, task, S, profile=profile, **kw) for S in scan_lengths]
