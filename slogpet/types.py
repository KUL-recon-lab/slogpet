"""The parameter objects: what a system is, what the task is, what a protocol is.

Everything a calculation needs is carried by small frozen dataclasses, so that a
scanner can be passed around, hashed, cached, printed, serialised and compared
without any of the code having to remember the order of nine positional
arguments.

Units: lengths in mm, coincidence timings in ps, all resolutions as FWHM.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import MU_WATER, S_ideal_closed
from .protocol import SampledProfile, ScanCoverage
from .resolution import FWHM, C_OVER_2, r_of

__all__ = ["Scanner", "Task", "Protocol", "AxialProfile", "SNRResult", "L_S_NEMA"]

L_S_NEMA = 700.0        # mm, the NEMA NU 2 line source


@dataclass(frozen=True)
class Scanner:
    """One PET system, or one hypothetical configuration of one.

    The three resolutions that enter the task are ``F_t`` (along the
    time-of-flight direction), ``F_y`` (transaxial) and ``F_z`` (axial).  ``F_t``
    may be given directly in mm, or as a coincidence time resolution ``ctr`` in
    ps, in which case ``F_t = 0.15 mm/ps x ctr``.  ``F_t is None`` means the
    system has no time of flight; the resolution factor is then evaluated in its
    non-TOF form, which depends on the object.

    The detector-pair efficiency ``epsilon`` may be given directly or left to be
    derived from a published NEMA sensitivity via ``S_nema``.

    The remaining fields are descriptive: they are what the table in the paper
    prints, and they carry the provenance of the numbers.
    """
    name: str
    L_pet: float                                  # axial length of the detector
    D_pet: float                                  # detector ring diameter
    F_y: Optional[float] = None                   # transaxial resolution
    F_z: Optional[float] = None                   # axial resolution
    F_t: Optional[float] = None                   # TOF resolution, mm
    ctr: Optional[float] = None                   # coincidence time resolution, ps
    L_mrd: float = np.inf                         # maximum ring difference, mm
    epsilon: Optional[float] = None               # detector-pair efficiency
    S_nema: Optional[float] = None                # NEMA sensitivity, cps/kBq
    crystal: str = ""                             # scintillator, e.g. "LSO"
    crystal_size: Optional[Sequence[float]] = None    # mm, (a, b, depth)
    energy_resolution: Optional[float] = None     # % FWHM at 511 keV
    energy_window: Optional[Sequence[float]] = None   # keV, (low, high)
    reference: Union[int, str] = ""               # key into the reference list
    assumed: Sequence[str] = ()                   # fields not measured for this system
    note: str = ""

    def __post_init__(self):
        if self.F_t is None and self.ctr is not None:
            object.__setattr__(self, "F_t", C_OVER_2 * self.ctr)
        # sequences must be tuples, or the dataclass is not hashable and the
        # profile cache cannot key on it
        for f in ("crystal_size", "energy_window", "assumed"):
            v = getattr(self, f)
            if isinstance(v, list):
                object.__setattr__(self, f, tuple(v))

    @property
    def has_tof(self) -> bool:
        return self.F_t is not None

    def S_ideal(self, L_s: float = L_S_NEMA) -> float:
        """Sensitivity of an ideal detector to the NEMA line source."""
        return S_ideal_closed(self.L_pet, self.D_pet, L_s, self.L_mrd)

    def efficiency(self, L_s: float = L_S_NEMA) -> Optional[float]:
        """The detector-pair efficiency: as given, or as implied by ``S_nema``."""
        if self.epsilon is not None:
            return self.epsilon
        if self.S_nema is None:
            return None
        return self.S_nema / (1000.0 * self.S_ideal(L_s))

    def is_assumed(self, field_name: str) -> bool:
        """True if this field was taken from a similar system rather than measured."""
        return field_name in self.assumed

    def r(self, task: "Task") -> float:
        """Resolution factor for this task; the non-TOF form if the system has no
        time of flight, in which case it depends on the object diameter."""
        return r_of(self.F_t, self.F_y, self.F_z, task.F_o, task.D_cyl)


@dataclass(frozen=True)
class Task:
    """The detection task: a SLoG of size ``F_o`` in a water cylinder of
    diameter ``D_cyl``."""
    F_o: float                      # SLoG size, mm FWHM
    D_cyl: float                    # object diameter, mm
    mu: float = MU_WATER            # attenuation coefficient, mm^-1

    @property
    def sigma_o(self) -> float:
        return self.F_o / FWHM


@dataclass(frozen=True)
class Protocol:
    """A multi-bed acquisition: ``n_beds`` positions at a constant ``spacing``,
    each acquired for the same fraction of the total time."""
    scan_length: float              # S, mm
    n_beds: int
    spacing: float                  # mm; 0 for a single bed
    min_eta: float                  # min over |z| <= S/2 of eta_N
    mean_eta: float = np.nan        # (1/S) int eta_N dz over the scan range
    max_eta: float = np.nan         # max over the scan range
    L_pet: float = np.nan           # kept so the overlap can be expressed

    @property
    def overlap(self) -> Optional[float]:
        """Bed overlap in per cent, or None for a single-bed acquisition."""
        return None if self.n_beds == 1 else 100.0 * (1.0 - self.spacing / self.L_pet)

    @property
    def coverage(self) -> ScanCoverage:
        """The three statistics together."""
        return ScanCoverage(self.min_eta, self.mean_eta, self.max_eta)


@dataclass(frozen=True)
class AxialProfile:
    """``eta(z)`` for one scanner in one object, tabulated on a lattice.

    It does not depend on the SLoG size, on the acquisition time or on the
    detector resolutions, which is why it is worth computing once and reusing it
    for every scan length.  The lattice is what the bed search runs on; calling
    the profile interpolates between its samples, which is for plotting.
    """
    scanner: Scanner
    D_cyl: float
    mu: float
    samples: SampledProfile = field(compare=False)
    integral: float = np.nan             # int eta dz, conserved when beds are tiled

    def __call__(self, z):
        return self.samples(z)

    @property
    def step_mm(self) -> float:
        """Spacing of the grid the profile is tabulated on."""
        return self.samples.step_mm

    @property
    def z(self):
        return self.samples.z_mm

    @property
    def values(self):
        return self.samples.values


@dataclass
class SNRResult:
    """Everything one (scanner, task, scan length) triple produces."""
    scanner: Scanner
    task: Task
    protocol: Protocol
    epsilon: float
    r: float
    snr2_min: float                 # minimum over the scan range
    z: np.ndarray                   # mm
    eta_N: np.ndarray
    snr2: np.ndarray                # same units as snr2_min

    @property
    def n_beds(self) -> int:
        return self.protocol.n_beds

    @property
    def overlap(self) -> Optional[float]:
        return self.protocol.overlap
