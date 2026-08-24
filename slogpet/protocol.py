"""Choosing where to put the bed positions of a multi-bed acquisition.

The problem
-----------
A scanner of axial length ``L_pet`` has a sensitivity profile ``eta(z)`` that
peaks at its centre and falls to zero at its ends.  To cover a scan range longer
than that, the patient is moved through in ``n_beds`` steps of a constant
``spacing``, each acquired for the same fraction ``1/n_beds`` of the total time.
The resulting profile is the average of shifted copies,

    eta_N(z) = (1 / n_beds) * sum_n eta(z - z_n),
    z_n      = (n - (n_beds - 1) / 2) * spacing,      n = 0 ... n_beds - 1

which ripples: it is highest where copies overlap and lowest between them.  What
matters clinically is the worst place in the range, so the protocol is chosen by
maximising ``min eta_N`` over ``|z| <= scan_length / 2``, over both the number of
beds and their spacing.

Only ``eta_N`` depends on the protocol, so the answer is independent of the SLoG
size, of the acquisition time and of the detector resolutions.  That is why it is
worth computing once and reusing.

How this module is built
------------------------
Five pieces, in the order one would derive them:

    1.  sample_single_bed_profile   eta(z) for one bed, tabulated on a grid
    2.  SampledProfile              that grid, and how to read values off it
    3.  tile_beds                   n_beds shifted copies, averaged
    4.  best_spacing_for_n_beds     the best spacing, for a given number of beds
    5.  optimise_bed_positions      the best number of beds

plus ``coverage``, which reports the minimum, mean and maximum of ``eta_N`` over
the scan range once an arrangement has been chosen.

Why a grid, and why bed spacings are multiples of 2 mm
------------------------------------------------------
Everything below works on a fixed grid of step ``PROFILE_STEP_MM``, and the bed
spacing is restricted to a whole number of *double* steps.  That restriction buys
two things.

*Exactness.*  The bed positions ``z_n`` above are half-integer multiples of the
spacing when ``n_beds`` is even.  Writing ``spacing = 2 * m * step`` makes

    z_n = a_n * m * step        with     a_n = 2n - (n_beds - 1)

and ``a_n`` is an integer for either parity, so every bed lands exactly on a grid
point, with no rounding.  Consequently ``eta_N`` is piecewise linear with *all*
of its knots on the grid, and a piecewise linear function attains its minimum
over an interval at a knot or at an end of the interval.  So evaluating ``eta_N``
at the grid points inside the range, plus the two ends, does not approximate the
minimum -- it finds it.

*Speed.*  Evaluating ``eta_N`` becomes an integer-indexed gather instead of an
interpolation, which is about ten times cheaper, and the whole bed search runs
some thirty times faster than the continuous search it replaced.

The price is that the spacing is quantised to 2 mm.  That is far below anything
clinically meaningful -- and far below the accuracy of the answer, since refining
the grid fourfold moves the minimum sensitivity by well under a per cent.

Lengths are in mm throughout.  Every function here is pure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt

from .geometry import MU_WATER, eta

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

__all__ = [
    "PROFILE_STEP_MM", "SEARCH_STEP_MM", "BED_COUNT_TOLERANCE",
    "SampledProfile", "SpacingChoice", "ScanCoverage", "BedArrangement",
    "sample_single_bed_profile", "profile_fwhm", "bed_positions", "tile_beds",
    "coverage", "best_spacing_for_n_beds", "optimise_bed_positions",
]

PROFILE_STEP_MM: float = 1.0
"""Grid on which eta is tabulated.  Bed spacings are multiples of twice this."""

SEARCH_STEP_MM: float = 10.0
"""Granularity of the exhaustive first pass over the bed spacing.  Only the
search is this coarse; the answer is refined to the grid step afterwards."""

BED_COUNT_TOLERANCE: float = 3e-2
"""How close to the best achievable minimum a smaller bed count may be and still
be preferred.  See optimise_bed_positions."""

_ZERO_PAD = 4
"""Zero samples kept beyond each end of the profile, so that an index clipped to
the array bounds returns zero rather than the value at the edge."""


# ============================================================ 1 and 2: the grid
@dataclass(frozen=True, eq=False)
class SampledProfile:
    """``eta(z)`` for a *single* bed position, tabulated on a uniform grid.

    Sample ``i`` sits at ``z = (i - origin_index) * step_mm``, so the grid always
    contains ``z = 0``.  ``values`` carries a few zeros beyond each end of the
    detector, which lets an out-of-range index be clipped into the array and
    still return zero -- no branching in the inner loop.

    Sample values are exact: each one is a quadrature evaluation of ``eta``, not
    an interpolation.  Interpolation only ever happens *between* samples, which
    is what ``__call__`` does, and which the bed search never needs.
    """

    step_mm: float
    origin_index: int
    values: FloatArray = field(repr=False)

    # -- reading values off the grid -----------------------------------------
    @property
    def z_mm(self) -> FloatArray:
        """The axial position of every sample."""
        return (np.arange(len(self.values)) - self.origin_index) * self.step_mm

    @property
    def integral_mm(self) -> float:
        """``int eta dz``.  Tiling beds conserves this: every bed contributes
        ``1/n_beds`` of the same profile, so a protocol redistributes counts, it
        does not create or destroy them."""
        return float(np.trapezoid(self.values, dx=self.step_mm))

    def __call__(self, z_mm: npt.ArrayLike) -> FloatArray:
        """``eta`` at arbitrary positions, by linear interpolation between
        samples.  For plotting and for reporting; the search does not use it."""
        return np.interp(np.asarray(z_mm, dtype=float), self.z_mm, self.values,
                         left=0.0, right=0.0)

    def index_of(self, z_mm: float) -> int:
        """The grid index nearest to a position."""
        return int(np.rint(z_mm / self.step_mm)) + self.origin_index

    def indices_within(self, half_width_mm: float) -> IntArray:
        """Indices of every grid point in ``|z| <= half_width_mm``."""
        k = int(np.floor(half_width_mm / self.step_mm))
        return np.arange(-k, k + 1) + self.origin_index

    def covers_exactly(self, half_width_mm: float) -> bool:
        """True if ``half_width_mm`` is itself a grid point, so that the grid
        points inside the range already include its ends."""
        k = np.floor(half_width_mm / self.step_mm)
        return bool(abs(k * self.step_mm - half_width_mm) < 1e-9)


def sample_single_bed_profile(
    L_pet_mm: float,
    D_pet_mm: float,
    D_cyl_mm: float,
    mu_per_mm: float = MU_WATER,
    L_mrd_mm: float = np.inf,
    step_mm: float = PROFILE_STEP_MM,
) -> SampledProfile:
    """Tabulate ``eta(z)`` for one bed position of one scanner in one object.

    The grid is anchored so that ``z = 0`` is a sample and spans the detector,
    which is where ``eta`` is non-zero.
    """
    half_width = int(np.floor(L_pet_mm / 2 / step_mm)) + 1
    k = np.arange(-half_width, half_width + 1)
    sampled = np.array([eta(index * step_mm, L_pet_mm, D_pet_mm, D_cyl_mm,
                            mu=mu_per_mm, L_mrd=L_mrd_mm) for index in k])
    padded = np.concatenate([np.zeros(_ZERO_PAD), sampled, np.zeros(_ZERO_PAD)])
    return SampledProfile(step_mm=step_mm,
                          origin_index=half_width + _ZERO_PAD,
                          values=padded)


def profile_fwhm(profile: SampledProfile) -> float:
    """Full width at half maximum of a single-bed profile, which peaks at z = 0.

    A useful rule of thumb: a scan range shorter than this needs only one bed.
    """
    half = profile.values[profile.origin_index:]
    z = np.arange(len(half)) * profile.step_mm
    return 2.0 * float(np.interp(-half[0] / 2.0, -half, z))


# =================================================== 3: overlapping n_beds beds
def bed_positions(n_beds: int, spacing_mm: float) -> FloatArray:
    """Where the beds sit, centred on z = 0."""
    return (np.arange(n_beds) - (n_beds - 1) / 2.0) * spacing_mm


def tile_beds(profile: SampledProfile, n_beds: int, spacing_mm: float,
              z_mm: npt.ArrayLike) -> FloatArray:
    """``eta_N`` at arbitrary positions: shifted copies of the profile, averaged.

    Interpolates, so it works for any spacing and any position.  This is the
    definition; the search below uses a faster equivalent restricted to the grid.
    """
    z = np.asarray(z_mm, dtype=float)
    return profile(z[..., None] - bed_positions(n_beds, spacing_mm)).mean(axis=-1)


def _bed_index_offsets(n_beds: int) -> IntArray:
    """``a_n = 2n - (n_beds - 1)``: twice each bed position in units of half the
    spacing.  Integer for both parities of ``n_beds`` -- see the module docstring."""
    return 2 * np.arange(n_beds) - (n_beds - 1)


def _tile_beds_on_grid(profile: SampledProfile, half_spacings: IntArray,
                       offsets: IntArray, z_indices: IntArray) -> FloatArray:
    """``eta_N`` at grid points, for a whole set of candidate spacings at once.

    ``half_spacings[j]`` is a spacing of ``2 * half_spacings[j] * step_mm``.  Bed
    ``n`` of candidate ``j`` is then shifted by exactly ``offsets[n] *
    half_spacings[j]`` grid steps, so this is a gather -- no interpolation, no
    rounding.  Returns shape ``(len(half_spacings), len(z_indices))``.
    """
    shifted = (z_indices[None, None, :]
               - offsets[None, :, None] * half_spacings[:, None, None])
    np.clip(shifted, 0, len(profile.values) - 1, out=shifted)
    return profile.values[shifted].mean(axis=1)


# ================================================= reporting on an arrangement
class SpacingChoice(NamedTuple):
    """The outcome of searching for the best spacing at a fixed bed count.

    A named tuple rather than a bare pair, so that a caller cannot silently get
    the two floats the wrong way round.
    """

    spacing_mm: float
    min_eta: float
    """The worst point of the scan range at that spacing.  This is what the bed
    count is chosen on; the mean and maximum are reported afterwards by
    ``coverage``, which is why they are not computed for every candidate."""


class ScanCoverage(NamedTuple):
    """How well an arrangement covers the scan range."""

    min_eta: float
    """The worst point in the range -- what the protocol is chosen to maximise."""
    mean_eta: float
    """``(1 / S) int eta_N dz`` over the range: the average sensitivity, which is
    what governs the total number of counts collected."""
    max_eta: float
    """The best point in the range, at the crest of the ripple."""

    @property
    def ripple(self) -> float:
        """Peak-to-trough spread relative to the mean.  Zero would be a perfectly
        flat profile across the range."""
        return (self.max_eta - self.min_eta) / self.mean_eta if self.mean_eta else np.nan


class BedArrangement(NamedTuple):
    """A chosen protocol, and how it performs."""

    n_beds: int
    spacing_mm: float
    coverage: ScanCoverage

    def overlap_percent(self, L_pet_mm: float) -> Optional[float]:
        """Bed overlap ``1 - spacing / L_pet``, which is what a scanner console
        asks for.  None for a single bed, where overlap is meaningless."""
        if self.n_beds == 1:
            return None
        return 100.0 * (1.0 - self.spacing_mm / L_pet_mm)


def coverage(profile: SampledProfile, n_beds: int, spacing_mm: float,
             scan_length_mm: float) -> ScanCoverage:
    """Minimum, mean and maximum of ``eta_N`` over ``|z| <= scan_length / 2``.

    Evaluated at every grid point in the range plus its two ends.  Because the
    bed positions are on the grid, that set is guaranteed to contain the true
    minimum and maximum, so these are exact rather than sampled.
    """
    half = scan_length_mm / 2.0
    z_indices = profile.indices_within(half)
    z = (z_indices - profile.origin_index) * profile.step_mm

    # The normal case: the spacing came from the search, so it is a whole number
    # of double steps and the values can be gathered rather than interpolated.
    half_steps = spacing_mm / (2.0 * profile.step_mm)
    if abs(half_steps - round(half_steps)) < 1e-9:
        values = _tile_beds_on_grid(profile, np.array([int(round(half_steps))]),
                                    _bed_index_offsets(n_beds), z_indices)[0]
    else:                                   # an arbitrary spacing, from a caller
        values = tile_beds(profile, n_beds, spacing_mm, z)

    if not profile.covers_exactly(half):
        ends = tile_beds(profile, n_beds, spacing_mm, np.array([-half, half]))
        z = np.concatenate([[-half], z, [half]])
        values = np.concatenate([[ends[0]], values, [ends[1]]])

    return ScanCoverage(min_eta=float(values.min()),
                        mean_eta=float(np.trapezoid(values, z) / scan_length_mm),
                        max_eta=float(values.max()))


# ============================== 4: the best spacing for a given number of beds
def _worst_point_of_each_candidate(
    profile: SampledProfile, half_spacings: IntArray, offsets: IntArray,
    z_indices: IntArray, half_scan_mm: float, coarsening: int,
) -> FloatArray:
    """``min eta_N`` over the scan range, for every candidate spacing.

    The minimum is localised rather than searched exhaustively in ``z``: the
    tiled profile is first evaluated on every ``coarsening``-th grid point, and
    only the neighbourhoods of its three deepest points are then examined at full
    resolution.  ``eta_N`` ripples with a period of roughly the bed spacing, so
    three troughs is ample, and this is exact in every case tested.
    """
    coarse = _tile_beds_on_grid(profile, half_spacings, offsets,
                                z_indices[::coarsening])
    deepest = np.argsort(coarse, axis=1)[:, :3]
    window = np.arange(-coarsening, coarsening + 1)
    around = (deepest[:, :, None] * coarsening + window).reshape(len(half_spacings), -1)
    nearby = np.unique(np.clip(around, 0, len(z_indices) - 1))
    worst = _tile_beds_on_grid(profile, half_spacings, offsets,
                               z_indices[nearby]).min(axis=1)

    # The ends of the scan range are not grid points in general, and a piecewise
    # linear function can take its minimum over a closed interval at an end.
    if not profile.covers_exactly(half_scan_mm):
        spacings = 2.0 * half_spacings * profile.step_mm
        beds = bed_positions(len(offsets), 1.0)[None, :] * spacings[:, None]
        for end in (-half_scan_mm, half_scan_mm):
            worst = np.minimum(worst, profile(end - beds).mean(axis=1))
    return worst


def best_spacing_for_n_beds(
    profile: SampledProfile,
    L_pet_mm: float,
    scan_length_mm: float,
    n_beds: int,
    search_step_mm: float = SEARCH_STEP_MM,
) -> SpacingChoice:
    """The spacing that maximises ``min eta_N``, for exactly ``n_beds`` beds.

    Every bed gets the same acquisition time and the same spacing -- equivalently
    a single constant overlap, which is what a scanner console lets one set.

    The search is two-stage.  First every candidate spacing is tried at
    ``search_step_mm`` granularity: exhaustive, so it cannot converge on the
    wrong local optimum, which a hill-climb over this rippling objective easily
    would.  Then the winner is refined to the grid step within one coarse step
    either side.  Both stages are needed: the objective is piecewise linear in
    the spacing, so its maximum sits at a kink and being 5 mm off is a
    first-order error, not a second-order one -- skipping the refinement costs
    about 0.4 per cent at the median.
    """
    half_scan = scan_length_mm / 2.0
    z_indices = profile.indices_within(half_scan)
    offsets = _bed_index_offsets(n_beds)
    z_coarsening = max(1, int(round(search_step_mm / profile.step_mm)))

    if n_beds == 1:
        # Only one arrangement is possible, so there is nothing to search -- but
        # it still goes through the same evaluation, which is what makes sure the
        # ends of the scan range are considered.  With a single bed the minimum
        # is usually AT an end, so skipping that check overstates it.
        candidates = np.array([0])
    else:
        # Beyond this the beds no longer overlap the range at all.
        max_half_spacing = int((L_pet_mm + scan_length_mm) / (n_beds - 1)
                               / (2 * profile.step_mm))
        coarse_stride = max(1, int(round(search_step_mm / (2 * profile.step_mm))))

        # -- stage one: every candidate, on a coarse z grid
        coarse = np.arange(0, max_half_spacing + 1, coarse_stride)
        coarse_worst = _tile_beds_on_grid(profile, coarse, offsets,
                                          z_indices[::z_coarsening]).min(axis=1)
        winner = int(np.argmax(coarse_worst))

        # -- stage two: refine around the winner, at the grid step
        candidates = np.arange(
            max(0, coarse[winner] - coarse_stride),
            min(max_half_spacing, coarse[winner] + coarse_stride) + 1)

    worst = _worst_point_of_each_candidate(profile, candidates, offsets, z_indices,
                                           half_scan, z_coarsening)
    best = int(np.argmax(worst))
    return SpacingChoice(spacing_mm=float(2 * candidates[best] * profile.step_mm),
                         min_eta=float(worst[best]))


# ======================================== 5: the best number of bed positions
def _most_beds_worth_trying(L_pet_mm: float, scan_length_mm: float) -> int:
    """Above this, extra beds only dilute the time each one gets."""
    return min(int(2 * scan_length_mm / L_pet_mm) + 4, 30)


def optimise_bed_positions(
    profile: SampledProfile,
    L_pet_mm: float,
    scan_length_mm: float,
    max_beds: Optional[int] = None,
    tolerance: float = BED_COUNT_TOLERANCE,
) -> BedArrangement:
    """The arrangement that maximises the worst point of the scan range.

    Each bed count is given its own best spacing, and the best of those is the
    achievable optimum.  The optimum in the bed count is very flat, though, so
    what is returned is the *smallest* bed count reaching within ``tolerance``
    of it.  That is the clinically sensible tie-break, since fewer beds mean less
    transport overhead, and it makes the reported count monotone in the scan
    length -- a plain argmax jumps around, because the curves for different bed
    counts cross each other.

    Which of several near-equal counts is picked hardly matters: what is being
    compared between scanners is the minimum sensitivity, and every candidate
    within ``tolerance`` delivers that to within ``tolerance``.
    """
    if max_beds is None:
        max_beds = _most_beds_worth_trying(L_pet_mm, scan_length_mm)

    searched = [best_spacing_for_n_beds(profile, L_pet_mm, scan_length_mm, n)
                for n in range(1, max_beds + 1)]
    achievable = max(choice.min_eta for choice in searched)

    for n_beds, choice in enumerate(searched, start=1):
        if choice.min_eta >= (1.0 - tolerance) * achievable:
            break
    else:                                        # pragma: no cover - unreachable
        n_beds, choice = 1, searched[0]

    return BedArrangement(
        n_beds=n_beds, spacing_mm=choice.spacing_mm,
        coverage=coverage(profile, n_beds, choice.spacing_mm, scan_length_mm))
