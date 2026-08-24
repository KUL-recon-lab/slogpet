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
    "coverage", "best_spacing_for_n_beds", "flattest_spacing_for_n_beds",
    "optimise_bed_positions",
]

PROFILE_STEP_MM: float = 1.0
"""Grid on which eta is tabulated.  Bed spacings are multiples of twice this."""

SEARCH_STEP_MM: float = 10.0
"""Granularity of the exhaustive first pass over the bed spacing.  Only the
search is this coarse; the answer is refined to the grid step afterwards.  It is
an absolute length, so on a short detector it is capped -- see
MIN_COARSE_SPACINGS."""

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
    count is chosen on; the mean is reported afterwards by ``coverage``, which is
    why it is not computed for every candidate."""
    peak_to_trough: float
    """``max / min`` for that arrangement -- how uneven it is."""


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
    def peak_to_trough(self) -> float:
        """``max / min``: how many times more sensitive the best point of the
        range is than the worst.  One would be perfectly flat; infinite means
        part of the range is not covered at all."""
        return self.max_eta / self.min_eta if self.min_eta > 0 else np.inf

    @property
    def ripple(self) -> float:
        """Peak-to-trough spread relative to the mean.  Unlike ``peak_to_trough``
        this stays finite when part of the range is uncovered, which makes it the
        better choice for a plot that has to include that case."""
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


def _candidate_half_spacings(profile: SampledProfile, L_pet_mm: float,
                             scan_length_mm: float, n_beds: int) -> IntArray:
    """Every spacing worth trying, in units of half a grid step.  Beyond the last
    one the beds no longer overlap the scan range at all."""
    if n_beds == 1:
        return np.array([0])
    return np.arange(0, int((L_pet_mm + scan_length_mm) / (n_beds - 1)
                            / (2 * profile.step_mm)) + 1)


def _support_half_width(profile: SampledProfile) -> float:
    """Half the width of the region where one bed contributes anything at all.

    Rounded up to the next grid point, so that it is never an underestimate.
    """
    non_zero = np.flatnonzero(profile.values > 0.0)
    if non_zero.size == 0:
        return 0.0
    reach = max(abs(int(non_zero[0]) - profile.origin_index),
                abs(int(non_zero[-1]) - profile.origin_index)) + 1
    return reach * profile.step_mm


def _covering_half_spacings(profile: SampledProfile, scan_length_mm: float,
                            n_beds: int) -> IntArray:
    """The spacings that could possibly cover the range, in half grid steps.

    With ``h`` the half-width of one bed's support, a spacing ``d`` leaves no
    part of the range uncovered only if the outermost bed still reaches the end
    of it and neighbouring beds still overlap each other:

        (n_beds - 1) * d / 2 + h >= scan_length / 2     and     d <= 2 h

    Both are necessary, neither is sufficient, so this is a bracket to search
    within rather than an answer.  It comes back empty when the two cannot hold
    at once, which is the honest statement that this many beds cannot cover this
    range at any spacing.
    """
    h = _support_half_width(profile)
    if n_beds == 1:
        return np.array([0]) if 2 * h >= scan_length_mm else np.empty(0, dtype=int)
    unit = 2.0 * profile.step_mm
    lowest = max(0.0, (scan_length_mm - 2 * h) / (n_beds - 1))
    first = int(np.ceil(lowest / unit - 1e-9))
    last = int(np.floor(2 * h / unit + 1e-9))
    return np.arange(first, last + 1) if last >= first else np.empty(0, dtype=int)


def _scan_every_spacing(profile: SampledProfile, L_pet_mm: float,
                        scan_length_mm: float, n_beds: int,
                        chunk: int = 256,
                        candidates: Optional[IntArray] = None,
                        ) -> tuple[IntArray, FloatArray, FloatArray]:
    """The minimum AND maximum over the scan range, for every candidate spacing.

    The fast search of ``best_spacing_for_n_beds`` only ever needs the minimum,
    and localises it to a few troughs.  A limit on the ripple needs the maximum
    too, and the crest can be anywhere in the range, so this evaluates the whole
    range for every candidate -- in chunks, to keep the working array bounded.

    Returns ``(half_spacings, min_eta, max_eta)``.
    """
    half_scan = scan_length_mm / 2.0
    z_indices = profile.indices_within(half_scan)
    offsets = _bed_index_offsets(n_beds)
    if candidates is None:
        candidates = _candidate_half_spacings(profile, L_pet_mm, scan_length_mm,
                                              n_beds)
    include_ends = not profile.covers_exactly(half_scan)

    lows, highs = [], []
    for start in range(0, len(candidates), chunk):
        block = candidates[start:start + chunk]
        tiled = _tile_beds_on_grid(profile, block, offsets, z_indices)
        low, high = tiled.min(axis=1), tiled.max(axis=1)
        if include_ends:
            beds = (bed_positions(n_beds, 1.0)[None, :]
                    * (2.0 * block * profile.step_mm)[:, None])
            for end in (-half_scan, half_scan):
                at_end = profile(end - beds).mean(axis=1)
                low = np.minimum(low, at_end)
                high = np.maximum(high, at_end)
        lows.append(low)
        highs.append(high)
    return candidates, np.concatenate(lows), np.concatenate(highs)


def flattest_spacing_for_n_beds(profile: SampledProfile, L_pet_mm: float,
                                scan_length_mm: float,
                                n_beds: int) -> Optional[SpacingChoice]:
    """The spacing that makes the profile as even as it can be, for this many beds.

    The counterpart of ``best_spacing_for_n_beds``: that one maximises the worst
    point regardless of how uneven the result is, this one minimises
    ``max / min`` regardless of how low the worst point ends up.  Returns None if
    no spacing covers the range at all.
    """
    spacings, low, high = _scan_every_spacing(profile, L_pet_mm, scan_length_mm, n_beds)
    covered = low > 0
    if not covered.any():
        return None
    ratio = np.where(covered, high / np.where(covered, low, 1.0), np.inf)
    best = int(np.argmin(ratio))
    return SpacingChoice(spacing_mm=float(2 * spacings[best] * profile.step_mm),
                         min_eta=float(low[best]), peak_to_trough=float(ratio[best]))


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


#: Fewest spacings the coarse pass may try before it is allowed to call a
#: winner.  Below this it cannot tell one basin of the rippling objective from
#: another, so the stride is shortened instead.
MIN_COARSE_SPACINGS: int = 12


def _coarse_stride(search_step_mm: float, step_mm: float,
                   max_half_spacing: int) -> int:
    """How far apart the coarse pass puts its candidates, in half grid steps.

    ``search_step_mm`` is an absolute length, chosen for detectors tens of
    centimetres long.  On a short one the whole range of useful spacings is only
    a few of those steps wide, which would leave two or three candidates to
    choose between, so the stride is capped to keep at least
    ``MIN_COARSE_SPACINGS`` of them.
    """
    stride = max(1, int(round(search_step_mm / (2 * step_mm))))
    return max(1, min(stride, max_half_spacing // MIN_COARSE_SPACINGS))


_NOTHING_COVERS = SpacingChoice(spacing_mm=0.0, min_eta=0.0, peak_to_trough=np.inf)


def _best_of_covering_spacings(profile: SampledProfile, L_pet_mm: float,
                               scan_length_mm: float, n_beds: int) -> SpacingChoice:
    """The best spacing among those that could cover the range, by trying each.

    Used when the coarse pass finds nothing that covers: the bracket above is
    usually empty (this many beds simply cannot reach), and where it is not, it
    is narrow enough to search outright.
    """
    candidates = _covering_half_spacings(profile, scan_length_mm, n_beds)
    if candidates.size == 0:
        return _NOTHING_COVERS
    spacings, low, high = _scan_every_spacing(profile, L_pet_mm, scan_length_mm,
                                              n_beds, candidates=candidates)
    best = int(np.argmax(low))
    if low[best] <= 0.0:
        return _NOTHING_COVERS
    return SpacingChoice(
        spacing_mm=float(2 * spacings[best] * profile.step_mm),
        min_eta=float(low[best]), peak_to_trough=float(high[best] / low[best]))


def best_spacing_for_n_beds(
    profile: SampledProfile,
    L_pet_mm: float,
    scan_length_mm: float,
    n_beds: int,
    search_step_mm: float = SEARCH_STEP_MM,
    max_peak_to_trough: Optional[float] = None,
) -> Optional[SpacingChoice]:
    """The spacing that maximises ``min eta_N``, for exactly ``n_beds`` beds.

    ``max_peak_to_trough`` caps how uneven the result may be: pass 1.2 to insist
    that the best point of the range is at most 20 per cent above the worst.  The
    default, None, imposes no limit.  Returns None if no spacing with this many
    beds can meet the limit.

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

    settled: Optional[SpacingChoice] = None
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
        coarse_stride = _coarse_stride(search_step_mm, profile.step_mm,
                                       max_half_spacing)

        # -- stage one: every candidate, on a coarse z grid
        coarse = np.arange(0, max_half_spacing + 1, coarse_stride)
        coarse_worst = _tile_beds_on_grid(profile, coarse, offsets,
                                          z_indices[::z_coarsening]).min(axis=1)
        winner = int(np.argmax(coarse_worst))

        if coarse_worst[winner] <= 0.0:
            # Not one coarse spacing reaches the whole range.  That may be the
            # truth -- too few beds for this range -- but it may also be a miss:
            # the spacings that DO cover form an interval bounded below by the
            # beds reaching the ends and above by the gaps opening up between
            # them, and that interval can be narrower than one coarse step.  A
            # tie at zero also tells argmax nothing, so it would return the
            # first candidate, a spacing of zero, all beds on top of each other.
            # Settle it by trying every spacing.
            settled = _best_of_covering_spacings(profile, L_pet_mm,
                                                 scan_length_mm, n_beds)

        # -- stage two: refine around the winner, at the grid step
        candidates = np.arange(
            max(0, coarse[winner] - coarse_stride),
            min(max_half_spacing, coarse[winner] + coarse_stride) + 1)

    if settled is not None:
        unlimited = settled
    else:
        worst = _worst_point_of_each_candidate(profile, candidates, offsets,
                                               z_indices, half_scan, z_coarsening)
        best = int(np.argmax(worst))
        spacing = float(2 * candidates[best] * profile.step_mm)
        ratio = coverage(profile, n_beds, spacing, scan_length_mm).peak_to_trough
        unlimited = SpacingChoice(spacing_mm=spacing, min_eta=float(worst[best]),
                                  peak_to_trough=ratio)

    if max_peak_to_trough is None or unlimited.peak_to_trough <= max_peak_to_trough:
        # Nothing more to do.  This spacing maximises the minimum over ALL
        # spacings, so if it also meets the limit it is the constrained optimum.
        return unlimited

    # It does not, so the limit binds and a different spacing has to be found:
    # one that gives up some of the minimum to even the profile out.
    return _best_within_ripple_limit(profile, L_pet_mm, scan_length_mm, n_beds,
                                     max_peak_to_trough)


def _best_within_ripple_limit(profile: SampledProfile, L_pet_mm: float,
                              scan_length_mm: float, n_beds: int,
                              max_peak_to_trough: float) -> Optional[SpacingChoice]:
    """The largest minimum reachable without exceeding the ripple limit.

    Every spacing is scanned rather than searched: the set that satisfies the
    limit need not be a single interval, so a coarse-then-refine search could
    step straight over it.
    """
    spacings, low, high = _scan_every_spacing(profile, L_pet_mm, scan_length_mm, n_beds)
    covered = low > 0
    ratio = np.where(covered, high / np.where(covered, low, 1.0), np.inf)
    allowed = covered & (ratio <= max_peak_to_trough)
    if not allowed.any():
        return None
    best = int(np.argmax(np.where(allowed, low, -np.inf)))
    return SpacingChoice(spacing_mm=float(2 * spacings[best] * profile.step_mm),
                         min_eta=float(low[best]), peak_to_trough=float(ratio[best]))


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
    max_peak_to_trough: Optional[float] = None,
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

    ``max_peak_to_trough`` trades sensitivity for uniformity.  By default there is
    no limit and the profile is allowed to ripple as much as it likes, which is
    what maximises the worst point.  Setting it to, say, 1.2 insists that the best
    point of the range is at most 20 per cent above the worst; each bed count is
    then given the best spacing that respects the limit -- which may not be the
    one that maximises its minimum -- and counts that cannot respect it at all are
    dropped.  A limit no arrangement can meet raises ``ValueError`` rather than
    quietly returning one that violates it.
    """
    if max_beds is None:
        max_beds = _most_beds_worth_trying(L_pet_mm, scan_length_mm)

    counts = range(1, max_beds + 1)
    searched = [(n, best_spacing_for_n_beds(profile, L_pet_mm, scan_length_mm, n,
                                            max_peak_to_trough=max_peak_to_trough))
                for n in counts]
    feasible = [(n, choice) for n, choice in searched if choice is not None]
    if not feasible:
        flattest = [flattest_spacing_for_n_beds(profile, L_pet_mm, scan_length_mm, n)
                    for n in counts]
        reachable = min((c.peak_to_trough for c in flattest if c is not None),
                        default=np.inf)
        raise ValueError(
            "no arrangement of at most %d beds keeps max/min at or below %.3f "
            "over a %.0f mm range; the most even one available reaches %.3f"
            % (max_beds, max_peak_to_trough, scan_length_mm, reachable))

    achievable = max(choice.min_eta for _n, choice in feasible)
    for n_beds, choice in feasible:
        if choice.min_eta >= (1.0 - tolerance) * achievable:
            break
    else:                                        # pragma: no cover - unreachable
        n_beds, choice = feasible[0]

    return BedArrangement(
        n_beds=n_beds, spacing_mm=choice.spacing_mm,
        coverage=coverage(profile, n_beds, choice.spacing_mm, scan_length_mm))
