# Tests

    python3 -m pytest                    # ~2 s, everything but the slow checks
    python3 -m pytest -m slow            # ~9 s, Monte Carlo and Eq. (53)
    python3 -m pytest -m ""              # both

Four kinds of test, in descending order of what they would catch:

**Properties that must hold whatever the code does.** `eta` even in `z`,
monotone in `|z|`, in the object diameter and in the ring diameter; the
attenuation factor bounded by the shortest chord; `S_ideal` monotone in length
and tending to 1 for an infinite scanner; `r` symmetric under permutation of its
three resolutions and bounded by 15/2; bed tiling conserving the integral of
`eta` for any bed count and any spacing. These do not depend on the values being
right, so they survive any recalibration of the inputs.

**Independent recomputation.** The closed form for `S_ideal` against adaptive
quadrature (now agreeing to 2e-15); the optimiser's reported minimum against a
brute-force minimum on an 80-times finer grid; `r_nonTOF` against a direct
numerical evaluation of Eq. (53) of Nuyts et al.; `eta` and `S_ideal` against
Monte Carlo sampling of directions. The last two are marked `slow`.

**Claims the paper makes.** That the detector-pair efficiency is a property of
the crystal rather than of the axial length -- systems sharing a crystal agree to
better than 20 % across a fourfold range of length; that the generic designs in
`detectors.json` are faithful to the systems they say they are based on; that the
one configuration with no counterpart in production is marked as hypothetical;
and that the task can reverse the ranking -- a long coarse non-TOF BGO system
loses to a short fine TOF LYSO one on a 3 mm SLoG and beats it on a 15 mm one.
If an edit to the data breaks one of these, an argument in the paper has changed.

**Pinned values.** `golden.json` holds 288 numbers -- every published system's
`S_ideal`, efficiency and `r`, every design's `r`, the optimised protocols, and
the SNR² of all ten configurations at four scan lengths, two SLoG sizes and two
cylinders. Compared at `rtol=1e-10`. A failure here is not necessarily a bug, but
it always has to be explained before `python3 tests/make_golden.py` is re-run.

These cover `slogpet` and `web/api.py`. `paper/` is deliberately untested: it is
presentation, it is expected to keep changing, and the check on it is looking at
the PDF.
