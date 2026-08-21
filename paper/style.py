"""Matplotlib setup shared by the figure scripts.

The paper's figures are produced through the pgf backend so that the fonts and
maths match the document exactly; Figure 1 is the exception, being a bitmap
rendered through Agg.  Each figure differs from the common settings in a few
sizes only, which are passed as keyword overrides.
"""
__all__ = ["PGF_RC", "AGG_RC", "SCENARIO_RC", "use_pgf", "use_agg",
           "LENGTH_RAMP", "SEQ", "HIGHLIGHT", "NEUTRAL"]

# --- colour -----------------------------------------------------------------
#
# Curves coloured by axial length (Figures 4-6) use one hue, light to dark.
# Length is an ordered quantity, so lightness carries it: longer is darker, the
# figure survives greyscale printing, and no pair can be confused under any
# colour-vision deficiency.  The four steps are evenly spaced in OKLab lightness
# (0.764, 0.622, 0.480, 0.338) and all clear 2:1 contrast on white.
#
# Figure 7 is the one genuinely categorical figure -- four detector designs with
# no order between them -- and its four hues are in slogpet/data/detectors.json.
# They were chosen by enumerating four-subsets of an eight-hue palette and
# keeping only those clearing the lightness band, the chroma floor and the
# all-pairs colour-vision and normal-vision separations in both light and dark.

SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # light -> dark
LENGTH_RAMP = {300.0: SEQ[0], 600.0: SEQ[1], 1000.0: SEQ[2], 1500.0: SEQ[3]}
HIGHLIGHT = "#eda100"      # the one chosen point, against the blue curve
NEUTRAL = "#7a7a7a"

PGF_RC = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "pgf.preamble": r"\usepackage{amsmath}",
}

# Figure 7 carries four rows and a four-block legend, so everything shrinks
SCENARIO_RC = {
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2,      # ten thin curves per panel; 1.0 pt printed faint
}

AGG_RC = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "text.usetex": False,
    "mathtext.fontset": "dejavuserif",
}


def _setup(backend, base, extra):
    import matplotlib as mpl
    mpl.use(backend)
    rc = dict(base)
    rc.update(extra)
    mpl.rcParams.update(rc)
    import matplotlib.pyplot as plt
    return plt


def use_pgf(**extra):
    """Select the pgf backend and return pyplot."""
    return _setup("pgf", PGF_RC, extra)


def use_agg(**extra):
    """Select the Agg backend and return pyplot."""
    return _setup("Agg", AGG_RC, extra)
