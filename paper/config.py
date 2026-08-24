"""What the paper plots: the configurations, the tasks, and the LaTeX spellings.

The physical parameters come from ``slogpet.data``; what lives here is the
paper's own editorial choices -- which lengths and cylinders appear in which
figure, how a configuration is named, and how a crystal size is set.
"""
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from slogpet import C_OVER_2, Scanner
from slogpet.data import (DetectorGroup, load_systems, load_detector_groups,
                          load_styles)

__all__ = ["OUT_DIR", "out", "D_PET", "LENGTHS", "DIAMS", "SCANNERS", "SCANLEN",
           "CYLS", "SLOG_SIZES", "D_CYL_TAB", "SLOG_FWHM", "SCEN_CYLS",
           "SYSTEMS", "GROUPS", "STYLES", "SCENARIOS",
           "scen_label", "group_desc"]

# figures and tables are written next to main.tex, whatever the working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "manuscript")


def out(name: str) -> str:
    """A path next to main.tex, whatever the working directory."""
    return os.path.join(OUT_DIR, name)


# --- the generic scanner of Figures 4-6, where no particular system is meant ---
D_PET    = 740.0                               # mm, detector diameter
LENGTHS  = [1500.0, 1000.0, 600.0, 300.0]      # mm, axial length of the detector
DIAMS    = [0.0, 200.0, 300.0]                 # mm, attenuating cylinder diameter
SCANNERS = [(300.0, "30"), (1000.0, "100"), (1500.0, "150")]
SCANLEN  = [200.0, 1000.0, 1800.0]             # mm: single organ, FDG body, total body
CYLS     = [200.0, 300.0]                      # mm

# --- the tasks --------------------------------------------------------------
SLOG_SIZES = (5.0, 10.0)      # F_o values (mm FWHM) tabulated in Table 2
D_CYL_TAB  = 300.0            # object diameter assumed for the non-TOF rows of Table 2
SLOG_FWHM  = (5.0, 10.0)      # F_o values (mm FWHM) plotted in Figure 7
SCEN_CYLS  = (200.0, 300.0)   # D_cyl values plotted in Figure 7

# --- the systems ------------------------------------------------------------
SYSTEMS = load_systems()          # slogpet/data/systems.json
GROUPS  = load_detector_groups()  # slogpet/data/detectors.json
STYLES  = load_styles()

# every configuration, flattened: (index of its detector design, style, Scanner)
SCENARIOS = [(gi, sty, sc) for gi, g in enumerate(GROUPS)
             for sc, sty in g.configurations]


# --- LaTeX spellings of quantities that are stored as numbers ----------------
def _size(dims: Iterable[float], sep: str = r"\times") -> str:
    """Crystal dimensions -> a LaTeX product, e.g. $3.2\times3.2\times20$."""
    return "$" + sep.join("%g" % v for v in dims) + "$"
def _window(w: Optional[Sequence[float]]) -> str:
    """Energy window -> '435--585'."""
    return "---" if w is None else "%g--%g" % tuple(w)
def _round_half_up(x: float) -> int:
    """Round .5 away from zero, unlike Python's format(), which rounds to even."""
    return int(np.floor(x + 0.5))
def scen_label(sc: Scanner) -> str:
    """A configuration is named by its length, and by its ring-difference limit
    if it has one."""
    s = r"$%.0f$ cm" % (sc.L_pet / 10)
    if np.isfinite(sc.L_mrd):
        s += r", MRD $%.0f$ cm" % (sc.L_mrd / 10)
    return s
def group_desc(g: DetectorGroup) -> Tuple[str, str, str]:
    """Three vendor-neutral lines describing a detector design, generated from
    the design itself so that they cannot drift from the numbers being plotted:
    crystal and efficiency, timing, spatial resolution and bore."""
    l1 = (r"$\mathbf{%s}\quad$%s mm %s, $\epsilon=%.2f$"
          % (g.id, _size(g.crystal_size, r"{\times}"), g.crystal, g.epsilon))
    l2 = (r"CTR $%d$ ps $=%.0f$ mm" % (_round_half_up(g.F_t / C_OVER_2), g.F_t)
          if g.has_tof else r"no time of flight")
    res = (r"$F_y=F_z=%.1f$ mm" % g.F_y if g.F_y == g.F_z
           else r"$F_y=%.1f$, $F_z=%.1f$ mm" % (g.F_y, g.F_z))
    l3 = res + r", $D_{\mathrm{PET}}=%d$ cm" % _round_half_up(g.D_pet / 10)
    return (l1, l2, l3)
