"""Presentation helpers for explore.py, kept out of the way.

Nothing here is about PET: it is the plumbing that makes the table and the
figures in ``explore.py`` readable and consistent, moved out so that the
notebook a reader sees is about the physics and the choices, not about how a
plot is put together.

The one decision worth knowing is where a figure goes.  Under a Jupyter kernel
-- JupyterLite included -- it is drawn beneath the cell; at a terminal it is
written as an HTML file and opened in a browser tab.  Importing this module
settles that once and, in a notebook, loads BokehJS.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, Range1d
from bokeh.plotting import figure

from slogpet.resolution import C_OVER_2      # mm per ps, for CTR <-> F_t

__all__ = ["COLOURS", "IN_NOTEBOOK", "in_notebook", "panel", "draw", "legend_of",
           "series", "systems_table", "show_table"]

# Colour-vision checked; cycled if you compare more systems than there are.
COLOURS = ("#2a78d6", "#eda100", "#d55181", "#008300", "#7a5195", "#4a4a48")

WIDTH = 780
HEIGHT = 270


def in_notebook() -> bool:
    """True when there is a kernel to draw into.

    Deliberately not a test for any particular kernel -- JupyterLite's Pyodide
    kernel does not import ipykernel, and others differ again.  What is asked
    instead is whether IPython is running at all, and whether it is the terminal
    shell, which is the one shell that cannot show a figure inline.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False                       # a plain python run
    shell = get_ipython()
    return shell is not None and type(shell).__name__ != "TerminalInteractiveShell"


IN_NOTEBOOK = in_notebook()
if IN_NOTEBOOK:
    output_notebook(hide_banner=True)


def panel(y_label: str, x_label: Optional[str] = None,
          x_range: Optional[Range1d] = None,
          y_axis_type: str = "linear") -> figure:
    """One plot, sized and tooled the same as all the others.

    Pass another panel's ``x_range`` to tie the two together, so that panning or
    zooming either moves both.
    """
    # x_range is passed only when there is one to share: bokeh rejects None
    shared: Dict[str, Any] = {"x_range": x_range} if x_range is not None else {}
    p = figure(width=WIDTH, height=HEIGHT, x_axis_label=x_label,
               y_axis_label=y_label, y_axis_type=y_axis_type,
               tools="pan,box_zoom,wheel_zoom,reset,save", **shared)
    p.add_tools(HoverTool(tooltips=[("", "$name"), ("x", "$x{0.0}"),
                                    ("y", "$y{0.000 a}")], mode="vline"))
    p.grid.grid_line_alpha = 0.35
    p.toolbar.logo = None
    return p


def draw(panels: Sequence[figure], filename: str) -> None:
    """Show a stack of panels: beneath the cell in a notebook, in a browser tab
    from a script.  One toolbar drives all of them."""
    if not IN_NOTEBOOK:
        output_file(filename, title="SLoG PET explorer")
    show(gridplot([[p] for p in panels], merge_tools=True, toolbar_location="right"))


def legend_of(p: figure, location: str = "top_left") -> None:
    """Make a panel's legend readable and clickable: clicking an entry hides
    that curve, which helps when several of them overlap."""
    p.legend.location = location
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.85


def series(protocols: Sequence[Any], attribute: str) -> np.ndarray:
    """One protocol attribute against scan length.

    ``None`` -- a scan length where the ripple limit could not be met -- becomes
    NaN, which leaves a gap in the curve rather than a wrong point.
    """
    return np.array([getattr(p, attribute) if p is not None else np.nan
                     for p in protocols], dtype=float)


# ------------------------------------------------------------------ tables
def systems_table(scanners: Sequence[Any]) -> pd.DataFrame:
    """The parameters that decide how a system performs, one row each.

    Every column is a number the model actually uses, except the crystal, which
    is there to make the families recognisable.  ``eps`` is the detector-pair
    efficiency: as published where a system quotes one, and otherwise worked out
    from its NEMA sensitivity and its geometry.
    """
    rows = []
    for scanner in scanners:
        rows.append({
            "system": scanner.name,
            "L_PET (cm)": round(scanner.L_pet / 10, 1),
            "D_PET (cm)": round(scanner.D_pet / 10, 1),
            "MRD (cm)": (np.nan if not np.isfinite(scanner.L_mrd)
                         else round(scanner.L_mrd / 10, 1)),
            "crystal": scanner.crystal or "",
            "F_y (mm)": scanner.F_y,
            "F_z (mm)": scanner.F_z,
            "CTR (ps)": (np.nan if scanner.F_t is None
                         else round(scanner.F_t / C_OVER_2)),
            "S_NEMA (cps/kBq)": (np.nan if scanner.S_nema is None
                                 else round(scanner.S_nema, 1)),
            "eps": round(scanner.efficiency(), 3),
        })
    return pd.DataFrame(rows).set_index("system")


def show_table(frame: pd.DataFrame) -> None:
    """Display a table: as HTML in a notebook, as aligned text from a script.

    A blank cell means the system has no such number -- no time of flight, no
    ring-difference limit -- rather than a missing measurement.
    """
    if not IN_NOTEBOOK:
        # the same "%g" the notebook gets, so the two agree: 210 not 210.0
        print(frame.to_string(na_rep="-", float_format=lambda v: "%g" % v))
        return
    from IPython.display import display
    try:
        numbers = frame.select_dtypes("number").columns
        display(frame.style.format("{:g}", na_rep="-", subset=numbers))
    except ImportError:                    # Styler needs jinja2; the frame alone
        display(frame)                     # is still perfectly readable
