"""Matplotlib setup shared by the figure scripts.

Every figure module has the same shape::

    def make(path=DEFAULT, ...):
        plt = figure_backend(path)
        ...
        return finish(fig, path)

and ``path`` decides where the figure goes:

* a file name (the default) selects the backend the paper is built with -- pgf
  for the vector figures, Agg for the one bitmap -- writes the file and prints
  where it went.  This is what ``make_all.py`` and the GitHub action do, and the
  output is byte-for-byte reproducible.
* ``None`` selects whatever interactive backend the session has, so the figure
  opens in a window and is returned for further poking.  This is for working at
  an ipython prompt::

      In [1]: from paper.fig_ripple import make
      In [2]: fig = make(None, max_peak_to_trough=1.25)

  Sizes, colours and line widths are the paper's in both cases; only the text
  rendering can differ.  If a LaTeX installation is on the PATH the labels are
  typeset by LaTeX exactly as in the document; if not, matplotlib's own mathtext
  draws them, which is close enough to judge a figure by.  The labels are
  written so that both readings work, so avoid text-mode LaTeX (``\\textbf``,
  ``\\%`` outside maths) when adding new ones.
"""
import shutil
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:                                  # only for the annotations
    from matplotlib.figure import Figure

__all__ = ["PGF_RC", "AGG_RC", "SCENARIO_RC", "SCREEN_RC",
           "use_pgf", "use_agg", "use_screen", "figure_backend", "finish",
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

SEQ: List[str] = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # light -> dark
LENGTH_RAMP: Dict[float, str] = {300.0: SEQ[0], 600.0: SEQ[1],
                                 1000.0: SEQ[2], 1500.0: SEQ[3]}
HIGHLIGHT: str = "#eda100"      # the one chosen point, against the blue curve
NEUTRAL: str = "#7a7a7a"

PGF_RC: Dict[str, Any] = {
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
SCENARIO_RC: Dict[str, Any] = {
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2,      # ten thin curves per panel; 1.0 pt printed faint
}

AGG_RC: Dict[str, Any] = {
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

# What changes when a figure is drawn on screen rather than into the document:
# a dpi that suits a laptop display, and text rendered by whatever is available.
SCREEN_RC: Dict[str, Any] = {
    "figure.dpi": 110,
    "figure.facecolor": "white",     # readable under a dark desktop theme
}

# Candidate window backends, in the order they are tried.  The first one that
# imports wins; matplotlib's own default is used if it is already interactive.
_WINDOW_BACKENDS = ("macosx", "qtagg", "gtk4agg", "gtk3agg", "tkagg", "webagg")

_screen_backend: Optional[str] = None       # remembered once one has worked
_usetex: Optional[bool] = None              # remembered once it has been tried


# --- backends ---------------------------------------------------------------
def _apply(backend: Optional[str], base: Mapping[str, Any],
           extra: Mapping[str, Any]) -> ModuleType:
    """Select *backend* (None = leave it alone), apply the rcParams, return pyplot."""
    import matplotlib as mpl
    if backend is not None:
        mpl.use(backend)
    rc: Dict[str, Any] = dict(base)
    rc.update(extra)
    mpl.rcParams.update(rc)
    import matplotlib.pyplot as plt
    return plt


_FILE_BACKENDS = frozenset(["agg", "pdf", "pgf", "ps", "svg", "cairo", "template"])


def _is_interactive(backend: str) -> bool:
    """Does this backend open a window?  Asked of matplotlib where it will
    answer -- the two spellings below cover 3.9 onwards -- and by name if not."""
    try:
        from matplotlib.backends.registry import backend_registry
        return bool(backend_registry.is_interactive_backend(backend))
    except Exception:
        pass
    try:
        from matplotlib import rcsetup
        return backend.lower() in [b.lower() for b in rcsetup.interactive_bk]
    except Exception:
        return backend.lower() not in _FILE_BACKENDS


def _select_window_backend() -> None:
    """Make sure a backend that can open a window is active.

    Needed because writing a figure to a file switches matplotlib to pgf or Agg
    for the whole process, so a later on-screen call has to switch back.
    """
    global _screen_backend
    import matplotlib as mpl
    if _screen_backend is None and _is_interactive(mpl.get_backend()):
        _screen_backend = mpl.get_backend()          # the session's own choice
    if _screen_backend is not None:
        mpl.use(_screen_backend)
        return
    for candidate in _WINDOW_BACKENDS:
        try:
            mpl.use(candidate)
        except Exception:
            continue
        _screen_backend = candidate
        return
    raise RuntimeError(
        "no window backend available (tried %s); either install one, or pass a "
        "file name so the figure is written instead of shown"
        % ", ".join(_WINDOW_BACKENDS))


def _latex_available() -> bool:
    """Can matplotlib really typeset these labels with LaTeX?

    It shells out to latex, converts the result with dvipng and needs the Type 1
    Computer Modern fonts on top, so the only honest test is to typeset
    something and see.  The answer is remembered for the rest of the session,
    and the fallback is announced once so that a figure that looks slightly off
    is not a mystery.
    """
    global _usetex
    if _usetex is not None:
        return _usetex
    _usetex = False
    if shutil.which("latex") and shutil.which("dvipng"):
        import matplotlib as mpl
        from matplotlib.texmanager import TexManager
        with mpl.rc_context({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": PGF_RC["pgf.preamble"]}):
            try:
                TexManager().make_png(r"$\eta_N\,\%$", fontsize=9, dpi=72)
                _usetex = True
            except Exception:
                pass
    if not _usetex:
        print("note: LaTeX is not usable here, so the labels are drawn with "
              "matplotlib's mathtext; the written figures are unaffected.")
    return _usetex


def use_pgf(**extra: Any) -> ModuleType:
    """Select the pgf backend and return pyplot."""
    return _apply("pgf", PGF_RC, extra)


def use_agg(**extra: Any) -> ModuleType:
    """Select the Agg backend and return pyplot."""
    return _apply("Agg", AGG_RC, extra)


def use_screen(rc: Mapping[str, Any] = PGF_RC, **extra: Any) -> ModuleType:
    """Select a window backend, keep the paper's sizes, and return pyplot."""
    _select_window_backend()
    screen: Dict[str, Any] = dict(SCREEN_RC)
    if _latex_available():
        # the document's own typesetting, so the window shows the printed figure
        screen["text.usetex"] = True
        screen["text.latex.preamble"] = PGF_RC["pgf.preamble"]
    else:
        screen["text.usetex"] = False
        screen["mathtext.fontset"] = "dejavuserif"
    screen.update(extra)
    return _apply(None, rc, screen)


def figure_backend(path: Optional[str], rc: Mapping[str, Any] = PGF_RC,
                   **extra: Any) -> ModuleType:
    """Return pyplot set up to draw a figure destined for *path*.

    A file name gives the backend that file needs (pgf, or Agg for a bitmap);
    ``None`` gives a window.  *rc* is the figure's own base style.
    """
    if path is None:
        return use_screen(rc, **extra)
    if rc is AGG_RC:
        return _apply("Agg", AGG_RC, extra)
    return _apply("pgf", PGF_RC, extra)


def finish(fig: "Figure", path: Optional[str], **savefig_kw: Any) -> "Figure":
    """Write the figure to *path*, or show it if *path* is None.  Returns the
    figure either way, so an interactive caller can keep working on it."""
    import matplotlib.pyplot as plt
    if path is None:
        plt.show()          # returns at once under ipython's %matplotlib
        return fig
    fig.savefig(path, **savefig_kw)
    print("wrote", path)
    plt.close(fig)          # a batch run builds six of these
    return fig
