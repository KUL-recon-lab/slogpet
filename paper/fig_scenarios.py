"""Figure 7: SNR^2 for ten configurations built from four detector designs.

The sweep over scan length is shared with the printed table of the same numbers,
which is what the text of the last section quotes, so both live here.
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from slogpet import Task, axial_profile, optimal_protocol, snr2_value

from .style import figure_backend, finish, SCENARIO_RC
from .config import (out, GROUPS, STYLES, SCENARIOS, SLOG_FWHM, SCEN_CYLS,
                     scen_label, group_desc)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

DEFAULT = out("fig-scenarios.pgf")


def scenario_data(Ss: "NDArray[np.float64]") -> Dict[Tuple[int, float], "NDArray[np.float64]"]:
    """min_z eta_N for every (scenario, D_cyl, S); the protocol does not depend on F_o."""
    out = {}
    for k, (grp, sty, sc) in enumerate(SCENARIOS):
        for D_cyl in SCEN_CYLS:
            prof = axial_profile(sc, D_cyl)
            out[(k, D_cyl)] = np.array(
                [optimal_protocol(prof, S).min_eta for S in Ss])
    return out


def make(path: Optional[str] = DEFAULT, nS: int = 41) -> "Figure":
    """Pass ``path=None`` to open the figure in a window instead of writing it."""
    plt = figure_backend(path, **SCENARIO_RC)
    from matplotlib.lines import Line2D

    Ss = np.linspace(50.0, 2000.0, nS)
    minEta = scenario_data(Ss)

    rows = [(F_o, D_cyl) for F_o in SLOG_FWHM for D_cyl in SCEN_CYLS]
    fig, axes = plt.subplots(4, 2, figsize=(6.0, 8.9), sharex=True)
    for i, (F_o, D_cyl) in enumerate(rows):
        task = Task(F_o, D_cyl)
        for k, (grp, sty, sc) in enumerate(SCENARIOS):
            y = snr2_value(sc.efficiency(), minEta[(k, D_cyl)], sc.r(task), F_o)
            kw = dict(color=GROUPS[grp].colour, ls=STYLES[sty], lw=1.2)
            axes[i, 0].plot(Ss/10, y, **kw)
            axes[i, 1].plot(Ss/10, y, **kw)
        axes[i, 0].set_ylim(bottom=0.0)
        axes[i, 1].set_yscale("log")
        axes[i, 0].set_ylabel(r"$F_{\mathrm{o}}=%.0f$ mm, $D_{\mathrm{cyl}}=%.0f$ cm"
                              % (F_o, D_cyl/10), fontsize=8)
        for j in (0, 1):
            axes[i, j].grid(True, lw=0.4, color="0.88")
    axes[0, 0].set_title("linear scale", fontsize=8.5)
    axes[0, 1].set_title("logarithmic scale", fontsize=8.5)
    for j in (0, 1):
        axes[-1, j].set_xlabel(r"scan length $S$ (cm)")
    fig.supylabel(r"$\mathrm{SNR}^2_{\min}$ (arbitrary units)", fontsize=9, x=0.005)

    # legend: one column per detector design, headed by the design
    blank = Line2D([], [], ls="none")
    def r_line(g: int) -> str:
        """The resolution factor of this detector at the two SLoG sizes.  For a TOF
        design r is independent of the object; for a non-TOF one it is not, so both
        cylinder diameters are given."""
        sc = GROUPS[g].scanners[0]
        if sc.has_tof:
            a, b = (sc.r(Task(F_o, SCEN_CYLS[0])) for F_o in SLOG_FWHM)
            return r"$r=%.3f$, $%.3f$" % (a, b)
        v = [[sc.r(Task(F_o, D)) for D in SCEN_CYLS] for F_o in SLOG_FWHM]
        return (r"$r=%.3f$, $%.3f$ ($D_{\mathrm{cyl}}{=}20$); $%.3f$, $%.3f$ ($30$)"
                % (v[0][0], v[1][0], v[0][1], v[1][1]))

    def block(g: int) -> List[Tuple[Any, str]]:
        gcol = GROUPS[g].colour
        out = [(blank, "\n".join(group_desc(GROUPS[g]) + (r_line(g),)))]
        for grp, sty, sc in SCENARIOS:
            if grp == g:
                out.append((Line2D([], [], color=gcol, ls=STYLES[sty], lw=1.2),
                            r"$L_{\mathrm{PET}}=$ " + scen_label(sc)))
        return out
    cols = [block(0) + [(blank, "")] + block(1),      # left  legend column
            block(2) + [(blank, "")] + block(3)]      # right legend column
    nmax = max(len(c) for c in cols)
    handles, labels = [], []
    for col in cols:
        col = col + [(blank, "")]*(nmax - len(col))
        handles += [h for h, _ in col]
        labels  += [l for _, l in col]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, handlelength=2.8, handletextpad=0.5,
               columnspacing=1.2, labelspacing=0.32, bbox_to_anchor=(0.52, -0.004))
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=1.0, rect=(0.015, 0.30, 1, 1))
    return finish(fig, path)


def print_numbers(nS: int = 41) -> None:
    """Numbers behind the scenario figure, for writing the text."""
    Ss = np.linspace(50.0, 2000.0, nS)
    minEta = scenario_data(Ss)
    for F_o in SLOG_FWHM:
        for D_cyl in SCEN_CYLS:
            print("\n=== F_o = %.0f mm, D_cyl = %.0f cm ===" % (F_o, D_cyl/10))
            print("%-28s %7s | %s" % ("configuration", "r",
                  "  ".join("S=%3.0f" % s for s in (200, 500, 1000, 1800))))
            task = Task(F_o, D_cyl)
            for k, (grp, sty, sc) in enumerate(SCENARIOS):
                r = sc.r(task)
                y = snr2_value(sc.efficiency(), minEta[(k, D_cyl)], r, F_o)
                vals = [np.interp(s, Ss, y) for s in (200.0, 500.0, 1000.0, 1800.0)]
                nm = ("G%d " % (grp+1)) + scen_label(sc).replace("$", "").replace("\\", "")
                print("%-28s %7.4f | %s" % (nm, r,
                      "  ".join("%6.4f" % v for v in vals)))


if __name__ == "__main__":
    make()
