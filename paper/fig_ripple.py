"""Figure 5: why the best bed count is not a smooth function of the scan length."""
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

from slogpet import (sample_single_bed_profile, best_spacing_for_n_beds,
                     coverage, tile_beds)

from .style import figure_backend, finish, SEQ, HIGHLIGHT
from .config import out, D_PET

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT = out("fig-ripple.pgf")


def make(path: Optional[str] = DEFAULT, L_pet: float = 1000.0,
         D_cyl: float = 200.0, S: float = 1500.0,
         max_peak_to_trough: Optional[float] = None,
         profile_counts: Sequence[int] = (1, 2, 4, 5),
         Ns: Sequence[int] = list(range(1, 15))
         ) -> "Figure":
    """Why the best N is not a smooth function of S: the tiled profile ripples, and how
    much it ripples depends on N in a way that is not monotone.

    Three panels.  Left, top: what each bed count achieves at the worst point of
    the range and on average -- the two pull in opposite directions, since a
    single bed has the highest mean of all and a minimum of zero.  Left, bottom:
    how uneven the result is.  Right: the profiles those numbers come from.

    ``max_peak_to_trough`` draws the same figure under a limit on how uneven the
    profile may be.  Bed counts that cannot meet it disappear from the left-hand
    panels, and those that can only meet it at a different spacing are shown at
    that spacing -- so the figure shows what the limit costs.  The default, None,
    imposes no limit and is what the paper prints.

    ``profile_counts`` are the bed counts drawn on the right; any that the limit
    rules out are replaced by the next feasible ones.
    """
    plt = figure_backend(path, **{"legend.fontsize": 7.5})
    profile = sample_single_bed_profile(L_pet, D_PET, D_cyl)
    res = {N: best_spacing_for_n_beds(profile, L_pet, S, N,
                                      max_peak_to_trough=max_peak_to_trough)
           for N in Ns}
    feasible = [N for N in Ns if res[N] is not None]
    if not feasible:
        raise ValueError("no bed count can keep max/min below %.3f"
                         % max_peak_to_trough)
    cov = {N: coverage(profile, N, res[N].spacing_mm, S) for N in feasible}
    best = max(cov[N].min_eta for N in feasible)
    Nrep = next(N for N in feasible if cov[N].min_eta >= 0.97 * best)

    # the profiles to draw: those asked for that survive the limit, topped up
    # with the next feasible counts so the panel keeps its four curves
    shown = [N for N in profile_counts if N in feasible]
    for N in feasible:
        if len(shown) >= len(profile_counts):
            break
        if N not in shown:
            shown.append(N)
    shown = sorted(shown)

    # constrained layout, not tight_layout: the right-hand panel spans both rows,
    # which tight_layout cannot handle
    fig = plt.figure(figsize=(6.0, 3.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.35, 1.0])
    a1 = fig.add_subplot(gs[0, 0])                       # min and mean against N
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)            # how uneven, against N
    a3 = fig.add_subplot(gs[:, 1])                       # the profiles themselves

    # ---- what each bed count achieves ------------------------------------
    a1.axhspan(0.97 * best, best, color="0.88", zorder=0)
    a1.plot(feasible, [cov[N].mean_eta for N in feasible], "s--", color=SEQ[1],
            ms=3.2, label=r"mean over the range")
    a1.plot(feasible, [cov[N].min_eta for N in feasible], "o-", color=SEQ[3],
            ms=4, label=r"minimum over the range")
    a1.plot([Nrep], [cov[Nrep].min_eta], "o", color=HIGHLIGHT, ms=7, mfc="none", mew=1.8)
    a1.set_ylabel(r"$\eta_N$")
    a1.grid(True, lw=0.4, color="0.85")
    # headroom for the legend, so that it does not sit on the mean curve
    a1.set_ylim(0.0, 1.28 * max(c.mean_eta for c in cov.values()))
    a1.annotate(r"within $3\,\%$ of best", xy=(10.5, 0.985 * best),
                xytext=(9.0, 0.50 * best), fontsize=7.5, color="0.35", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.55"))
    if 1 in feasible and cov[1].min_eta == 0.0:
        a1.annotate(r"range not covered", xy=(1.06, 0.02 * best),
                    xytext=(3.6, 0.28 * best), fontsize=7.5, color="#7a7a7a",
                    va="center", ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#7a7a7a"))
    a1.legend(loc="upper right", frameon=True, framealpha=1.0, handlelength=1.8,
              borderpad=0.3, labelspacing=0.22, borderaxespad=0.4)
    a1.tick_params(labelbottom=False)

    # ---- how uneven it is --------------------------------------------------
    # N = 1 leaves part of the range uncovered, so its ratio is infinite and is
    # left out rather than drawn off the top of the panel.
    covered = [N for N in feasible if cov[N].min_eta > 0]
    a2.axhline(1.0, color="0.55", lw=0.6, ls=(0, (1, 2)))
    a2.plot(covered, [cov[N].peak_to_trough for N in covered], "o-",
            color=SEQ[2], ms=4)
    a2.plot([Nrep], [cov[Nrep].peak_to_trough], "o", color=HIGHLIGHT, ms=7,
            mfc="none", mew=1.8)
    a2.set_xlabel("number of bed positions $N$")
    a2.set_ylabel(r"$\max\eta_N\,/\,\min\eta_N$")
    a2.grid(True, lw=0.4, color="0.85")
    if max_peak_to_trough is not None:
        a2.axhline(max_peak_to_trough, color=HIGHLIGHT, lw=0.9, ls=(0, (4, 2)))
        a2.text(0.985, max_peak_to_trough, "limit", fontsize=7.5, color="0.35",
                ha="right", va="top",
                transform=a2.get_yaxis_transform(which="grid"))
    a2.set_ylim(bottom=1.0)
    a2.set_xticks(Ns[::2])

    # ---- the profiles behind those numbers ---------------------------------
    Zw = 2.0 * S / 3.0                                 # plot out to +/- 100 cm
    Z = np.linspace(-Zw, Zw, 1601)
    a3.axvspan(-S / 20, S / 20, color="0.90", zorder=0)
    for N, col in zip(shown, SEQ):
        d = res[N].spacing_mm
        p = tile_beds(profile, N, d, Z)
        inr = np.abs(Z) <= S / 2
        lbl = (r"$N=1$ (single bed)" if N == 1 else
               r"$N=%d$, $%.0f\%%$ overlap" % (N, 100 * (1 - d / L_pet)))
        a3.plot(Z / 10, p, color=col, lw=1.0, ls=("--" if N == 1 else "-"), label=lbl)
        a3.plot(Z[inr][np.argmin(p[inr])] / 10, p[inr].min(), "o", color=col, ms=3.5)
        a3.axhline(p[inr].min(), color=col, lw=0.5, ls=(0, (3, 3)))
    a3.set_xlabel(r"axial position $z$ (cm)")
    a3.set_ylabel(r"$\eta_N(z)$")
    a3.set_xlim(-Zw / 10, Zw / 10)
    a3.set_ylim(0, None)
    a3.axvline(-S / 20, color="0.45", lw=0.7, ls=(0, (4, 3)))
    a3.axvline(S / 20, color="0.45", lw=0.7, ls=(0, (4, 3)))
    a3.text(0.0, 0.965, r"range of interest", transform=a3.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="0.35")
    a3.grid(True, lw=0.4, color="0.85")
    a3.legend(loc="lower right", frameon=True, framealpha=1.0, handlelength=1.4,
              borderpad=0.3, labelspacing=0.25)

    return finish(fig, path)


if __name__ == "__main__":
    make()
