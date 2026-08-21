"""Figure 5: why the best bed count is not a smooth function of the scan length."""
import numpy as np

from slogpet import eta_lattice, best_for_N, tiled_profile

from .style import use_pgf, SEQ, HIGHLIGHT
from .config import out, D_PET

DEFAULT = out("fig-ripple.pgf")


def make(path=DEFAULT, L_pet=1000.0, D_cyl=200.0, S=1500.0):
    """Why the best N is not a smooth function of S: the tiled profile ripples, and how
    much it ripples depends on N in a way that is not monotone."""
    plt = use_pgf(**{"legend.fontsize": 7.5})
    lat = eta_lattice(L_pet, D_PET, D_cyl)
    Ns = list(range(1, 15))
    res = {N: best_for_N(lat, L_pet, S, N) for N in Ns}
    best = max(M for M, _ in res.values())
    Nrep = next(N for N in Ns if res[N][0] >= 0.97*best)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.5))
    a1.axhspan(0.97*best, best, color="0.88", zorder=0)
    a1.plot(Ns, [res[N][0] for N in Ns], "o-", color=SEQ[2], ms=4)
    a1.plot([Nrep], [res[Nrep][0]], "o", color=HIGHLIGHT, ms=7, mfc="none", mew=1.8)
    a1.set_xlabel("number of bed positions $N$")
    a1.set_ylabel(r"$\min_{|z|\le S/2}\eta_N(z)$")
    a1.grid(True, lw=0.4, color="0.85")
    a1.text(0.97, 0.06, r"within $3\,\%$ of best", transform=a1.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="0.35")
    a1.annotate(r"range not covered", xy=(1.05, 0.02*best), xytext=(1.8, 0.42*best),
                fontsize=7.5, color="#7a7a7a", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#7a7a7a"))
    a1.set_xticks(Ns[::2])

    Zw = 2.0*S/3.0                                   # plot out to +/- 100 cm
    Z = np.linspace(-Zw, Zw, 1601)
    a2.axvspan(-S/20, S/20, color="0.90", zorder=0)
    for N, col in zip((1, 2, 4, 5), SEQ):
        M, d = res[N]
        p = tiled_profile(lat, N, d, Z)
        inr = np.abs(Z) <= S/2
        lbl = (r"$N=1$ (single bed)" if N == 1 else
               r"$N=%d$, %.0f\%% overlap" % (N, 100*(1-d/L_pet)))
        a2.plot(Z/10, p, color=col, lw=1.0, ls=("--" if N == 1 else "-"), label=lbl)
        a2.plot(Z[inr][np.argmin(p[inr])]/10, p[inr].min(), "o", color=col, ms=3.5)
        a2.axhline(p[inr].min(), color=col, lw=0.5, ls=(0, (3, 3)))
    a2.set_xlabel(r"axial position $z$ (cm)")
    a2.set_ylabel(r"$\eta_N(z)$")
    a2.set_xlim(-Zw/10, Zw/10)
    a2.axvline(-S/20, color="0.45", lw=0.7, ls=(0, (4, 3)))
    a2.axvline( S/20, color="0.45", lw=0.7, ls=(0, (4, 3)))
    a2.text(0.0, 0.965, r"range of interest", transform=a2.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="0.35")
    a2.grid(True, lw=0.4, color="0.85")
    a2.legend(loc="lower right", frameon=True, framealpha=1.0, handlelength=1.4,
              borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    make()
