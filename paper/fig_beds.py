"""Figure 6: the optimised multi-bed acquisition as a function of scan length --
the minimum of eta_N, the bed count, and the overlap."""
import numpy as np

from slogpet import eta_lattice, optimise_beds

from .style import use_pgf, LENGTH_RAMP
from .config import out, D_PET, SCANNERS, SCANLEN, CYLS

DEFAULT = out("fig-beds.pgf")


def make(path=DEFAULT):
    plt = use_pgf()
    colours = LENGTH_RAMP
    Ss = np.concatenate([np.arange(5., 60., 2.5), np.arange(60., 205., 5.)])*10
    fig, axes = plt.subplots(4, 2, figsize=(6.0, 8.0), sharex=True)
    for col, D_cyl in enumerate(CYLS):
        axes[3, col].axhline(50.0, color="0.55", lw=0.6, ls=(0, (1, 2)))
        for L_pet, lab in SCANNERS:
            lat = eta_lattice(L_pet, D_PET, D_cyl)
            res = [optimise_beds(lat, L_pet, S) for S in Ss]
            M = [r[2] for r in res]
            axes[0, col].plot(Ss/10, M, color=colours[L_pet],
                              label=r"$L_{\mathrm{PET}}=%s$\,cm" % lab)
            axes[1, col].plot(Ss/10, M, color=colours[L_pet])
            axes[2, col].step(Ss/10, [r[0] for r in res], color=colours[L_pet], where="post")
            ov = [100*(1 - r[1]/L_pet) if r[0] > 1 else np.nan for r in res]
            axes[3, col].plot(Ss/10, ov, color=colours[L_pet])
        axes[0, col].set_ylim(bottom=0.0)
        axes[1, col].set_yscale("log")
        axes[0, col].set_title(r"$D_{\mathrm{cyl}}=%.0f$ cm" % (D_cyl/10), fontsize=9)
        axes[3, col].set_xlabel(r"scan length $S$ (cm)")
        axes[3, col].set_ylim(0, 100)
        for r in (0, 1, 2, 3):
            axes[r, col].grid(True, lw=0.4, color="0.85")
            for S in SCANLEN:
                axes[r, col].axvline(S/10, color="0.55", lw=0.6, ls=(0, (4, 3)))
    axes[0, 0].set_ylabel(r"$\min_{|z|\le S/2}\eta_N(z)$" "\n" r"(linear)")
    axes[1, 0].set_ylabel(r"$\min_{|z|\le S/2}\eta_N(z)$" "\n" r"(logarithmic)")
    axes[2, 0].set_ylabel(r"optimal $N$")
    axes[3, 0].set_ylabel(r"optimal overlap (\%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colours[L], lw=1.1,
                      label=r"$L_{\mathrm{PET}}=%s$\,cm" % lab)
               for L, lab in SCANNERS]
    axes[2, 0].legend(handles=handles, loc="upper left", frameon=True,
                      framealpha=1.0, handlelength=1.6)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.8)
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    make()
