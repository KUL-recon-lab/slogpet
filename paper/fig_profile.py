"""Figure 4: the axial profile eta(z) for four detector lengths and three object
diameters."""
from typing import Optional, TYPE_CHECKING

import numpy as np

from slogpet import eta

from .style import figure_backend, finish, LENGTH_RAMP
from .config import out, D_PET, LENGTHS, DIAMS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT = out("fig-profile.pgf")


def make(path: Optional[str] = DEFAULT) -> "Figure":
    """Pass ``path=None`` to open the figure in a window instead of writing it."""
    plt = figure_backend(path)

    colours = LENGTH_RAMP
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 6.0), sharex=True)

    for ax, D_cyl in zip(axes, DIAMS):
        for L_pet in LENGTHS:
            z = np.linspace(-L_pet / 2, L_pet / 2, 401)
            y = [eta(zz, L_pet, D_PET, D_cyl) for zz in z]
            ax.plot(z / 10.0, y, color=colours[L_pet],
                    label=r"$L_{\mathrm{PET}}=%.0f$ cm" % (L_pet / 10))
        lbl = (r"$D_{\mathrm{cyl}}=0$ (geometry only)" if D_cyl == 0
               else r"$D_{\mathrm{cyl}}=%.0f$ cm" % (D_cyl / 10))
        ax.text(0.985, 0.93, lbl, transform=ax.transAxes, ha="right", va="top")
        ax.set_ylabel(r"$\eta(z)$")
        ax.set_xlim(-80, 80)
        ax.set_ylim(0, None)
        ax.grid(True, lw=0.4, color="0.85")
        ax.margins(y=0.18)
    axes[-1].set_xlabel(r"axial position $z$ (cm)")
    axes[0].legend(loc="upper left", ncol=2, frameon=True, framealpha=1.0,
                   borderpad=0.4, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    return finish(fig, path)


if __name__ == "__main__":
    make()
