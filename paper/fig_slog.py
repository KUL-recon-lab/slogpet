"""Figure 1: the two hypotheses and their difference, at perfect resolution and
after smoothing with a kernel of the SLoG size."""
import numpy as np

from .style import use_agg
from .config import out

DEFAULT = out("fig-slog2d.pdf")


def make(path=DEFAULT, smooth_frac=1.0, contrast=0.35):
    """Transaxial cuts through the two hypotheses and through their difference.

    Nuyts et al. Eq. (9) decomposes the SLoG exactly into two strictly positive blobs of
    equal total activity,

        SLoG(r) = 3 so^3 G3(r, so)  -  so r^2 G3(r, so),
                  \\_ small hot _/     \\_ wider blob, cold interior _/

    so lambda_1 = B + hollow blob and lambda_2 = B + hot blob differ by exactly the SLoG.
    Smoothing with a 3-D Gaussian of width s is closed form for all three, because
    convolution commutes with d/d(so):  G3(.,so) -> G3(.,u) with u = sqrt(so^2 + s^2), and
    SLoG -> so^5/u (r^2/u^2 - 3) G3(r,u)/u.  The objects are radially symmetric, so one cut
    through the centre says everything.
    """
    plt = use_agg()

    so = 1.0
    L = 4.0
    g = np.linspace(-L, L, 601)
    X, Y = np.meshgrid(g, g)
    r2 = X**2 + Y**2

    def G3(r2_, s):
        return np.exp(-0.5*r2_/(s*s))/((2*np.pi)**1.5 * s**3)

    def triple(s_ker):
        """hot, hollow and SLoG after smoothing with a 3-D Gaussian of width s_ker."""
        u = np.sqrt(so*so + s_ker*s_ker)
        hot  = 3.0*so**3 * G3(r2, u)
        slog = so**5/u * (r2/(u*u) - 3.0) * G3(r2, u) / u
        slog = -slog                       # sign convention: positive core
        return hot, hot - slog, slog

    rows = [triple(0.0), triple(smooth_frac*so)]
    peak = rows[0][0].max()                       # hot blob at perfect resolution
    rows = [tuple(a/peak for a in row) for row in rows]
    dmax = float(np.abs(rows[0][2]).max())

    fig = plt.figure(figsize=(5.9, 3.5))
    gs = fig.add_gridspec(2, 8,
                          width_ratios=[1, 1, 0.07, 0.05, 0.26, 1, 0.07, 0.05],
                          wspace=0.06, hspace=0.06)
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j, c in enumerate((0, 1, 5)):
            axes[i, j] = fig.add_subplot(gs[i, c])
    cax1, cax2 = fig.add_subplot(gs[:, 3]), fig.add_subplot(gs[:, 7])

    ext = [-L, L, -L, L]
    for i, (hot, hollow, slog) in enumerate(rows):
        for j, im_ in enumerate((hollow, hot)):                 # lambda_1, then lambda_2
            mb = axes[i, j].imshow(1.0 + contrast*im_, extent=ext, origin="lower",
                                   cmap="inferno", vmin=1.0 - 0.22*contrast,
                                   vmax=1.0 + contrast, interpolation="bilinear")
        md = axes[i, 2].imshow(contrast*slog, extent=ext, origin="lower", cmap="RdBu_r",
                               vmin=-contrast*dmax, vmax=contrast*dmax,
                               interpolation="bilinear")
        axes[i, 2].contour(X, Y, slog, levels=[0.0], colors="0.30", linewidths=0.4)
        axes[i, 2].text(0.04, 0.04, "peak %.2f" % (contrast*slog.max()),
                        transform=axes[i, 2].transAxes, fontsize=7, color="0.25")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for j, ttl in enumerate((r"$\lambda_1$: wide, hollow blob",
                             r"$\lambda_2$: small, hot blob",
                             r"$\lambda_2-\lambda_1$: the SLoG")):
        axes[0, j].set_title(ttl, pad=4)
    axes[0, 0].set_ylabel("perfect\nresolution")
    axes[1, 0].set_ylabel("smoothed with\na kernel of\nthe SLoG size")

    cb1 = fig.colorbar(mb, cax=cax1, ticks=[1.0, 1.0 + contrast])
    cb1.ax.set_yticklabels(["$B$", "$B+\\Delta$"])
    cb2 = fig.colorbar(md, cax=cax2, ticks=[-contrast*dmax, 0.0, contrast*dmax])
    cb2.ax.set_yticklabels(["$-\\Delta$", "0", "$+\\Delta$"])
    for c in (cb1, cb2):
        c.outline.set_linewidth(0.5)
        c.ax.tick_params(width=0.5, labelsize=7)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    make()
