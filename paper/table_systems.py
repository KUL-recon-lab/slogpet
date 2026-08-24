"""Table 2: detector-pair efficiencies and resolution factors of current systems."""
from typing import List

import numpy as np

from slogpet import Task

from .config import out, SYSTEMS, SLOG_SIZES, D_CYL_TAB, _size, _window

DEFAULT = out("table.tex")


def write(path: str = DEFAULT, L_s: float = 700.0) -> str:
    """Detector-pair efficiencies and resolution factors of current systems."""
    rows: List[str] = []
    prev_vendor = None
    dash = "---"
    for sc in SYSTEMS:
        vendor = sc.name.split()[0]
        if prev_vendor is not None and vendor != prev_vendor:
            rows.append(r"\addlinespace")
        prev_vendor = vendor
        Si = sc.S_ideal(L_s)
        eps = sc.efficiency(L_s)
        mrd = dash if np.isinf(sc.L_mrd) else f"{sc.L_mrd:.0f}"
        c_ps = dash if sc.ctr is None else f"{sc.ctr:.0f}"
        c_mm = dash if sc.ctr is None else f"{sc.F_t:.0f}"
        er = dash if sc.energy_resolution is None else f"{sc.energy_resolution:.1f}"
        star = lambda f: r"$^{\ast}$" if sc.is_assumed(f) else ""
        y_ = dash if sc.F_y is None else f"{sc.F_y:.1f}{star('F_y')}"
        z_ = dash if sc.F_z is None else f"{sc.F_z:.1f}{star('F_z')}"
        rr = [dash if sc.F_y is None or sc.F_z is None
              else f"{sc.r(Task(F_o, D_CYL_TAB)):.3f}" for F_o in SLOG_SIZES]
        rows.append(f"{sc.name} & {sc.L_pet:.0f} & {sc.D_pet:.0f} & {mrd} & "
                    f"{sc.crystal} & {_size(sc.crystal_size)} & "
                    f"{c_ps} & {c_mm} & {er} & {_window(sc.energy_window)} & "
                    f"{y_} & {z_} & "
                    f"{Si:.4f} & {sc.S_nema:.1f}$^{{{sc.reference}}}$ & {eps:.2f} & "
                    f"{rr[0]} & {rr[1]} " + r"\\")
    hd = lambda s: r"\multicolumn{1}{c}{\begin{tabular}{@{}c@{}}" + s + r"\end{tabular}}"
    header = [
        r"\begin{tabular}{@{}l rrr l l rr rc rr r rr rr@{}}", r"\toprule",
        r"& \multicolumn{3}{c}{geometry} & \multicolumn{6}{c}{detector}"
        r" & \multicolumn{2}{c}{resolution} & \multicolumn{3}{c}{sensitivity}"
        r" & \multicolumn{2}{c}{$r$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-10}\cmidrule(lr){11-12}"
        r"\cmidrule(lr){13-15}\cmidrule(l){16-17}",
        r"system & " + " & ".join([
            hd(r"$\Lpet$\\(mm)"), hd(r"$\Dpet$\\(mm)"), hd(r"$\Lmrd$\\(mm)"),
            hd(r"crystal\\ "), hd(r"size\\(mm$^3$)"),
            hd(r"CTR\\(ps)"), hd(r"CTR\\(mm)"),
            hd(r"$\Delta E/E$\\(\%)"), hd(r"window\\(keV)"),
            hd(r"$F_y$\\(mm)"), hd(r"$F_z$\\(mm)"),
            hd(r"$S_{\mathrm{ideal}}$\\ "), hd(r"$S_{\mathrm{NEMA}}$\\(cps/kBq)"),
            hd(r"$\epsilon$\\ "),
            hd(r"$F_{\mathrm{o}}{=}5$\\mm"), hd(r"$F_{\mathrm{o}}{=}10$\\mm")]) + r" \\",
        r"\midrule"]
    open(path, "w").write("\n".join(header + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    print("wrote", path)
    return path


if __name__ == "__main__":
    write()
