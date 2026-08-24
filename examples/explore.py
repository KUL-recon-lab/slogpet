# %% [markdown]
# # SLoG detectability of PET systems
#
# Everything you might want to change is in the next cell.  Edit it, run
# everything, and the figures and the table below follow.  The calculation is the
# same `slogpet` package the paper uses -- nothing here computes physics of its
# own.
#
# This file is a plain Python script.  It runs at a terminal with
# `python examples/explore.py`, and it is the source the notebook on the web page
# is generated from, so the two cannot drift apart.

# %%
# ------------------------------ parameters ---------------------------------
# Systems to compare, by name.  Everything the paper tabulates is available;
# print_catalogue() below lists them.
SYSTEMS = (
    "Biograph Vision Quadra",
    "Omni 128",
    "uMI Panorama GS",
)

# A system of your own, or None.  Give either epsilon (the detector-pair
# efficiency) or S_nema (the NEMA NU 2 sensitivity in cps/kBq, 70 cm line
# source); the package works the other one out from the geometry.
CUSTOM = None
#CUSTOM = dict(name="my scanner", L_pet=1200.0, D_pet=760.0,
#               F_y=3.5, F_z=3.5, ctr=200.0, S_nema=250.0)

# The detection task
F_O_MM = 5.0                  # SLoG size, mm FWHM
D_CYL_MM = 200.0              # water cylinder diameter, mm

# The acquisition
SCAN_LENGTHS_CM = (5.0, 200.0, 41)     # first, last, how many
PROFILE_AT_CM = 100.0                  # scan length for the axial panels
RIPPLE_LIMIT = None                    # e.g. 1.2 for at most 20 % ripple, or None

# What to draw in the single-bed panel: the profile alone, or multiplied by the
# detector-pair efficiency and the resolution factor.
MULTIPLY_BY_EPSILON = False
MULTIPLY_BY_R = False

# %%
# ------------------------------ setup --------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import slogpet as sp
from slogpet.data import load_systems

COLOURS = ("#2a78d6", "#eda100", "#d55181", "#008300")   # colour-vision checked
plt.rcParams.update({"figure.figsize": (7.0, 3.6), "axes.grid": True,
                     "grid.color": "0.88", "grid.linewidth": 0.5,
                     "figure.constrained_layout.use": True})


def print_catalogue():
    """Every system name that SYSTEMS above will accept."""
    for scanner in load_systems():
        print("%-28s %5.0f cm" % (scanner.name, scanner.L_pet / 10))


def chosen_scanners():
    """SYSTEMS resolved against the catalogue, plus CUSTOM if there is one."""
    catalogue = load_systems()
    out = []
    for name in SYSTEMS:
        hits = [s for s in catalogue if s.name == name]
        if not hits:
            raise SystemExit("no system called %r; run print_catalogue()" % name)
        out.append(hits[0])                  # the first, where a name repeats
    if CUSTOM:
        out.append(sp.Scanner(**CUSTOM))
    return out


def label(scanner):
    return "%s (%.0f cm)" % (scanner.name, scanner.L_pet / 10)


scanners = chosen_scanners()
task = sp.Task(F_O_MM, D_CYL_MM)
first, last, count = SCAN_LENGTHS_CM
scan_lengths = np.linspace(first * 10, last * 10, int(count))
print("%d configurations, %d scan lengths, SLoG %.1f mm in a %.0f cm cylinder"
      % (len(scanners), len(scan_lengths), F_O_MM, D_CYL_MM / 10))

# %%
# ------------------------------ the calculation -----------------------------
# One axial profile per system, then the optimised bed protocol at every scan
# length.  A scan length where no arrangement can meet RIPPLE_LIMIT gives NaN,
# so the curve breaks there instead of the run failing.
profiles = {}
snr2_min = {}
for scanner in scanners:
    profile = sp.axial_profile(scanner, task.D_cyl, task.mu)
    profiles[scanner.name] = profile
    values = []
    for S in scan_lengths:
        try:
            protocol = sp.optimal_protocol(profile, float(S),
                                           max_peak_to_trough=RIPPLE_LIMIT)
        except ValueError:                   # the limit cannot be met here
            values.append(np.nan)
            continue
        values.append(sp.snr2_value(scanner.efficiency(), protocol.min_eta,
                                    scanner.r(task), task.F_o))
    snr2_min[scanner.name] = np.array(values)
    print("done:", label(scanner))

# %%
# ------------------------------ detectability vs scan length ----------------
fig, ax = plt.subplots()
for scanner, colour in zip(scanners, COLOURS):
    ax.plot(scan_lengths / 10, snr2_min[scanner.name], color=colour, lw=1.8,
            label=label(scanner))
ax.set_xlabel("scan length $S$ (cm)")
ax.set_ylabel(r"SNR$^2$ (minimum over the range)")
ax.set_ylim(bottom=0.0)
ax.legend()
ax.set_title("worst point of the scan range, with the best bed protocol"
             + ("" if RIPPLE_LIMIT is None else
                r", max/min $\leq$ %g" % RIPPLE_LIMIT))
fig.show()

# %%
# ------------------------------ along the axis ------------------------------
# The ripple is the bed structure: the curves above are the minima of these.
S = PROFILE_AT_CM * 10
z = np.linspace(-S / 2, S / 2, 401)
fig, ax = plt.subplots()
for scanner, colour in zip(scanners, COLOURS):
    try:
        protocol = sp.optimal_protocol(profiles[scanner.name], S,
                                       max_peak_to_trough=RIPPLE_LIMIT)
    except ValueError:
        print("no arrangement meets the limit at %g cm:" % PROFILE_AT_CM,
              label(scanner))
        continue
    eta = sp.tile_beds(profiles[scanner.name].samples, protocol.n_beds,
                       protocol.spacing, z)
    ax.plot(z / 10, sp.snr2_value(scanner.efficiency(), eta, scanner.r(task),
                                  task.F_o),
            color=colour, lw=1.8,
            label="%s, %d bed%s" % (label(scanner), protocol.n_beds,
                                    "" if protocol.n_beds == 1 else "s"))
ax.set_xlabel("axial position $z$ (cm)")
ax.set_ylabel(r"SNR$^2(z)$")
ax.set_ylim(bottom=0.0)
ax.legend()
ax.set_title("scan length %g cm" % PROFILE_AT_CM)
fig.show()

# %%
# ------------------------------ one bed position ----------------------------
# What every protocol is built from.  Multiply by epsilon and r above to see
# where the systems actually separate: the bare profiles are nearly identical.
fig, ax = plt.subplots()
for scanner, colour in zip(scanners, COLOURS):
    reach = 0.55 * scanner.L_pet
    zz = np.linspace(-reach, reach, 401)
    y = profiles[scanner.name].samples(zz)
    if MULTIPLY_BY_EPSILON:
        y = y * scanner.efficiency()
    if MULTIPLY_BY_R:
        y = y * scanner.r(task)
    ax.plot(zz / 10, y, color=colour, lw=1.8, label=label(scanner))
ax.set_xlabel("axial position $z$ (cm)")
ax.set_ylabel(("ε " if MULTIPLY_BY_EPSILON else "") + r"$\eta(z)$"
              + (" r" if MULTIPLY_BY_R else ""))
ax.set_ylim(bottom=0.0)
ax.legend()
ax.set_title("a single bed position"
             + (", times ε" if MULTIPLY_BY_EPSILON else "")
             + (", times r" if MULTIPLY_BY_R else ""))
fig.show()

# %%
# ------------------------------ the numbers ---------------------------------
# At PROFILE_AT_CM.  "~" marks a NEMA sensitivity implied by epsilon rather than
# published.
header = ("configuration", "L_PET", "D_PET", "CTR", "S_NEMA", "eps", "r",
          "beds", "overlap", "SNR2_min")
print("%-30s %6s %6s %6s %8s %6s %8s %5s %8s %10s" % header)
print("%-30s %6s %6s %6s %8s %6s %8s %5s %8s %10s"
      % ("", "cm", "cm", "ps", "cps/kBq", "", "", "", "%", ""))
for scanner in scanners:
    nema = (scanner.S_nema if scanner.S_nema is not None
            else 1000 * scanner.S_ideal() * scanner.efficiency())
    marker = "" if scanner.S_nema is not None else "~"
    ctr = ("—" if scanner.F_t is None
           else "%.0f" % (scanner.F_t / sp.C_OVER_2))
    try:
        protocol = sp.optimal_protocol(profiles[scanner.name], S,
                                       max_peak_to_trough=RIPPLE_LIMIT)
        beds = "%d" % protocol.n_beds
        overlap = "—" if protocol.overlap is None else "%.0f" % protocol.overlap
        value = "%10.4g" % sp.snr2_value(scanner.efficiency(), protocol.min_eta,
                                         scanner.r(task), task.F_o)
    except ValueError:
        beds = overlap = value = "—"
    print("%-30s %6.1f %6.1f %6s %8s %6.3f %8.4g %5s %8s %10s"
          % (label(scanner), scanner.L_pet / 10, scanner.D_pet / 10, ctr,
             marker + "%.1f" % nema, scanner.efficiency(), scanner.r(task),
             beds, overlap, value))
