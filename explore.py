# %% [markdown]
# # Comparing PET systems on a resolution-dependent task
#
# How well a PET system detects a small lesion depends on three things that pull
# against each other: how much of the emitted signal it catches, how sharply it
# resolves the lesion, and how evenly it covers the range being scanned.  
#
# **To use it:** edit the parameters in the second cell, then run everything.
# Nothing is sent anywhere; the calculation runs in your own browser.
#
# **To keep a figure:** right-click it and *Save image as...* -- the figures
# are SVG, so they stay sharp at any size.  `fig.savefig("profiles.svg")` in
# a cell writes the file next to the notebook instead, where the file browser
# on the left can download it.
#
# ## Setting up
#
# The list printed below is every system the package knows about; the names are
# what the next cell expects.

# %%
import numpy as np
import matplotlib.pyplot as plt

import slogpet as sp
from slogpet.data import load_systems

# Draw the figures in a notebook as SVG rather than PNG: sharp at any zoom, and
# saved as vectors.  Plain Python rather than the %config magic, so that this
# file stays runnable as a script; outside a notebook there is no inline
# backend and this quietly does nothing.
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats("svg")
except Exception:
    pass

plt.rcParams.update({"figure.figsize": (7.0, 3.6), "axes.grid": True,
                     "grid.color": "0.88", "grid.linewidth": 0.5,
                     "figure.constrained_layout.use": True})

all_predified_systems = load_systems()

print("predefined systems")
print("------------------")
for scanner in all_predified_systems:
    print(scanner.label)

print()


# %% [markdown]
# ## What to compare, and on what task
#
# Everything you are likely to want to change is here.
#
# * `system_names` -- published systems, by name, from the list above.
# * `custom_scanners` -- a system of your own.  Give either `epsilon`, the
#   detector-pair efficiency, or `S_nema`, the NEMA NU 2 sensitivity in cps/kBq;
#   the package derives one from the other and the geometry.
# * `task` -- the lesion (`F_o`, its size as a Gaussian FWHM in mm) and the body
#   it sits in (`D_cyl`, a water cylinder).  A smaller lesion asks more of the
#   resolution, so which system wins can change with `F_o`.
# * `scan_lengths_mm` -- the axial range to cover, swept from a single organ to
#   a total-body scan.
# * `ripple_limit` -- how uneven the sensitivity along the range is allowed to
#   be.  `None` lets the optimiser maximise the worst point regardless of the
#   ripple, which is what the paper does; `1.2` insists the best point is at
#   most 20 % above the worst, which costs sensitivity and usually more beds.
#
# Re-running this cell rebuilds `scanners`, so an edit here is picked up by
# everything below.

# %%
# setup scanners to be compared

# names of predefined systems to compare; must be in the catalogue
system_names = (
    "uMI Panorama",
    "Biograph Vision Quadra",
    "Omni 128",
    "uMI Panorama GS",
)

# list of custom scanners to compare; must be instances of slogpet.types.Scanner
custom_scanners = [sp.Scanner(name="my scanner", L_pet=700.0, D_pet=760.0, F_y=3.5, F_z=3.5, ctr=200.0, epsilon=0.2)]
#custom_scanners = None

# defined the SLoG task to be used for the comparison
task = sp.Task(F_o = 5.0, D_cyl = 200.0, mu = 0.0096)   # mu is the linear attenuation coefficient of water at 511 keV, mm^-1

# The acquisition
scan_lengths_mm = np.linspace(50.0, 2000.0, 41)

# max ripple amplitude of the sensitivity axial profile (max_sens(z) / min_sens(z)) allowed in the optimal bed protocol; None means no limit
ripple_limit = None   

# %%
# setup the scanners to be compared, from the names above and any custom ones

scanners = []
for name in system_names:
    hits = [s for s in all_predified_systems if s.name == name]
    if not hits:
        raise SystemExit("no system called %r; run print_catalogue()" % name)
    scanners.append(hits[0])                  # the first, where a name repeats

if custom_scanners is not None:
    scanners.extend(custom_scanners)


# %% [markdown]
# ## One bed position: where the systems separate
#
# `eta(z)` is the fraction of a point source's emissions the detector sees at
# axial position `z`, for a single bed position.  It peaks at the centre of the
# detector and falls to zero at its ends, and it is what every multi-bed
# acquisition is built from.
#
# The three panels multiply it up, one factor at a time:
#
# 1. `eta(z)` alone -- geometry.  Note how little the *peaks* differ: a longer
#    detector buys width, not height.
# 2. times `epsilon`, the detector-pair efficiency -- how many of the pairs that
#    reach the crystals are actually recorded.
# 3. times `r`, the resolution factor -- how much of the signal survives the
#    system's spatial and timing resolution for a lesion of this size.  For a
#    system without time of flight this depends on the body diameter too.
#
# The systems look alike in the first panel and fan out by a large factor in the
# third: the profile is not where the difference lives.

# %%
# -------------
# calculate and visualize the single bed axial efficiency profiles for each scanner

single_bed_eta_profiles = {}
for scanner in scanners:
    single_bed_eta_profiles[scanner.name] = sp.axial_profile(scanner, task.D_cyl, task.mu)


# visualize the single bed axial efficiency profiles for each scanner
fig, ax = plt.subplots(3,1, sharex=True, figsize=(7.0, 8.0))
for scanner in scanners:
    reach = 0.55 * scanner.L_pet
    zz = np.linspace(-reach, reach, 401)
    y = single_bed_eta_profiles[scanner.name].samples(zz)
    # solid angle coverage only
    ax[0].plot(zz / 10, y, lw=1.8, label=scanner.label)
    # solid angle coverage times average detector-pair efficiency
    ax[1].plot(zz / 10, scanner.efficiency() * y, lw=1.8, label=scanner.label)
    # solid angle coverage times average detector-pair efficiency times SLoG resolution factor
    ax[2].plot(zz / 10, scanner.r(task) * scanner.efficiency() * y, lw=1.8, label=scanner.label)

for axx in ax:
    axx.set_ylim(bottom=0.0)

ax[-1].set_xlabel("axial position $z$ (cm)")
ax[0].set_ylabel(r"$\eta(z)$")
ax[1].set_ylabel(r"$\varepsilon \, \eta(z)$")
ax[2].set_ylabel(r"$r \, \varepsilon \, \eta(z)$")
ax[0].legend()


# %% [markdown]
# ## Covering a range: how many bed positions, and how far apart
#
# A scan range longer than the detector is covered by moving the patient through
# in steps, each acquired for the same fraction of the total time.  The tiled
# profile `eta_N(z)` ripples -- highest where neighbouring bed positions overlap,
# lowest between them -- and what matters clinically is the worst point.  So the
# package maximises `min eta_N` over the range, over both the number of beds and
# their spacing, and prefers a smaller bed count when it is within 3 % of the
# best.
#
# The panels show what it chose: the bed count, the overlap between neighbouring
# positions, and the sensitivity at the worst point.  The bed count is a step
# function of the scan length, and the overlap is not monotone -- both are
# consequences of the ripple, not of the search.
#
# With a `ripple_limit` set, a scan length where no arrangement can meet it has
# no protocol at all; those points come back as `None` and leave a gap.

# %%
# calculate the optimal bed protocols for each scanner and scan length
protocols = {}

for scanner in scanners:
    pcol = []
    for S in scan_lengths_mm:
        try:
            protocol = sp.optimal_protocol(single_bed_eta_profiles[scanner.name], float(S), max_peak_to_trough=ripple_limit)
            pcol.append(protocol)
        except ValueError:                   # the limit cannot be met here
            pcol.append(None)
                
    protocols[scanner.name] = pcol


fig2, ax2 = plt.subplots(3,1, sharex=True, figsize=(7.0, 8.0))
for scanner in scanners:
    ax2[0].plot(scan_lengths_mm / 10, [p.n_beds if p is not None else np.nan for p in protocols[scanner.name]], drawstyle='steps-post', lw=1.8, label=scanner.label)
    ax2[1].plot(scan_lengths_mm / 10, [p.overlap if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=scanner.label)
    ax2[2].plot(scan_lengths_mm / 10, [p.min_eta if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=scanner.label)

ax2[-1].set_xlabel("scan length $S$ (cm)")
ax2[0].set_ylabel(r"optimal number of beds")
ax2[1].set_ylabel(r"optimal overlap")
ax2[2].set_ylabel(r"$\min_z \eta_N(z)$")
ax2[0].legend()


# %% [markdown]
# ## Detectability
#
# Putting it together.  The squared signal-to-noise ratio of the Hotelling
# observer for this task factorises as
#
# `SNR^2(z) = T x [S^2 sigma_o^3 / (16 pi sqrt(pi) B)] x epsilon eta_N(z) x r`
#
# where the first bracket describes the patient and the acquisition time and is
# the same for every system compared.  What is plotted below is the system's own
# part, `epsilon eta_N r sigma_o^3`, at the worst point of the scan range -- the
# quantity that actually distinguishes one scanner from another.
#
# The log panel is the useful one for comparing systems that differ by more than
# a factor of a few.

# %%
# calculate the minimum SNR^2 for each scanner and scan length, using the optimal protocols

snr2_min = {}

for scanner in scanners:
    values = []
    for protocol in protocols[scanner.name]:
        if protocol is None:
            values.append(np.nan)
        else:
            values.append(sp.snr2_value(scanner.efficiency(), protocol.min_eta,
                                        scanner.r(task), task.F_o))

    snr2_min[scanner.name] = np.array(values)

fig3, ax3 = plt.subplots(2,1, figsize=(7.0, 6.0), sharex=True)
for scanner in scanners:
    ax3[0].plot(scan_lengths_mm / 10,  snr2_min[scanner.name], lw=1.8, label=scanner.label)
    ax3[1].semilogy(scan_lengths_mm / 10,  snr2_min[scanner.name], lw=1.8, label=scanner.label)

ax3[-1].set_xlabel(r"scan length $S$ (cm)")
ax3[0].set_ylabel(r"minimum SNR$^2$")
ax3[1].set_ylabel(r"minimum SNR$^2$")
ax3[0].legend()


# %% [markdown]
# ---
# The figures above are drawn as each cell runs.  The call below matters only at
# a terminal: `python explore.py` would otherwise exit and take its windows with
# it.  In a notebook, and in ipython with `%matplotlib`, the figures have already
# been shown and this returns at once.

# %%
plt.show()
