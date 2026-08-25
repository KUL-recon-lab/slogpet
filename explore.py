# %% [markdown]
# # Comparing PET systems on a resolution-dependent task
#
# How well a PET system detects a small lesion depends on three things that pull
# against each other: how much of the emitted signal it catches, how sharply it
# resolves the lesion, and how evenly it covers the range being scanned.  This
# notebook works all three out for whichever systems you choose, using the same
# `slogpet` package as the paper -- there is no physics in this file, only
# choices and plots.
#
# **To use it:** edit the parameters in the second cell, then run everything.
# Nothing is sent anywhere; the calculation runs in your own browser.
#
# ## Setting up
#
# The table below is every system the package knows about, with the parameters
# the model actually uses.  The names in the first column are what the next cell
# expects.  A blank cell means the system has no such number -- no time of
# flight, no ring-difference limit -- rather than a measurement nobody made.

# %%
from itertools import cycle

import numpy as np

import slogpet as sp
from slogpet.data import load_systems

# The figures are drawn with bokeh, so they can be zoomed, panned and read off
# by hovering.  plots.py next to this file holds that plumbing -- where a figure
# goes, how a panel is set up, the colours -- so that what follows is about the
# physics rather than about plot construction.
from plots import (COLOURS, draw, panel, series, shared_legend, show_table,
                   systems_table)

all_predified_systems = load_systems()

show_table(systems_table(all_predified_systems))


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
custom_scanners = [sp.Scanner(name="my-scanner-1", L_pet=300.0, D_pet=760.0, F_y=4.0, F_z=4.0, ctr=120.0, epsilon=0.2),
                   sp.Scanner(name="my-scanner-1", L_pet=250.0, D_pet=760.0, F_y=3.8, F_z=3.8, ctr=100.0, S_nema=15.0)]
#custom_scanners = None

# defined the SLoG task to be used for the comparison
task = sp.Task(F_o = 5.0, D_cyl = 200.0, mu = 0.0096)   # mu is the linear attenuation coefficient of water at 511 keV, mm^-1

# The acquisition
scan_lengths_mm = np.linspace(50.0, 2000.0, 41)

# max ripple amplitude of the sensitivity axial profile (max_sens(z) / min_sens(z)) allowed in the optimal bed protocol; None means no limit
ripple_limit = None   

# %%
# setup selected predified systems and any custom scanners for comparison

scanners = []
for name in system_names:
    hits = [s for s in all_predified_systems if s.name == name]
    if not hits:
        raise SystemExit("no system called %r; run print_catalogue()" % name)
    scanners.append(hits[0])                  # the first, where a name repeats

if custom_scanners is not None:
    scanners.extend(custom_scanners)


# %% [markdown]
# ## What the task alone decides
#
# The same table, with the systems you defined yourself added to it, and two
# columns that only mean anything once the task is fixed:
#
# * $r$, the resolution factor -- how much of a lesion this size survives the
#   system's spatial and timing resolution.  A system without time of flight
#   also pays for the body diameter here.
# * $\varepsilon \, r$, the product of the two things a system brings to the
#   task, before any acquisition is chosen.
#
# Neither depends on the scan length or on the bed protocol, so this is the
# ranking to beat: everything after this is about how much of it survives over
# a long range.

# %%
show_table(systems_table(list(all_predified_systems) + list(custom_scanners or []),
                         task))


# %% [markdown]
# ## One bed position: where the systems separate
#
# $\eta(z)$ is the fraction of a point source's emissions the detector sees at
# axial position $z$, for a single bed position.  It peaks at the centre of the
# detector and falls to zero at its ends, and it is what every multi-bed
# acquisition is built from.
#
# The three panels multiply it up, one factor at a time:
#
# 1. $\eta(z)$ alone -- geometry.  Note how little the *peaks* differ: a longer
#    detector buys width, not height.
# 2. times $\varepsilon$, the detector-pair efficiency -- how many of the pairs
#    that reach the crystals are actually recorded.
# 3. times $r$, the resolution factor -- how much of the signal survives the
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
geometry = panel("eta(z)")
with_eps = panel("eps * eta(z)", x_range=geometry.x_range)
with_r = panel("r * eps * eta(z)", "axial position z (cm)",
               x_range=geometry.x_range)

curves = {}                      # label -> its line in each of the three panels
for scanner, colour in zip(scanners, cycle(COLOURS)):
    reach = 0.55 * scanner.L_pet
    zz = np.linspace(-reach, reach, 401)
    y = single_bed_eta_profiles[scanner.name].samples(zz)
    style = dict(color=colour, line_width=2, name=scanner.label)
    curves[scanner.label] = [
        # solid angle coverage only
        geometry.line(zz / 10, y, **style),
        # solid angle coverage times average detector-pair efficiency
        with_eps.line(zz / 10, scanner.efficiency() * y, **style),
        # solid angle coverage times average detector-pair efficiency times SLoG resolution factor
        with_r.line(zz / 10, scanner.r(task) * scanner.efficiency() * y, **style),
    ]

for p in (geometry, with_eps, with_r):
    p.y_range.start = 0.0
shared_legend(geometry, curves)
draw([geometry, with_eps, with_r], "explore-single-bed.html")


# %% [markdown]
# ## Covering a range: how many bed positions, and how far apart
#
# A scan range longer than the detector is covered by moving the patient through
# in steps, each acquired for the same fraction of the total time.  The tiled
# profile $\eta_N(z)$ ripples -- highest where neighbouring bed positions
# overlap, lowest between them -- and what matters clinically is the worst point.
# So the package maximises $\min_z \eta_N(z)$ over the range, over both the
# number of beds and their spacing, and prefers a smaller bed count when it is
# within $3\,\%$ of the best.
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


beds = panel("optimal number of beds")
overlap = panel("optimal overlap (%)", x_range=beds.x_range)
worst = panel("min eta_N(z) over the range", "scan length S (cm)",
              x_range=beds.x_range)

curves = {}
for scanner, colour in zip(scanners, cycle(COLOURS)):
    style = dict(color=colour, line_width=2, name=scanner.label)
    protocol = protocols[scanner.name]
    curves[scanner.label] = [
        beds.step(scan_lengths_mm / 10, series(protocol, "n_beds"),
                  mode="after", **style),
        overlap.line(scan_lengths_mm / 10, series(protocol, "overlap"), **style),
        worst.line(scan_lengths_mm / 10, series(protocol, "min_eta"), **style),
    ]

shared_legend(beds, curves)
draw([beds, overlap, worst], "explore-protocols.html")


# %% [markdown]
# ## Detectability
#
# Putting it together.  The squared signal-to-noise ratio of the Hotelling
# observer for this task factorises as
#
# $$\mathrm{SNR}^2(z) \;=\; T \;\times\;
#   \frac{\dot{S}^2\,\sigma_\mathrm{o}^3}{16\pi\sqrt{\pi}\,\dot{B}}
#   \;\times\; \varepsilon\,\eta_N(z) \;\times\; r$$
#
# where $T$ is the acquisition time and the fraction describes the patient --
# both the same for every system compared.  What is plotted below is the
# system's own part, $\varepsilon\,\eta_N\,r\,\sigma_\mathrm{o}^3$, at the
# worst point of the scan range -- the quantity that actually distinguishes one
# scanner from another.
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

linear = panel("minimum SNR^2")
logarithmic = panel("minimum SNR^2", "scan length S (cm)",
                    x_range=linear.x_range, y_axis_type="log")

curves = {}
for scanner, colour in zip(scanners, cycle(COLOURS)):
    style = dict(color=colour, line_width=2, name=scanner.label)
    curves[scanner.label] = [
        linear.line(scan_lengths_mm / 10, snr2_min[scanner.name], **style),
        logarithmic.line(scan_lengths_mm / 10, snr2_min[scanner.name], **style),
    ]

linear.y_range.start = 0.0
shared_legend(linear, curves, "top_right")
draw([linear, logarithmic], "explore-detectability.html")


# %% [markdown]
# ---
# Each figure has a toolbar: drag to pan, box- or wheel-zoom, hover to read a
# curve off, and the save button writes a PNG.  Clicking a name in the legend
# hides that system, which helps when several overlap.  The three panels of a
# figure share their x axis, so zooming one zooms all of them.
#
# Run as a script rather than in a notebook, the same three figures are written
# next to this file as `explore-*.html` and opened in a browser tab.
