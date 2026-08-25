# %%
# ------------------------------ parameters ---------------------------------
# Systems to compare, by name.  Everything the paper tabulates is available;
# print_catalogue() below lists them.
SYSTEMS = (
    "uMI Panorama",
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
SCAN_LENGTHS_CM = (5.0, 200.0, 61)     # first, last, how many
PROFILE_AT_CM = 100.0                  # scan length for the axial panels
RIPPLE_LIMIT = None                    # e.g. 1.2 for at most 20 % ripple, or None

# %%
# ------------------------------ setup --------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import slogpet as sp
from slogpet.data import load_systems

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
# -------------
# calculate and visualize the single bed axial efficiency profiles for each scanner

profiles = {}
for scanner in scanners:
    profile = sp.axial_profile(scanner, task.D_cyl, task.mu)
    profiles[scanner.name] = profile

fig, ax = plt.subplots(3,1, sharex=True, figsize=(7.0, 8.0), layout="constrained")
for scanner in scanners:
    reach = 0.55 * scanner.L_pet
    zz = np.linspace(-reach, reach, 401)
    y = profiles[scanner.name].samples(zz)
    # solid angle coverage only
    ax[0].plot(zz / 10, y, lw=1.8, label=label(scanner))
    # solid angle coverage times average detector-pair efficiency
    ax[1].plot(zz / 10, scanner.efficiency() * y, lw=1.8, label=label(scanner))
    # solid angle coverage times average detector-pair efficiency times SLoG resolution factor
    ax[2].plot(zz / 10, scanner.r(task) * scanner.efficiency() * y, lw=1.8, label=label(scanner))

for axx in ax:
    axx.set_ylim(bottom=0.0)

ax[-1].set_xlabel("axial position $z$ (cm)")
ax[0].set_ylabel(r"$\eta(z)$")
ax[1].set_ylabel(r"$\varepsilon \, \eta(z)$")
ax[2].set_ylabel(r"$r \, \varepsilon \, \eta(z)$")
ax[0].legend()
fig.show()


# %%
# calculate the optimal bed protocols for each scanner and scan length
protocols = {}

for scanner in scanners:
    pcol = []
    for S in scan_lengths:
        try:
            protocol = sp.optimal_protocol(profiles[scanner.name], float(S), max_peak_to_trough=RIPPLE_LIMIT)
            pcol.append(protocol)
        except ValueError:                   # the limit cannot be met here
            pcol.append(None)
                
    protocols[scanner.name] = pcol
    print("done:", label(scanner))


fig2, ax2 = plt.subplots(3,1, sharex=True, figsize=(7.0, 8.0), layout="constrained")
for scanner in scanners:
    ax2[0].plot(scan_lengths / 10, [p.n_beds if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=label(scanner))
    ax2[1].plot(scan_lengths / 10, [p.overlap if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=label(scanner))
    ax2[2].plot(scan_lengths / 10, [p.min_eta if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=label(scanner))

ax2[-1].set_xlabel("scan length $S$ (cm)")
ax2[0].set_ylabel(r"optimal number of beds")
ax2[1].set_ylabel(r"optimal overlap")
ax2[2].set_ylabel(r"$\min_z \eta_N(z)$")
ax2[0].legend()

fig2.show()

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

fig3, ax3 = plt.subplots(2,1, figsize=(7.0, 6.0), layout="constrained", sharex=True)
for scanner in scanners:
    ax3[0].plot(scan_lengths / 10,  snr2_min[scanner.name], lw=1.8, label=label(scanner))
    ax3[1].semilogy(scan_lengths / 10,  snr2_min[scanner.name], lw=1.8, label=label(scanner))

ax3[-1].set_xlabel(r"scan length $S$ (cm)")
ax3[0].set_ylabel(r"minimum SNR$^2$")
ax3[1].set_ylabel(r"minimum SNR$^2$")
ax3[0].legend()

fig3.show()
