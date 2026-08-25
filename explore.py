# %%
import numpy as np
import matplotlib.pyplot as plt

import slogpet as sp
from slogpet.data import load_systems

plt.rcParams.update({"figure.figsize": (7.0, 3.6), "axes.grid": True,
                     "grid.color": "0.88", "grid.linewidth": 0.5,
                     "figure.constrained_layout.use": True})

all_predified_systems = load_systems()

print("predefined systems")
print("------------------")
for scanner in all_predified_systems:
    print(scanner.label)

print()


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
#CUSTOM_SCANNER = None
CUSTOM_SCANNERS = [sp.Scanner(name="my scanner", L_pet=1200.0, D_pet=760.0, F_y=3.5, F_z=3.5, ctr=200.0, S_nema=250.0)]

# The detection task
task = sp.Task(F_o = 5.0, D_cyl = 200.0, mu = 0.0096)   # mu is the linear attenuation coefficient of water at 511 keV, mm^-1

# The acquisition
scan_lengths_mm = np.linspace(50.0, 2000.0, 41)
RIPPLE_LIMIT = None                    # e.g. 1.2 for at most 20 % ripple, or None

# %%
# ------------------------------ setup --------------------------------------

scanners = []
for name in SYSTEMS:
    hits = [s for s in all_predified_systems if s.name == name]
    if not hits:
        raise SystemExit("no system called %r; run print_catalogue()" % name)
    scanners.append(hits[0])                  # the first, where a name repeats

if CUSTOM_SCANNERS is not None:
    scanners.extend(CUSTOM_SCANNERS)


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
fig.show()


# %%
# calculate the optimal bed protocols for each scanner and scan length
protocols = {}

for scanner in scanners:
    pcol = []
    for S in scan_lengths_mm:
        try:
            protocol = sp.optimal_protocol(profiles[scanner.name], float(S), max_peak_to_trough=RIPPLE_LIMIT)
            pcol.append(protocol)
        except ValueError:                   # the limit cannot be met here
            pcol.append(None)
                
    protocols[scanner.name] = pcol


fig2, ax2 = plt.subplots(3,1, sharex=True, figsize=(7.0, 8.0), layout="constrained")
for scanner in scanners:
    ax2[0].plot(scan_lengths_mm / 10, [p.n_beds if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=scanner.label)
    ax2[1].plot(scan_lengths_mm / 10, [p.overlap if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=scanner.label)
    ax2[2].plot(scan_lengths_mm / 10, [p.min_eta if p is not None else np.nan for p in protocols[scanner.name]], lw=1.8, label=scanner.label)

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
    ax3[0].plot(scan_lengths_mm / 10,  snr2_min[scanner.name], lw=1.8, label=scanner.label)
    ax3[1].semilogy(scan_lengths_mm / 10,  snr2_min[scanner.name], lw=1.8, label=scanner.label)

ax3[-1].set_xlabel(r"scan length $S$ (cm)")
ax3[0].set_ylabel(r"minimum SNR$^2$")
ax3[1].set_ylabel(r"minimum SNR$^2$")
ax3[0].legend()

fig3.show()
