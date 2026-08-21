# `slogpet`

Pure implementation of the model behind *Comparing short and long PET systems on
a resolution-dependent task*.

## The short way

    import slogpet as sp

    quadra = sp.Scanner("Quadra", L_pet=1060.0, D_pet=820.0,
                        F_y=3.4, F_z=3.8, ctr=230.0, S_nema=175.3)
    task   = sp.Task(F_o=5.0, D_cyl=300.0)          # 5 mm SLoG in a 30 cm cylinder

    res = sp.snr2(quadra, task, scan_length=1000.0)
    res.epsilon, res.r, res.snr2_min                # 0.276, 0.167, 6.52e-3
    res.n_beds, res.overlap                         # 2 beds, 35 % overlap
    res.z, res.eta_N, res.snr2                      # the profile over the scan range

`Scanner` takes either `ctr` in ps or `F_t` in mm (`F_t = 0.15 mm/ps x ctr`);
`F_t is None` means no time of flight, and the resolution factor is then
evaluated in its object-dependent non-TOF form. It takes either `epsilon`
directly or `S_nema`, from which the efficiency is derived.

To sweep a parameter, build the axial profile once -- it depends on neither the
SLoG size nor the scan length, and it is the only expensive step:

    sp.snr2_curve(quadra, task, scan_lengths=np.arange(50., 2000., 25.))

`sp.axial_profile` is memoised, so an interactive frontend that changes the SLoG
size or the scan length pays for it only once per (scanner, cylinder).

## The long way

    sp.eta(z, L_pet, D_pet, D_cyl, L_mrd=np.inf)   # axial coverage x attenuation
    sp.S_ideal_closed(L_pet, D_pet, L_s, L_mrd)    # NEMA line-source sensitivity
    sp.r_tof(F_t, F_y, F_z, F_o)                   # resolution factor, TOF
    sp.r_nontof(F_y, F_z, F_o, D_cyl)              # resolution factor, no TOF
    sp.eta_lattice(L_pet, D_pet, D_cyl)            # -> Lattice, eta on a 1 mm grid
    sp.optimise_beds(lattice, L_pet, S)            # -> (N*, spacing*, min eta_N)

## Published systems

    from slogpet.data import load_systems, load_detector_groups, load_references

    for sc in load_systems():
        print(sc.name, sc.crystal, sc.efficiency(), sc.reference)

`data/systems.json` holds ten published configurations with the source of every
number, a free-text `note` where a value needs one, and an `assumed` list naming
the fields that were taken from a similar system rather than measured.
`data/detectors.json` holds the four generic detector designs and the lengths
they are offered in. JSON rather than YAML so the package needs nothing outside
the standard library, and so the provenance is queryable rather than buried in
comments.

`slogpet.data.verify()` checks referential integrity: every reference number has
a source, every design points at a listed system, every style is defined, every
system has an efficiency.

## The bed search

`eta` is tabulated on a 1 mm lattice and bed spacings are multiples of 2 mm, so
that every bed position is an exact number of lattice steps. `eta_N` is then
piecewise linear with all its knots on the lattice, which means its minimum over
the scan range is attained *at* a lattice point: the reported minimum is the
minimum, not an estimate of it. Evaluating it becomes a gather rather than an
interpolation, and the offset search can be exhaustive at 1 cm before refining,
so it cannot settle in the wrong local optimum. About thirty times faster than
the continuous search it replaced, and slightly more accurate.

Both steps are module constants — `slogpet.H_LATTICE` and `slogpet.H_SEARCH` —
and `eta_lattice(..., h=0.25)` will refine the first if you want to check that
the answer does not depend on it. It does not: a fourfold refinement moves the
minimum sensitivity by well under a per cent.

## Conventions

Lengths in mm, coincidence timings in ps, all resolutions as FWHM,
attenuation coefficients in mm^-1.

Every function is pure. Geometry is always passed in explicitly; there is no
module-level mutable state, no caching, no printing and no file I/O, and nothing
imports matplotlib. That is what lets the paper's figure scripts, a web frontend
and a notebook all sit on the identical code path.

`snr2` returns the system part of the Hotelling SNR^2,
`epsilon x eta_N(z) x r x sigma_o^3`, in units of `T Sdot^2 / (16 pi sqrt(pi) Bdot)`;
the omitted factors are properties of the patient and of the acquisition time and
are the same for every system being compared. Pass `scale` to put them back.

`slogpet.validation` is the exception that proves the rule: it prints a report,
because its only job is to check the analytic results against Monte Carlo and
against a direct numerical evaluation of Eq. (53) of Nuyts et al. (2025).

Dependencies: numpy and scipy. Both are available in Pyodide, so the package
runs unmodified in the browser.
