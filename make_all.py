#!/usr/bin/env python3
"""Regenerate everything the paper needs.

    python3 make_all.py                # every figure and table
    python3 make_all.py --check        # also the cross-checks
    python3 make_all.py --numbers      # also print the numbers behind Figure 7
    python3 make_all.py --only beds ripple
    python3 make_all.py --show --only beds ripple    # windows, write nothing
    python3 make_all.py --list

Each target is one module in ``paper/`` and can also be run on its own, e.g.
``python3 -m paper.fig_beds``.  All physics comes from the ``slogpet`` package;
nothing here computes anything.

Every figure module takes ``path=None`` to open a window instead of writing a
file, which is what ``--show`` does here and what is meant for an ipython
prompt::

    In [1]: from paper.fig_beds import make
    In [2]: fig = make(None)

The full run takes about nine seconds; it took two minutes before the bed
search moved onto a lattice.
"""
import argparse
import sys
import time
from typing import List, Optional, Sequence, Tuple

# (name, module, entry point, roughly how long it takes)
TARGETS: List[Tuple[str, str, str, str]] = [
    ("table",     "paper.table_systems", "write", "fast"),
    ("slog",      "paper.fig_slog",      "make",  "fast"),
    ("scenarios", "paper.fig_scenarios", "make",  "slow"),
    ("profile",   "paper.fig_profile",   "make",  "fast"),
    ("beds",      "paper.fig_beds",      "make",  "slow"),
    ("ripple",    "paper.fig_ripple",    "make",  "fast"),
]

# the order matters only in that Figure 1 uses a different matplotlib backend;
# it is generated before the pgf figures, as it always has been


def run(names: Sequence[str], show: bool = False) -> None:
    """Regenerate the named targets, or draw them on screen if *show*.

    On screen the figures are all opened at once and the run blocks on the last
    one, so that the windows can be compared side by side; the table has no
    on-screen form and is skipped.
    """
    import importlib
    if show:
        from paper.style import use_screen
        plt = use_screen()
        plt.ion()                      # so that each show() returns at once
    total = 0.0
    for name, mod, entry, _ in TARGETS:
        if name not in names:
            continue
        if show and entry != "make":
            print("   (%s is a table, nothing to show)" % name)
            continue
        t0 = time.time()
        getattr(importlib.import_module(mod), entry)(*([None] if show else []))
        dt = time.time() - t0
        total += dt
        print("   %5.1f s" % dt)
    print("\ntotal %.1f s" % total)
    if show:
        plt.ioff()
        plt.show()                     # hold the windows open


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="TARGET",
                    help="regenerate only these (see --list)")
    ap.add_argument("--list", action="store_true", help="list the targets and exit")
    ap.add_argument("--show", action="store_true",
                    help="open the figures in windows instead of writing files")
    ap.add_argument("--check", action="store_true",
                    help="run the data, Monte Carlo and Eq. (53) cross-checks first")
    ap.add_argument("--numbers", action="store_true",
                    help="print the numbers quoted in the text of the last section")
    a = ap.parse_args(argv)

    if a.list:
        for name, mod, entry, cost in TARGETS:
            print("  %-10s %-22s %s()  [%s]" % (name, mod, entry, cost))
        return 0

    if a.check:
        from slogpet.data import verify
        from slogpet.validation import (run_checks, check_protocol,
                                        check_resolution_factor, check_nontof)
        from paper.config import D_PET, LENGTHS
        n_sys, n_ref = verify()
        print("data files consistent: %d systems, %d sources\n" % (n_sys, n_ref))
        run_checks(D_PET, LENGTHS)
        check_protocol()
        check_resolution_factor()
        check_nontof()
        print()

    known = [t[0] for t in TARGETS]
    names = a.only or known
    unknown = [n for n in names if n not in known]
    if unknown:
        ap.error("unknown target(s): %s; known: %s"
                 % (", ".join(unknown), ", ".join(known)))
    try:
        run(names, show=a.show)
    except RuntimeError as exc:          # no window backend, most likely
        print("error: %s" % exc, file=sys.stderr)
        return 1

    if a.numbers:
        from paper.fig_scenarios import print_numbers
        print_numbers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
