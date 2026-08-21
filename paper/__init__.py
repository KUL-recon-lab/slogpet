"""Everything specific to the paper: the configurations it discusses, the
figures it prints and the LaTeX it generates.

One module per artefact.  Each has a single entry point -- ``make(path)`` for a
figure, ``write(path)`` for a table -- takes its physics from ``slogpet`` and
adds none of its own, and can be run on its own:

    python3 -m paper.fig_beds

``make_all.py`` in the repository root runs them all.
"""
