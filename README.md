# Comparing short and long PET systems on a resolution-dependent task

    manuscript/         the document: main.tex, the TikZ sources, and the
                        generated figures and tables it inputs
    slogpet/            the model: pure, side-effect-free functions and the
                        Scanner / Task / Protocol parameter objects
    slogpet/data/       published system parameters, with their sources
    paper/              one module per figure and table, writing into manuscript/
    tests/              see tests/README.md
    web/                the browser frontend; see web/README.md
    make_all.py         regenerate every figure and table

## Layers

`slogpet` knows the physics and nothing else: no matplotlib, no LaTeX, no
printing, no file I/O, no mutable module state. Geometry is always an explicit
argument. See `slogpet/README.md`.

`tests/` says what ought to be true (see `tests/README.md`):

    python3 -m pytest              # ~2 s
    python3 -m pytest -m slow      # + Monte Carlo and Eq. (53)

`slogpet/data/systems.json` and `slogpet/data/detectors.json` hold the numbers:
ten published systems with the source of every value, and the four generic
detector designs of Figure 7. Adding a scanner is a data edit, not a code change.
`slogpet.data.verify()` checks referential integrity and runs as part of
`make_figures.py --check`.

`paper/` knows the paper: which systems are tabulated, which scenarios are
plotted, what the axes look like. One module per artefact, each with a single
entry point, each runnable on its own. It imports `slogpet` and adds no physics
of its own.

    python3 make_all.py                 # every figure and table (~9 s)
    python3 make_all.py --check         # also the cross-checks
    python3 make_all.py --numbers       # also the numbers quoted in section 6
    python3 make_all.py --only beds     # just one
    python3 make_all.py --list          # what the targets are

    python3 -m paper.fig_beds           # or a module directly

Everything lands in `manuscript/`, next to the `main.tex` that inputs it:

    python3 make_all.py && (cd manuscript && latexmk -pdf main.tex)

| module | artefact |
|---|---|
| `paper/table_systems.py` | `table.tex` (Table 2) |
| `paper/fig_slog.py` | `fig-slog2d.pdf` (Figure 1) |
| `paper/fig_profile.py` | `fig-profile.pgf` (Figure 4) |
| `paper/fig_ripple.py` | `fig-ripple.pgf` (Figure 5) |
| `paper/fig_beds.py` | `fig-beds.pgf` (Figure 6) |
| `paper/fig_scenarios.py` | `fig-scenarios.pgf` (Figure 7) |
| `paper/config.py` | what the paper plots, and its LaTeX spellings |
| `paper/style.py` | matplotlib settings, and where a figure goes |

### Looking at a figure instead of printing it

Every `make()` takes `path=None`, which draws the figure in a window and returns
it instead of writing a file -- the same sizes, colours and line widths, and the
same labels, typeset by LaTeX if there is one on the PATH and by matplotlib's
mathtext if not:

    pixi run -e explore ipython

    In [1]: from paper.fig_ripple import make
    In [2]: fig = make(None)                          # the printed figure
    In [3]: fig = make(None, max_peak_to_trough=1.25) # with a ripple limit
    In [4]: fig.axes[1].set_ylim(1.0, 1.3)

    python3 make_all.py --show          # or all of them at once, writing nothing

The `explore` environment exists because `matplotlib-base`, which is all the
paper build needs, has no window backend at all.

`make_figures.py` is now a shim that forwards to `make_all.py`; it can be deleted.

## The browser frontend

    python3 web/build.py     # regenerate the self-contained web/index.html

One HTML file, no server, no build chain: `build.py` embeds the package sources
and the data files, and the page runs them in Pyodide. Pick systems, designs or
your own geometry; read off SNR² against scan length, the axial profile, and the
derived quantities. See `web/README.md`.

## Environments

Dependencies are managed with [pixi](https://pixi.sh); `pixi.toml` explains why
there are four environments rather than one.

    pixi run test              # the fast suite (~2 s)
    pixi run test-all          # also Monte Carlo and Eq. (53)
    pixi run -e paper pdf      # regenerate every figure, then build the manuscript
    pixi run -e explore show   # draw the figures on screen instead
    pixi run -e web web        # regenerate web/index.html

TeX is deliberately *not* a pixi dependency: conda-forge's `texlive-core` ships
the binaries with an empty texmf tree, so it can typeset neither this document
nor matplotlib's text-measurement run. Install it from the system instead --
`pixi run -e paper check-tex` reports whether yours is complete.

Note that TeX is needed to *generate* the figures, not only to typeset the
document: matplotlib's pgf backend shells out to `pdflatex` to measure text.

Two GitHub Actions build on this: `paper.yml` runs the tests, rebuilds the
figures and uploads the PDF, and `pages.yml` publishes the browser frontend.
Both end by checking that the tracked generated files are not stale.

## What is tested, and what is not

`slogpet` is tested carefully -- 373 assertions over its properties, independent
recomputations of it, the claims the paper makes about it, and 288 pinned
values. `web/api.py` is covered too, since it is the contract the browser
depends on.

`paper/` is deliberately **not** tested. It is presentation: which systems are
tabulated, what the axes look like, how the legend reads. It is expected to keep
changing, and a test suite over it would cost more than it saves. The check on it
is looking at the PDF.

## Colour

The figures were checked with a colour-vision validator, and the original
palette did not pass: `#b3261e` and `#2f7d32` sat at OKLab ΔE 2.8 under
deuteranopia, which is to say a red-green colourblind reader — roughly one man
in twelve — could not tell those two curves apart; blue and violet were nearly
as close under protanopia. Line style distinguished length, but between detector
designs colour was the only cue. Both palettes were replaced.

**Figures 4–6 colour curves by axial length, which is an ordered quantity**, so
they now use one hue, light to dark: `#86b6ef` (30 cm), `#3987e5` (60 cm),
`#1c5cab` (100 cm), `#0d366b` (150 cm). The steps are evenly spaced in OKLab
lightness (gaps of 0.142) and all clear 2:1 contrast on white, so the figures
survive greyscale printing and no pair can be confused under any colour-vision
deficiency. The same length always gets the same colour in both figures, and
Figure 5 colours its four bed counts from the same ramp. The definition is in
`paper/style.py`.

**Figure 7 is the one genuinely categorical figure** — four detector designs with
no order between them — and uses `#2a78d6`, `#eda100`, `#d55181`, `#008300`,
in `slogpet/data/detectors.json`. These were chosen by enumerating four-subsets
of an eight-hue palette and keeping only those clearing the lightness band, the
chroma floor and the all-pairs colour-vision and normal-vision separations in
both light and dark: worst all-pairs ΔE is 13.0 under protanopia and 24.6 for
normal vision, against 2.8 and 13.0 before. The amber sits at 2.11:1 contrast,
below the 3:1 guideline; the relief for that is the legend, which spells out
every design, and the curves were widened from 1.0 to 1.2 pt.

The browser frontend uses the same categorical four, so the two agree.

## Status of the restructuring

1. ~~freeze the current numbers~~ -- the oracle did its job across steps 2-5
   and was retired once the refactoring was over
2. **done** -- move the pure functions into `slogpet/`; `D_PET`/`R` globals removed
3. **done** -- dataclasses `Scanner` / `Task` / `Protocol` / `SNRResult` and a
   single `snr2()` entry point; `SYSTEMS` and `SCENARIOS` are now `Scanner` objects
4. **done** -- `SYSTEMS` / `SCENARIOS` moved to `slogpet/data/*.json` with
   sources, notes and an integrity check
5. **done** -- split into `paper/` plus `make_all.py`
6. **done** -- `tests/`: properties, independent recomputation, the
   paper's own claims, and 288 pinned values
7. **done** -- `web/`: the same package running in the browser through Pyodide,
   verified against CPython to 2.5e-16
