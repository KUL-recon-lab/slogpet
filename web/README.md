# The browser frontend

A single self-contained HTML file that runs the `slogpet` package in the reader's
browser through [Pyodide](https://pyodide.org). No server, no build chain, no
account, and no data leaves the machine.

    python3 web/build.py       # regenerate web/index.html
    open web/index.html        # needs a connection the first time, for Pyodide

## Why this way

The whole point of the restructuring was that the paper, the tests and the
frontend sit on one code path. A JavaScript re-implementation would have broken
that on the first edit. Instead `build.py` embeds the package sources and the
data files verbatim into the page; at load time they are written into Pyodide's
virtual filesystem and imported, so the browser executes the same functions
`make_all.py` does.

That is checked, not assumed: driving the page with a headless Chromium and
comparing 742 numbers across five configurations against CPython gives a worst
relative difference of **2.5e-16** — the JSON round trip, and nothing else. A
custom system defined through the form agrees exactly.

The cost is the first load: about 30 MB of Python, NumPy and SciPy, cached by
the browser afterwards. Later interaction is fast, because the bed search runs on
a lattice (see `slogpet/README.md`) and because `axial_profile` and
`optimal_protocol` are memoised — changing the SLoG size or the object diameter
reuses everything that does not depend on them.

## Publishing it

Any static host works, since there is nothing to run server-side.

    # GitHub Pages
    git subtree push --prefix web origin gh-pages
    # or copy web/index.html anywhere that serves files

`index.html` fetches Pyodide from jsDelivr. To serve it yourself instead — an
air-gapped network, or insurance against the CDN — download the Pyodide release
next to the page and point at it:

    <script src="./pyodide/pyodide.js"></script>
    loadPyodide({ indexURL: "./pyodide/" })

Only `pyodide.js`, `pyodide.asm.js`, `pyodide.asm.wasm`, `python_stdlib.zip`,
`pyodide-lock.json` and the NumPy, SciPy and libopenblas wheels are needed —
about 30 MB, not the full 350 MB distribution.

## What is on the page

Pick any number of published systems, generic detector designs, or systems of
your own, and read off SNR² against scan length, the axial profile at one scan
length showing the bed structure, and a table of the derived quantities. The
sources for every published number are at the foot of the page.

Colour identifies the crystal and dash the configuration, exactly as in
Figure 7 — so a family keeps its colour when the selection changes, and at most
four crystals and four configurations each can be shown at once. Colours are
capped rather than cycled: a fifth crystal is not given a made-up hue.

## Files

| file | what it is |
|---|---|
| `api.py` | the only place the browser and the package meet; JSON in, JSON out |
| `template.html` | markup and styling, with two placeholders |
| `app.js` | the interface: charts, controls, table. No physics. |
| `build.py` | embeds the package and fills the placeholders |
| `index.html` | generated — do not edit by hand |

`api.py` is covered by `tests/test_webapi.py`, so the page cannot drift from the
package without a test failing.

## Colour

The four categorical colours are `#2a78d6`, `#eda100`, `#d55181`, `#008300` on
light and `#3987e5`, `#c98500`, `#d55181`, `#008300` on dark — the same four the
paper's Figure 7 now uses, so the page and the paper agree. They were chosen by
running every four-subset of an eight-hue palette through a colour-vision
validator and keeping only combinations that clear the lightness band, the
chroma floor, and the all-pairs CVD and normal-vision separations in both modes.
Dash patterns carry the configuration, and the table repeats every value in text.

The paper's original Figure 7 colours failed the same check — red and green at
ΔE 2.8 under deuteranopia — and have been replaced; see the repository README.
