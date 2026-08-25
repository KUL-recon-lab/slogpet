#!/usr/bin/env python3
"""Build the JupyterLite site: explore.py + slogpet/ -> lite/site/.

    pixi run -e lite serve      # build it and serve it at http://localhost:8009
    pixi run -e lite site       # build only

Three steps, no wheel and nothing installed at run time:

1.  ``explore.py`` becomes ``explore.ipynb``.  Jupytext does the conversion, so
    the ``# %%`` markers in the script are all the structure the notebook needs.
    The script is what the repository tracks -- readable diffs, runs at a
    terminal, and cannot drift from the notebook because the notebook is derived
    from it every build.

2.  ``plots.py`` and ``slogpet/`` are copied in beside it.  JupyterLite mounts everything under
    ``lite/files/`` as the reader's drive, the notebook's working directory is
    that drive, and IPython puts the working directory on ``sys.path`` -- so
    ``import slogpet`` finds the sources sitting next to the notebook.  numpy,
    scipy and bokeh come from Pyodide itself.

3.  ``jupyter lite build``.  Pyodide is fetched from a CDN at page load rather
    than vendored, which keeps the site around 20 MB.

The kernel is told which packages to load at startup, rather than left to work
it out: Pyodide otherwise decides by parsing each cell for imports, which sees
neither what ``slogpet`` imports internally nor anything at all in a cell that
contains a magic.  See ``preload_packages`` below.

``lite/files/`` and ``lite/site/`` are both generated; only this script and the
configuration next to it are tracked.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, "explore.py")
HELPERS = os.path.join(ROOT, "plots.py")     # imported by the notebook
PACKAGE = os.path.join(ROOT, "slogpet")

# The convention: everything under {lite-dir}/files/ is copied to {site}/files/
# and indexed, and is what the reader sees in the file browser.
FILES = os.path.join(HERE, "files")
SITE = os.path.join(HERE, "site")
NOTEBOOK = os.path.join(FILES, "explore.ipynb")

PORT = 8009

# Loaded when the kernel starts, so that nothing depends on Pyodide guessing.
# scipy is here because slogpet imports it internally, where no cell can see it.
# pandas is named although Pyodide's bokeh happens to depend on it: bokeh itself
# asks for narwhals, not pandas, so inheriting it would be luck rather than a
# decision -- and the table in explore.py needs it outright.
PRELOAD = ["numpy", "scipy", "bokeh", "pandas"]
KERNEL = "@jupyterlite/pyodide-kernel-extension:kernel"


def _tool(name: str) -> str:
    """A console script from this environment, on PATH or not."""
    return (shutil.which(name)
            or os.path.join(os.path.dirname(sys.executable), name))


def assemble() -> None:
    """Fill lite/files/ with what the reader's drive should contain."""
    if os.path.isdir(FILES):
        shutil.rmtree(FILES)
    os.makedirs(FILES)

    subprocess.run([_tool("jupytext"), "--to", "ipynb", SCRIPT, "-o", NOTEBOOK],
                   check=True)
    shutil.copy(HELPERS, FILES)
    shutil.copytree(PACKAGE, os.path.join(FILES, "slogpet"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("assembled %s" % os.path.relpath(FILES, ROOT))


def build() -> int:
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    cmd = [_tool("jupyter"), "lite", "build",
           "--lite-dir", HERE, "--output-dir", SITE,
           # source maps are for debugging JupyterLab itself, and are most of
           # the site's weight: 70 MB with them, 19 MB without
           "--no-sourcemaps", "--no-unused-shared-packages"]
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode:
        return result.returncode
    preload_packages()

    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _s, files in os.walk(SITE) for f in files)
    count = sum(len(files) for _d, _s, files in os.walk(SITE))
    print("wrote %s (%d files, %.0f MB)" % (SITE, count, total / 1024 / 1024))
    return 0


def preload_packages() -> None:
    """Name the startup packages in the built site's configuration.

    Done to the finished file rather than through a ``jupyter-lite.json`` in the
    lite directory, because the Pyodide kernel addon rewrites this very key
    during the build: depending on the order the build steps happen to run in,
    a setting placed there is sometimes kept and sometimes dropped.  Merging
    afterwards always holds.
    """
    path = os.path.join(SITE, "jupyter-lite.json")
    with open(path) as fh:
        config = json.load(fh)
    kernel = (config.setdefault("jupyter-config-data", {})
                    .setdefault("litePluginSettings", {})
                    .setdefault(KERNEL, {}))
    kernel.setdefault("loadPyodideOptions", {})["packages"] = list(PRELOAD)
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    print("kernel preloads: %s" % ", ".join(PRELOAD))


def serve() -> int:
    """Serve the built site, as a static host would."""
    import http.server
    import functools

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=SITE)
    print("\n  http://localhost:%d/notebooks/index.html?path=explore.ipynb\n"
          "\nCtrl-C to stop." % PORT)
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    assemble()
    code = build()
    if code or "--serve" not in argv:
        return code
    return serve()


if __name__ == "__main__":
    sys.exit(main())
