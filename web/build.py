#!/usr/bin/env python3
"""Generate web/index.html: one self-contained file, no build chain, no server.

The package sources and the data files are embedded verbatim, so there is a
single source of truth -- edit ``slogpet/`` and re-run this.  The page then
writes them into Pyodide's virtual filesystem and imports them, which means the
browser executes exactly the code the paper does, not a re-implementation.

    python3 web/build.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FILES = (["api.py"]
         + ["slogpet/" + f for f in sorted(os.listdir(os.path.join(ROOT, "slogpet")))
            if f.endswith(".py")]
         + ["slogpet/data/" + f
            for f in sorted(os.listdir(os.path.join(ROOT, "slogpet", "data")))
            if f.endswith(".json")])


def read(rel):
    base = HERE if rel == "api.py" else ROOT
    with open(os.path.join(base, rel), encoding="utf-8") as fh:
        return fh.read()


def main():
    payload = {rel: read(rel) for rel in FILES}
    blob = ("window.SLOGPET_FILES = "
            + json.dumps(payload, ensure_ascii=False, indent=0) + ";")
    html = read_local("template.html")
    app = read_local("app.js")
    for token, body in (("/*__FILES__*/", blob), ("/*__APP__*/", app)):
        assert html.count(token) == 1, token
        html = html.replace(token, body)
    out = os.path.join(HERE, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    kb = os.path.getsize(out) / 1024
    print("wrote %s (%.0f kB, %d embedded files)" % (out, kb, len(payload)))


def read_local(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    sys.exit(main())
