#!/usr/bin/env python3
"""Assemble an .ipynb from ordered raw cell files in a directory.

Files are sorted lexicographically. Extension decides cell type:
  *.py  -> code cell
  *.md  -> markdown cell
This avoids all JSON-escaping pain: each cell is written as a raw file.
"""
import json, sys, glob, os

cells_dir = sys.argv[1]
out_path = sys.argv[2]

files = sorted(glob.glob(os.path.join(cells_dir, "*")))
cells = []
for f in files:
    base = os.path.basename(f)
    if not os.path.isfile(f):
        continue
    if not (base.endswith(".py") or base.endswith(".md")):
        continue
    with open(f, "r") as fh:
        src = fh.read()
    if base.endswith(".md"):
        ctype = "markdown"
    elif base.endswith(".py"):
        ctype = "code"
    else:
        continue
    cell = {
        "cell_type": ctype,
        "id": os.path.splitext(base)[0].replace("_", "-"),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }
    if ctype == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(out_path, "w") as fh:
    json.dump(nb, fh, indent=1)

# Validate it round-trips
with open(out_path) as fh:
    json.load(fh)
print("wrote", out_path, "with", len(cells), "cells (validated JSON)")
