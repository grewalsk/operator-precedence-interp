#!/usr/bin/env bash
# Build the paper: figures (from committed CSV/JSON) then the PDF.
set -euo pipefail
cd "$(dirname "$0")"

echo "[build] regenerating figures from committed data..."
for f in figs/make_*.py; do
  echo "  - $f"; python3 "$f"
done

echo "[build] compiling PDF (pdflatex x2 + bibtex)..."
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/tmp/paper_build.log 2>&1 || {
  echo "[build] FAILED. tail of log:"; tail -40 /tmp/paper_build.log; exit 1; }

pages=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/{print $2}' || echo "?")
echo "[build] OK -> main.pdf ($pages pages total)"
cp -f main.pdf paper.pdf
