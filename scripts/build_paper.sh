#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v bibtex >/dev/null 2>&1; then
  bibtex main
elif command -v bibtex8 >/dev/null 2>&1; then
  bibtex8 main
else
  echo "Install bibtex or bibtex8 to build the bibliography." >&2
  exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if grep -Eq 'undefined references|undefined citations|Overfull \\hbox' main.log; then
  echo "Unresolved reference or layout issue; inspect paper/main.log." >&2
  exit 1
fi
