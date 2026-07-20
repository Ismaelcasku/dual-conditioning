#!/usr/bin/env bash
# Create the exp0/exp1/exp2 symlinks that figure1 expects at the repo root,
# pointing into data/derived/. Run once before reproduce/verify_figures.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
ln -sfn data/derived/exp0 exp0
ln -sfn data/derived/exp1 exp1
ln -sfn data/derived/exp2 exp2
echo "Linked exp0/exp1/exp2 -> data/derived/"
