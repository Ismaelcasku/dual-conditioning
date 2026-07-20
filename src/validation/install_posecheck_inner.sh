#!/usr/bin/env bash
set -Eeuo pipefail

VENV="/work/.venvs/posecheck"
PYTHON="/opt/openmm-validation/bin/python"

"$PYTHON" -m venv "$VENV"

"$VENV/bin/python" -m pip install \
    --no-cache-dir \
    --upgrade \
    pip setuptools wheel

# PoseCheck 1.3.1 exige pandas==2.0.0.
# No fijamos manualmente prolif/datamol/pandas para no contradecir su metadata.
"$VENV/bin/python" -m pip install \
    --no-cache-dir \
    "numpy==1.26.4" \
    "posecheck==1.3.1"

echo
echo "=== Versiones instaladas ==="

"$VENV/bin/python" - <<'PY'
import importlib.metadata
import numpy
import pandas
import rdkit
import prolif
import datamol
import MDAnalysis

from posecheck import PoseCheck

for package in [
    "posecheck",
    "pandas",
    "numpy",
    "rdkit",
    "prolif",
    "datamol",
    "MDAnalysis",
    "hydride",
]:
    try:
        print(
            f"{package}: "
            f"{importlib.metadata.version(package)}"
        )
    except Exception as exc:
        print(f"{package}: NOT_FOUND ({exc})")

print("POSECHECK_IMPORT_STATUS=OK")
PY

"$VENV/bin/python" -m pip check

"$VENV/bin/python" -m pip freeze \
    | sort \
    > /work/.venvs/posecheck_requirements_freeze.txt

echo "POSECHECK_INSTALL_STATUS=OK"
