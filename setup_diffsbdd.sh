#!/usr/bin/env bash
# Clone DiffSBDD at the pinned commit and install the guidance and soft-mask integrations.
# DiffSBDD itself is NOT redistributed in this repo (see THIRD_PARTY.md).
#
# Usage:
#   ./setup_diffsbdd.sh [target_dir]
# Default target_dir: ./external/DiffSBDD
set -euo pipefail

REPO_URL="https://github.com/arneschneuing/DiffSBDD.git"
PINNED_COMMIT="5d0d38d16c8932a0339fd2ce3f67ade98bbdff27"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-${HERE}/external/DiffSBDD}"

shopt -s nullglob
PATCHES=("${HERE}"/patches/*.patch)
SOFTMASK_OVERLAY="${HERE}/patches/softmask_overlay"
if [ "${#PATCHES[@]}" -eq 0 ]; then
  echo "[setup] ERROR: no unified DiffSBDD patch found in patches/*.patch" >&2
  echo "[setup] Level-B setup is intentionally disabled until the final integration patch is added." >&2
  exit 2
fi

if [ -e "${DEST}" ]; then
  echo "[setup] ERROR: ${DEST} already exists. Remove it or pass another path."
  exit 1
fi

echo "[setup] Cloning DiffSBDD (clean) into ${DEST}"
mkdir -p "$(dirname "${DEST}")"
git clone "${REPO_URL}" "${DEST}"
cd "${DEST}"
echo "[setup] Checking out pinned commit ${PINNED_COMMIT}"
git checkout "${PINNED_COMMIT}"
HEAD="$(git rev-parse HEAD)"
cd - >/dev/null

if [ "${HEAD}" != "${PINNED_COMMIT}" ]; then
  echo "[setup] ERROR: HEAD ${HEAD} != pinned ${PINNED_COMMIT}"
  exit 1
fi

echo "[setup] Applying reviewed patches from patches/"
for p in "${PATCHES[@]}"; do
  echo "  git apply ${p}"
  ( cd "${DEST}" && git apply --check "${p}" && git apply "${p}" )
done

if [ ! -d "${SOFTMASK_OVERLAY}" ]; then
  echo "[setup] ERROR: missing soft-mask overlay: ${SOFTMASK_OVERLAY}" >&2
  exit 2
fi

echo "[setup] Installing centered soft-mask overlay"
install -m 0644 \
  "${SOFTMASK_OVERLAY}/equivariant_diffusion/conditional_model_softmask.py" \
  "${DEST}/equivariant_diffusion/conditional_model_softmask.py"
install -m 0644 \
  "${SOFTMASK_OVERLAY}/inpaint_softmask.py" \
  "${DEST}/inpaint_softmask.py"
install -m 0644 \
  "${SOFTMASK_OVERLAY}/softmask_atom_order.py" \
  "${DEST}/softmask_atom_order.py"

echo "[setup] Verifying patched file hashes"
(
  cd "${DEST}"
  printf '%s  %s\n' \
    'd8fcced3e17352357f18a1a4cd737f1b5f2caa48117d2d3ba66e1bbb5e20e441' \
    'equivariant_diffusion/conditional_model.py' \
    '659c328fde62ad59fe913938e66f40179857a17267c8d9158c8da6f1d3f305d6' \
    'inpaint.py' \
    'ae521cc7fae48c27582719b8881f83a094f815308236b0c38cd9e46bdb47c0b9' \
    'lightning_modules.py' \
    'e8cd642f4fb1314540781c7eca960a22afcf393d634c3b0744dfbfcce99f5059' \
    'equivariant_diffusion/conditional_model_softmask.py' \
    '61b5a4731395e9a724ad58a2f9b390d0c03d88129e501ea38cd175a6407cfc98' \
    'inpaint_softmask.py' \
    'f9a5aac8d6b25038c95a59be872a93e0d5b08453eb32fe22dc6d43c8fc34a69f' \
    'softmask_atom_order.py' | sha256sum --check --strict
  python -m py_compile \
    equivariant_diffusion/conditional_model.py \
    equivariant_diffusion/conditional_model_softmask.py \
    inpaint.py \
    inpaint_softmask.py \
    lightning_modules.py \
    softmask_atom_order.py
)

echo "[setup] DiffSBDD ready at ${DEST} (commit ${HEAD})"
echo "[setup] For guided and soft-mask runs, include ${HERE}/src, ${HERE}/src/growth, and ${DEST} in PYTHONPATH."
echo "[setup] Next: download the checkpoint (see THIRD_PARTY.md) and verify its SHA256."
