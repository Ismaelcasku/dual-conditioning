#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[[ -f config/paths.env ]] && source config/paths.env
PROJECT="${PROJECT:-$REPO_ROOT}"
: "${SIF:?Set SIF in config/paths.env}"
SINGULARITY_BIN="${SINGULARITY_BIN:-/usr/bin/singularity}"

"$SINGULARITY_BIN" exec \
  --cleanenv \
  --bind "$PROJECT:/work" \
  "$SIF" \
  /bin/bash \
  /work/src/validation/install_posecheck_inner.sh
