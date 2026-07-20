#!/usr/bin/env bash
#SBATCH --job-name=postmin_aggregate
#SBATCH --partition=gpu
#SBATCH --nodelist=gorina9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/validation/%x_%j.out
#SBATCH --error=logs/validation/%x_%j.err
#SBATCH --export=NONE

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[[ -f config/paths.env ]] && source config/paths.env
PROJECT="${PROJECT:-$REPO_ROOT}"
: "${SIF:?Set SIF in config/paths.env}"
mkdir -p logs/validation

OUT_REL="${OUT_REL:-artifacts/validation/restrained_minimization/exp06_openmm_gaff/post_minimization_validation_v2}"
EXPECTED="${EXPECTED:-10}"
SINGULARITY_BIN="${SINGULARITY_BIN:-/usr/bin/singularity}"

"$SINGULARITY_BIN" exec \
  --cleanenv \
  --bind "$PROJECT:/work" \
  "$SIF" \
  /usr/bin/env \
    VIRTUAL_ENV=/work/.venvs/posecheck \
    PATH=/work/.venvs/posecheck/bin:/opt/openmm-validation/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    /work/.venvs/posecheck/bin/python \
    /work/src/validation/aggregate_postmin_posecheck_shape.py \
      --root "/work/$OUT_REL" \
      --expected "$EXPECTED"

test -s "$PROJECT/$OUT_REL/compact_summary.tsv"
test -s "$PROJECT/$OUT_REL/complete_summary.tsv"
echo "AGGREGATION_TASK_STATUS=OK"
