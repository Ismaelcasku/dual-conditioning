#!/usr/bin/env bash
#SBATCH --job-name=postmin_posecheck
#SBATCH --partition=gpu
#SBATCH --nodelist=gorina9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%3
#SBATCH --output=logs/validation/%x_%A_%a.out
#SBATCH --error=logs/validation/%x_%A_%a.err
#SBATCH --export=NONE

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[[ -f config/paths.env ]] && source config/paths.env
PROJECT="${PROJECT:-$REPO_ROOT}"
: "${SIF:?Set SIF in config/paths.env}"
mkdir -p logs/validation

MANIFEST="${MANIFEST:-$PROJECT/manifests/validation/postmin_posecheck_shape.tsv}"
OUT_REL="${OUT_REL:-artifacts/validation/restrained_minimization/exp06_openmm_gaff/post_minimization_validation_v2}"
SINGULARITY_BIN="${SINGULARITY_BIN:-/usr/bin/singularity}"
LINE_NUMBER="$((SLURM_ARRAY_TASK_ID + 2))"

IFS=$'\t' read -r ID GROUP RESULT LIGAND FIXED < <(
    awk -F '\t' -v line="$LINE_NUMBER" \
        'NR == line {print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5}' \
        "$MANIFEST"
)

test -n "${ID:-}" || { echo "ERROR: missing manifest row." >&2; exit 1; }
TASKDIR="$PROJECT/$OUT_REL/$ID"
mkdir -p "$TASKDIR"

"$SINGULARITY_BIN" exec \
  --cleanenv \
  --bind "$PROJECT:/work" \
  "$SIF" \
  /usr/bin/env \
    VIRTUAL_ENV=/work/.venvs/posecheck \
    PATH=/work/.venvs/posecheck/bin:/opt/openmm-validation/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    /work/.venvs/posecheck/bin/python \
    /work/src/validation/postmin_posecheck_shape.py \
      --project /work \
      --candidate-id "$ID" \
      --group "$GROUP" \
      --result-json "/work/$RESULT" \
      --ligand-sdf "/work/$LIGAND" \
      --fixed-json "/work/$FIXED" \
      --output "/work/$OUT_REL/$ID" \
      --reduce /work/.venvs/posecheck/bin/reduce

test -s "$TASKDIR/summary.tsv"
test -s "$TASKDIR/validation_results.json"
echo "TASK_STATUS=OK"
