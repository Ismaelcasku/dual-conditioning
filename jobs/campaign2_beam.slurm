#!/usr/bin/bash
#SBATCH --job-name=campaign2_beam
#SBATCH --partition=gpu
#SBATCH --exclude=gorina8
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-39%2

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not defined}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/artifacts/tmp"

if [[ -f "$PROJECT_ROOT/config/paths.env" ]]; then
  source "$PROJECT_ROOT/config/paths.env"
else
  echo "ERROR: config/paths.env not found" >&2
  exit 1
fi

: "${SANDBOX:?SANDBOX is not defined in config/paths.env}"
if [[ "$SANDBOX" != /* ]]; then
  SANDBOX="$PROJECT_ROOT/$SANDBOX"
fi
if [[ ! -e "$SANDBOX" ]]; then
  echo "ERROR: Singularity sandbox/image not found: $SANDBOX" >&2
  exit 1
fi

export TMPDIR="$PROJECT_ROOT/artifacts/tmp"
export SINGULARITY_TMPDIR="$PROJECT_ROOT/artifacts/tmp"

MANIFEST="${MANIFEST:-artifacts/phase4_evo_beam/beam_manifest.tsv}"
if [[ "$MANIFEST" != /* ]]; then
  MANIFEST="$PROJECT_ROOT/$MANIFEST"
fi
if [[ ! -s "$MANIFEST" ]]; then
  echo "ERROR: manifest not found or empty: $MANIFEST" >&2
  exit 1
fi

if [[ -x /usr/bin/singularity ]]; then
  SINGULARITY_BIN=/usr/bin/singularity
elif command -v singularity >/dev/null 2>&1; then
  SINGULARITY_BIN="$(command -v singularity)"
else
  echo "ERROR: singularity not found" >&2
  exit 1
fi

eval "$(
/usr/bin/python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import shlex
import sys

manifest = sys.argv[1]
task_id = int(sys.argv[2])
with open(manifest, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

matches = [row for row in rows if int(row["task_id"]) == task_id]
if len(matches) != 1:
    raise SystemExit(
        f"expected one manifest row for task {task_id}; found {len(matches)}"
    )

row = matches[0]
required = [
    "pair_id",
    "variant",
    "k_beams",
    "seed",
    "lambda_global",
    "experiment_id",
    "out_dir",
    "A_complex",
    "B_ligand",
]
missing = [key for key in required if not row.get(key)]
if missing:
    raise SystemExit(f"empty manifest fields: {missing}")

for key in required:
    print(f"{key.upper()}={shlex.quote(row[key])}")
PY
)"

if [[ "$OUT_DIR" = /* ]]; then
  RUN_DIR="$OUT_DIR"
else
  RUN_DIR="$PROJECT_ROOT/$OUT_DIR"
fi

echo "=== PHASE 4 VARIABLE-INCREMENT SEARCH ==="
echo "JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "EXPERIMENT=$EXPERIMENT_ID"
echo "PAIR=$PAIR_ID VARIANT=$VARIANT K=$K_BEAMS SEED=$SEED"
echo "MANIFEST=$MANIFEST"
echo "OUT_DIR=$RUN_DIR"
echo "NODE=${SLURMD_NODENAME:-unknown}"
echo "SANDBOX=$SANDBOX"

if [[ -s "$RUN_DIR/run_status.json" ]]; then
  STATUS="$(
    /usr/bin/python3 - "$RUN_DIR/run_status.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("run_status", "unknown"))
PY
  )"
  case "$STATUS" in
    completed_graduated|completed_all_dead|stopped_max_stages)
      echo "SKIP: terminal run_status=$STATUS"
      exit 0
      ;;
    *)
      echo "RESUME: non-terminal run_status=$STATUS"
      ;;
  esac
fi

"$SINGULARITY_BIN" exec \
  --nv \
  --bind "$PROJECT_ROOT:/mnt/proyecto" \
  --pwd /mnt/proyecto \
  "$SANDBOX" \
  /usr/bin/bash -lc '
    set -Eeuo pipefail
    export DC_PROJECT_ROOT=/mnt/proyecto
    export PYTHONPATH=/mnt/proyecto/src:/mnt/proyecto/src/growth:/mnt/proyecto/external/DiffSBDD:${PYTHONPATH:-}
    exec /opt/conda/bin/python \
      src/growth/orchestrator_beam.py \
      "$@"
  ' bash \
    --pair "$PAIR_ID" \
    --k_beams "$K_BEAMS" \
    --seed "$SEED" \
    --lambda_global "$LAMBDA_GLOBAL" \
    --pdbfile "/mnt/proyecto/$A_COMPLEX" \
    --b_ligand "/mnt/proyecto/$B_LIGAND" \
    --out_dir "/mnt/proyecto/$OUT_DIR"

echo "=== TASK DONE ==="
if [[ -s "$RUN_DIR/run_status.json" ]]; then
  cat "$RUN_DIR/run_status.json"
else
  echo "WARNING: run_status.json was not created" >&2
fi
