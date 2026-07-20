#!/bin/bash
#SBATCH --job-name=stage2_growth
#SBATCH --partition=gpu
#SBATCH --exclude=gorina8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-143%3

set -Eeuo pipefail
cd "$HOME/bhome/ML/dual_conditioning"
[[ -f codes/configs/paths.env ]] && source codes/configs/paths.env

PROJECT="${PROJECT:-$PWD}"
SANDBOX="${SANDBOX:-/path/to/diffsbdd_container_or_sandbox}"
export TMPDIR="$PROJECT/artifacts/tmp"
export SINGULARITY_TMPDIR="$PROJECT/artifacts/tmp"
mkdir -p "$TMPDIR" logs

MANIFEST="artifacts/phase2_staged_growth/stage2_manifest.tsv"

if [[ -x /usr/bin/singularity ]]; then
    SINGULARITY_BIN=/usr/bin/singularity
else
    SINGULARITY_BIN=singularity
fi

# --- leer la fila del manifest correspondiente a esta tarea ---
eval "$(
python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, shlex, sys
manifest, task_id = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(manifest), delimiter="\t"))
match = [r for r in rows if int(r["task_id"]) == task_id]
if len(match) != 1:
    raise SystemExit("expected 1 row for task {}, got {}".format(task_id, len(match)))
r = match[0]
for k in ["pair_id","A_local","B_global","branch","add_n","n_stages",
          "n_samples","seed","replica","lambda_global","experiment_id",
          "out_dir","A_complex","B_ligand"]:
    print("{}={}".format(k.upper(), shlex.quote(r[k])))
PY
)"

echo "=== STAGE 2 STAGED GROWTH ==="
echo "TASK=$SLURM_ARRAY_TASK_ID EXP=$EXPERIMENT_ID"
echo "pair=$PAIR_ID branch=$BRANCH add_n=$ADD_N n_stages=$N_STAGES n_samples=$N_SAMPLES seed=$SEED replica=$REPLICA"
echo "out_dir=$OUT_DIR"

# guarda: no re-ejecutar si ya hay trajectory.tsv
if [[ -s "$OUT_DIR/trajectory.tsv" ]]; then
    echo "SKIP: trajectory.tsv ya existe en $OUT_DIR"
    exit 0
fi

"$SINGULARITY_BIN" exec \
  --bind "$PROJECT:/mnt/proyecto" \
  "$SANDBOX" \
  bash -lc "
    set -Eeuo pipefail
    cd /mnt/proyecto
    export PYTHONPATH=/mnt/proyecto/codes:/mnt/proyecto/codes/utils:\${PYTHONPATH:-}
    python codes/growth/orchestrator.py \
      --pair $PAIR_ID \
      --branch $BRANCH \
      --add_n $ADD_N \
      --n_stages $N_STAGES \
      --seed $SEED \
      --lambda_global $LAMBDA_GLOBAL \
      --pdbfile /mnt/proyecto/$A_COMPLEX \
      --b_ligand /mnt/proyecto/$B_LIGAND \
      --out_dir /mnt/proyecto/$OUT_DIR
  "

echo "=== TASK DONE ==="
cat "$OUT_DIR/trajectory.tsv" 2>/dev/null || echo "no trajectory.tsv"
