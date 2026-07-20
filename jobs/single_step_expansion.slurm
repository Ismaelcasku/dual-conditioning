#!/bin/bash
#SBATCH --job-name=stage1_growth
#SBATCH --partition=gpu
#SBATCH --exclude=gorina8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-119%3

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -f config/paths.env ]] && source config/paths.env

PROJECT="${PROJECT:-$REPO_ROOT}"
: "${SANDBOX:?Set SANDBOX in config/paths.env}"

MANIFEST="artifacts/phase1_fragment_growing/stage1_manifest.tsv"

CKPT="artifacts/checkpoints/crossdocked_fullatom_cond.ckpt"

export TMPDIR="$PROJECT/artifacts/tmp"
export SINGULARITY_TMPDIR="$PROJECT/artifacts/tmp"

mkdir -p "$TMPDIR" logs

if [[ -x /usr/bin/singularity ]]; then
    SINGULARITY_BIN=/usr/bin/singularity
else
    SINGULARITY_BIN=singularity
fi

###############################################################################
# Leer exactamente la fila correspondiente del manifest.
# Se anaden dos columnas nuevas respecto a la campana: RESAMPLINGS y ADD_N_NODES.
###############################################################################

eval "$(
python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import shlex
import sys

manifest = sys.argv[1]
task_id = int(sys.argv[2])

with open(manifest) as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

matches = [row for row in rows if int(row["task_id"]) == task_id]

if len(matches) != 1:
    raise SystemExit(
        "Expected one manifest row for task {}, found {}".format(task_id, len(matches))
    )

row = matches[0]

for key in [
    "pair_id",
    "A_local",
    "B_global",
    "seed",
    "lambda_global",
    "resamplings",
    "add_n_nodes",
    "n_samples_requested",
    "experiment_id",
    "out_sdf",
    "A_complex",
    "B_ligand",
    "fixed_json",
    "fixed_atoms",
]:
    print("{}={}".format(key.upper(), shlex.quote(row[key])))
PY
)"

###############################################################################
# Mascara fija segun A (identica a la campana)
###############################################################################

case "$PAIR_ID" in
    x0874_x1093)
        FIXED_ATOMS_ARRAY=(
            C02 C04 C05 C06 C07 C08 C09
        )
        ;;
    x0434_x1093|x0434_x2193)
        FIXED_ATOMS_ARRAY=(
            C4 C5 C6 N C1 C2 C3
        )
        ;;
    *)
        echo "ERROR: unknown pair: $PAIR_ID" >&2
        exit 2
        ;;
esac

OUTDIR=$(dirname "$OUT_SDF")
METADATA="${OUTDIR}/run_metadata.txt"

mkdir -p "$OUTDIR"

if [[ -s "$OUT_SDF" ]]; then
    echo "ERROR: output already exists: $OUT_SDF" >&2
    exit 3
fi

for required_file in \
    "$CKPT" \
    "$A_COMPLEX" \
    "$B_LIGAND" \
    "$FIXED_JSON" \
    "src/generation/run_inpaint_seeded.py" \
    "external/DiffSBDD/inpaint.py"
do
    if [[ ! -s "$required_file" ]]; then
        echo "ERROR: missing required file: $required_file" >&2
        exit 4
    fi
done

###############################################################################
# add_n_nodes: celda vacia -> el modelo decide (no se pasa el flag).
# valor entero  -> se pasa --add_n_nodes.
# Se construye el fragmento de argumento aqui, en el host.
###############################################################################

if [[ -n "${ADD_N_NODES}" ]]; then
    ADD_N_NODES_ARG="--add_n_nodes ${ADD_N_NODES}"
    ADD_N_NODES_META="${ADD_N_NODES}"
else
    ADD_N_NODES_ARG=""
    ADD_N_NODES_META="auto_size_distribution"
fi

cat > "$METADATA" <<EOF
experiment=stage1_fragment_growing
pair_id=${PAIR_ID}
A_local=${A_LOCAL}
B_global=${B_GLOBAL}
seed=${SEED}
lambda_global=${LAMBDA_GLOBAL}
lambda_local=1.0
alpha=0.3
resamplings=${RESAMPLINGS}
add_n_nodes=${ADD_N_NODES_META}
n_samples_requested=${N_SAMPLES_REQUESTED}
fixed_atoms=${FIXED_ATOMS}
fixed_json=${FIXED_JSON}
guide_space=x0
guide_field=shape
guide_r=1.7
guide_clip=1.0
timesteps=50
output_sdf=${OUT_SDF}
EOF

echo "============================================================"
echo "STAGE 1 FRAGMENT GROWING"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "PAIR=${PAIR_ID}"
echo "A_LOCAL=${A_LOCAL}"
echo "B_GLOBAL=${B_GLOBAL}"
echo "SEED=${SEED}"
echo "LAMBDA=${LAMBDA_GLOBAL}"
echo "RESAMPLINGS=${RESAMPLINGS}"
echo "ADD_N_NODES=${ADD_N_NODES_META}"
echo "N_SAMPLES=${N_SAMPLES_REQUESTED}"
echo "FIXED_ATOMS=${FIXED_ATOMS_ARRAY[*]}"
echo "OUT_SDF=${OUT_SDF}"
echo "============================================================"

"$SINGULARITY_BIN" exec \
  --nv \
  --bind "$PROJECT:/mnt/proyecto" \
  "$SANDBOX" \
  bash -lc "
    set -Eeuo pipefail

    cd /mnt/proyecto/external/DiffSBDD

    export PYTHONPATH=/mnt/proyecto/src:/mnt/proyecto/external/DiffSBDD:\${PYTHONPATH:-}
    export PYTHONHASHSEED=${SEED}

    export DC_GUIDE_SPACE=x0
    export DC_GUIDE_FIELD=shape
    export DC_GUIDE_ALPHA=0.3
    export DC_GUIDE_DEBUG=0

    python /mnt/proyecto/src/generation/run_inpaint_seeded.py \
      --seed ${SEED} \
      /mnt/proyecto/external/DiffSBDD/inpaint.py \
      /mnt/proyecto/${CKPT} \
      --pdbfile /mnt/proyecto/${A_COMPLEX} \
      --ref_ligand A:404 \
      --fix_atoms ${FIXED_ATOMS_ARRAY[*]} \
      --outfile /mnt/proyecto/${OUT_SDF} \
      --n_samples ${N_SAMPLES_REQUESTED} \
      --resamplings ${RESAMPLINGS} \
      --timesteps 50 \
      --center ligand \
      --lambda_global ${LAMBDA_GLOBAL} \
      --guide_r 1.7 \
      --guide_clip 1.0 \
      --b_shape_ligand /mnt/proyecto/${B_LIGAND} \
      ${ADD_N_NODES_ARG} \
      --sanitize
  "

if [[ ! -s "$OUT_SDF" ]]; then
    echo "ERROR: missing or empty output SDF" >&2
    exit 5
fi

N_RECORDS=$(grep -c '\$\$\$\$' "$OUT_SDF" || true)

{
    echo "n_sdf_records_written=${N_RECORDS}"
    echo "completed_at=$(date --iso-8601=seconds)"
} >> "$METADATA"

sha256sum "$OUT_SDF" > "${OUT_SDF}.sha256"

echo "GENERATION_DONE"
echo "pair=${PAIR_ID}"
echo "seed=${SEED}"
echo "lambda=${LAMBDA_GLOBAL}"
echo "resamplings=${RESAMPLINGS}"
echo "add_n_nodes=${ADD_N_NODES_META}"
echo "requested=${N_SAMPLES_REQUESTED}"
echo "written=${N_RECORDS}"
echo "outfile=${OUT_SDF}"
