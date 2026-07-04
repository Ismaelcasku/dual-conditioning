#!/bin/bash
#SBATCH --job-name=exp04_seed_rep
#SBATCH --partition=gpu
#SBATCH --exclude=gorina8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-14%3

set -Eeuo pipefail

cd "$HOME/bhome/ML/dual_conditioning"

[[ -f codes/configs/paths.env ]] && source codes/configs/paths.env

PROJECT="${PROJECT:-$PWD}"
SANDBOX="${SANDBOX:-/mnt/beegfs/home/u2930408/ML/drug_design_engine/artifacts/containers/dde_main_sandbox}"

export TMPDIR="$PROJECT/artifacts/tmp"
export SINGULARITY_TMPDIR="$PROJECT/artifacts/tmp"

mkdir -p "$TMPDIR" logs

if [[ -x /usr/bin/singularity ]]; then
    SINGULARITY_BIN=/usr/bin/singularity
else
    SINGULARITY_BIN=singularity
fi

###############################################################################
# Diseño experimental
#
# Cada semilla se ejecuta con los tres lambda:
#
# task 0,1,2   -> seed 1101, lambda 0,50,100
# task 3,4,5   -> seed 2202, lambda 0,50,100
# ...
###############################################################################

LAMBDAS=(0.0 50.0 100.0)
SEEDS=(1101 2202 3303 4404 5505)

N_LAMBDAS="${#LAMBDAS[@]}"

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / N_LAMBDAS))
LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID % N_LAMBDAS))

SEED="${SEEDS[$SEED_INDEX]}"
LAMBDA="${LAMBDAS[$LAMBDA_INDEX]}"

ALPHA=0.3
N_SAMPLES=10

CKPT="artifacts/checkpoints/crossdocked_fullatom_cond.ckpt"

A_COMPLEX="data/mpro/prepared/silvr_xchem_hits/x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_complex.pdb"

B_LIGAND="data/mpro/prepared/silvr_xchem_hits/x1093__5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20/5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20_ligand.sdf"

RUNROOT="artifacts/phase0_exp04_seed_replicates/x0874_x1093"

OUTDIR="${RUNROOT}/seed_${SEED}/lambda_${LAMBDA}"
OUT_SDF="${OUTDIR}/exp04_x0874_x1093_seed_${SEED}_lambda_${LAMBDA}_n${N_SAMPLES}.sdf"
METADATA="${OUTDIR}/run_metadata.txt"

mkdir -p "$OUTDIR"

# Evitar sobrescribir silenciosamente una ejecución previa.
if [[ -s "$OUT_SDF" ]]; then
    echo "ERROR: output already exists: $OUT_SDF" >&2
    exit 3
fi

cat > "$METADATA" <<EOF
experiment=exp04_seed_replicates
pair=x0874_x1093
slurm_job_id=${SLURM_JOB_ID}
slurm_array_task_id=${SLURM_ARRAY_TASK_ID}
host=$(hostname)
seed=${SEED}
lambda_global=${LAMBDA}
alpha=${ALPHA}
n_samples_requested=${N_SAMPLES}
guide_space=x0
guide_field=shape
resamplings=5
timesteps=50
guide_r=1.7
guide_clip=1.0
output_sdf=${OUT_SDF}
EOF

echo "============================================================"
echo "EXP04 SEEDED REPLICATE"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "SEED_INDEX=${SEED_INDEX}"
echo "LAMBDA_INDEX=${LAMBDA_INDEX}"
echo "SEED=${SEED}"
echo "LAMBDA=${LAMBDA}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "OUT_SDF=${OUT_SDF}"
echo "============================================================"

"$SINGULARITY_BIN" exec \
  --nv \
  --bind "$PROJECT:/mnt/proyecto" \
  "$SANDBOX" \
  bash -lc "
    set -Eeuo pipefail

    cd /mnt/proyecto/codes/vendor/diffsbdd

    export PYTHONPATH=/mnt/proyecto/codes:/mnt/proyecto/codes/vendor/diffsbdd:\${PYTHONPATH:-}
    export PYTHONHASHSEED=${SEED}

    export DC_GUIDE_SPACE=x0
    export DC_GUIDE_FIELD=shape
    export DC_GUIDE_ALPHA=${ALPHA}
    export DC_GUIDE_DEBUG=0

    python /mnt/proyecto/codes/utils/run_inpaint_seeded.py \
      --seed ${SEED} \
      /mnt/proyecto/codes/vendor/diffsbdd/inpaint.py \
      /mnt/proyecto/${CKPT} \
      --pdbfile /mnt/proyecto/${A_COMPLEX} \
      --ref_ligand A:404 \
      --fix_atoms C02 C04 C05 C06 C07 C08 C09 \
      --outfile /mnt/proyecto/${OUT_SDF} \
      --n_samples ${N_SAMPLES} \
      --resamplings 5 \
      --timesteps 50 \
      --center ligand \
      --lambda_global ${LAMBDA} \
      --guide_r 1.7 \
      --guide_clip 1.0 \
      --b_shape_ligand /mnt/proyecto/${B_LIGAND} \
      --sanitize
  "

if [[ ! -s "$OUT_SDF" ]]; then
    echo "ERROR: generation finished but SDF is missing or empty: $OUT_SDF" >&2
    exit 4
fi

N_RECORDS=$(grep -c '\$\$\$\$' "$OUT_SDF" || true)

{
    echo "n_sdf_records_written=${N_RECORDS}"
    echo "completed_at=$(date --iso-8601=seconds)"
} >> "$METADATA"

sha256sum "$OUT_SDF" > "${OUT_SDF}.sha256"

echo "GENERATION_DONE"
echo "seed=${SEED}"
echo "lambda=${LAMBDA}"
echo "requested=${N_SAMPLES}"
echo "written=${N_RECORDS}"
echo "outfile=${OUT_SDF}"
