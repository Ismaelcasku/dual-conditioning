#!/bin/bash
#SBATCH --job-name=exp05_x0434_l50
#SBATCH --partition=gpu
#SBATCH --exclude=gorina8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3%2

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
# Diseño:
#
# task 0 -> x0434 → x1093, lambda 0
# task 1 -> x0434 → x1093, lambda 50
# task 2 -> x0434 → x2193, lambda 0
# task 3 -> x0434 → x2193, lambda 50
###############################################################################

TARGETS=(x1093 x2193)
LAMBDAS=(0.0 50.0)

N_LAMBDAS="${#LAMBDAS[@]}"

TARGET_INDEX=$((SLURM_ARRAY_TASK_ID / N_LAMBDAS))
LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID % N_LAMBDAS))

TARGET="${TARGETS[$TARGET_INDEX]}"
LAMBDA="${LAMBDAS[$LAMBDA_INDEX]}"

SEED=6606
N_SAMPLES=5
ALPHA=0.3

CKPT="artifacts/checkpoints/crossdocked_fullatom_cond.ckpt"

A_COMPLEX="data/mpro/prepared/silvr_xchem_hits/x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_complex.pdb"

case "$TARGET" in
    x1093)
        B_LIGAND="data/mpro/prepared/silvr_xchem_hits/x1093__5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20/5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20_ligand.sdf"
        ;;
    x2193)
        B_LIGAND="data/mpro/prepared/silvr_xchem_hits/x2193__5RHD_Mpro-x2193_AAR-POS-5507155c-1/5RHD_Mpro-x2193_AAR-POS-5507155c-1_ligand.sdf"
        ;;
    *)
        echo "ERROR: unknown target $TARGET" >&2
        exit 2
        ;;
esac

FIXED_ATOMS=(C4 C5 C6 N C1 C2 C3)

OUTDIR="artifacts/phase0_exp05_x0434_lambda50_pilot/${TARGET}/seed_${SEED}/lambda_${LAMBDA}"

OUT_SDF="${OUTDIR}/exp05_x0434_to_${TARGET}_seed_${SEED}_lambda_${LAMBDA}_n${N_SAMPLES}.sdf"

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
    "codes/utils/run_inpaint_seeded.py" \
    "artifacts/fixed_atom_screen/x0434_blind/x0434_fixed_anchor_BLIND.json"
do
    if [[ ! -s "$required_file" ]]; then
        echo "ERROR: missing required file: $required_file" >&2
        exit 4
    fi
done

cat > "$METADATA" <<EOF
experiment=exp05_x0434_lambda50_pilot
A_local=x0434
B_global=${TARGET}
seed=${SEED}
lambda_global=${LAMBDA}
alpha=${ALPHA}
n_samples_requested=${N_SAMPLES}
fixed_atoms=C4 C5 C6 N C1 C2 C3
anchor_selection=blind_largest_ring_system_plus_one_heavy_atom_shell
guide_space=x0
guide_field=shape
guide_r=1.7
guide_clip=1.0
resamplings=5
timesteps=50
output_sdf=${OUT_SDF}
EOF

echo "============================================================"
echo "EXP05 X0434 LAMBDA-50 PILOT"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "A_LOCAL=x0434"
echo "B_GLOBAL=${TARGET}"
echo "SEED=${SEED}"
echo "LAMBDA=${LAMBDA}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "FIXED_ATOMS=${FIXED_ATOMS[*]}"
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
    export DC_GUIDE_DEBUG=1

    python /mnt/proyecto/codes/utils/run_inpaint_seeded.py \
      --seed ${SEED} \
      /mnt/proyecto/codes/vendor/diffsbdd/inpaint.py \
      /mnt/proyecto/${CKPT} \
      --pdbfile /mnt/proyecto/${A_COMPLEX} \
      --ref_ligand A:404 \
      --fix_atoms C4 C5 C6 N C1 C2 C3 \
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
    echo "ERROR: generation finished but output SDF is missing or empty" >&2
    exit 5
fi

N_RECORDS=$(grep -c '\$\$\$\$' "$OUT_SDF" || true)

{
    echo "n_sdf_records_written=${N_RECORDS}"
    echo "completed_at=$(date --iso-8601=seconds)"
} >> "$METADATA"

sha256sum "$OUT_SDF" > "${OUT_SDF}.sha256"

echo "GENERATION_DONE"
echo "target=${TARGET}"
echo "lambda=${LAMBDA}"
echo "seed=${SEED}"
echo "requested=${N_SAMPLES}"
echo "written=${N_RECORDS}"
echo "outfile=${OUT_SDF}"
