#!/usr/bin/env bash
#SBATCH --job-name=exp06_openmm
#SBATCH --partition=gpu
#SBATCH --nodelist=gorina9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/validation/%x_%A_%a.out
#SBATCH --error=logs/validation/%x_%A_%a.err
#SBATCH --export=NONE

set -Eeuo pipefail

export HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[[ -f config/paths.env ]] && source config/paths.env
PROJECT="${PROJECT:-$REPO_ROOT}"
: "${SIF:?Set SIF in config/paths.env}"
mkdir -p logs/validation

MANIFEST="${MANIFEST:-$PROJECT/manifests/validation/minimization_candidates.tsv}"
POSTROOT="${POSTROOT:-$PROJECT/artifacts/validation/postprocessing/exp06_diffsbdd_native_anchor_aware/sdf}"
OUTROOT="${OUTROOT:-$PROJECT/artifacts/validation/restrained_minimization/exp06_openmm_gaff/array}"
SINGULARITY="${SINGULARITY_BIN:-/usr/bin/singularity}"

LINE_NUMBER="$((SLURM_ARRAY_TASK_ID + 2))"

IFS=$'\t' read -r ID GROUP < <(
    awk -F '\t'         -v line="$LINE_NUMBER"         'NR == line {print $1 "\t" $2}'         "$MANIFEST"
)

if [ -z "${ID:-}" ]; then
    echo "ERROR: no candidate for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

if [[ "$ID" =~ ^(x[0-9]+)_(x[0-9]+)_seed([0-9]+)_lambda([0-9]+)_sample([0-9]+)$ ]]
then
    A_LOCAL="${BASH_REMATCH[1]}"
    B_GLOBAL="${BASH_REMATCH[2]}"
    SEED="${BASH_REMATCH[3]}"
    LAMBDA="${BASH_REMATCH[4]}"
    SAMPLE="${BASH_REMATCH[5]}"
else
    echo "ERROR: invalid candidate identifier: $ID"
    exit 1
fi

PAIR="${A_LOCAL}_${B_GLOBAL}"

TASKDIR="$OUTROOT/$ID"
INPUTDIR="$TASKDIR/input"
MINOUT="$TASKDIR/minimization"

CANDIDATE="$INPUTDIR/${ID}.sdf"
FIXED_JSON="$INPUTDIR/${ID}_fixed_atoms.json"

mkdir -p     "$INPUTDIR"     "$MINOUT"

on_error() {
    rc=$?

    {
        printf 'candidate_id\tgroup\tstatus\texit_code\n'
        printf '%s\t%s\tFAILED\t%s\n'             "$ID" "$GROUP" "$rc"
    } > "$TASKDIR/status.tsv"

    echo "TASK_STATUS=FAILED"
    echo "exit_code=$rc"

    exit "$rc"
}

trap on_error ERR

echo "=================================================="
echo "candidate_id=$ID"
echo "group=$GROUP"
echo "A_local=$A_LOCAL"
echo "B_global=$B_GLOBAL"
echo "seed=$SEED"
echo "lambda=$LAMBDA"
echo "sample=$SAMPLE"
echo "host=$(hostname)"
echo "=================================================="

# -----------------------------------------------------------
# 1. Localizar el postprocessed.sdf.
# -----------------------------------------------------------

mapfile -t SOURCES < <(
    find "$POSTROOT"         -type f         -path "*/${PAIR}/seed_${SEED}/lambda_${LAMBDA}/postprocessed.sdf"         | sort
)

if [ "${#SOURCES[@]}" -eq 0 ]; then
    echo "ERROR: no source SDF found for $ID"
    exit 2
fi

ORDERED_SOURCES=()

for source in "${SOURCES[@]}"; do
    [[ "$source" == */anchor_lf80_extended/* ]]         && ORDERED_SOURCES+=("$source")
done

for source in "${SOURCES[@]}"; do
    [[ "$source" == */anchor_lf80_refsize/* ]]         && ORDERED_SOURCES+=("$source")
done

for source in "${SOURCES[@]}"; do
    if [[ "$source" != */anchor_lf80_extended/* ]]         && [[ "$source" != */anchor_lf80_refsize/* ]]
    then
        ORDERED_SOURCES+=("$source")
    fi
done

# -----------------------------------------------------------
# 2. Extraer el registro candidato.
# -----------------------------------------------------------

rm -f "$CANDIDATE"
: > "$TASKDIR/extraction.log"

SELECTED_SOURCE=""

for SOURCE in "${ORDERED_SOURCES[@]}"; do
    echo "trying_source=$SOURCE"         | tee -a "$TASKDIR/extraction.log"

    rm -f "$CANDIDATE"

    if "$SINGULARITY" exec         --cleanenv         --bind "$PROJECT:/work"         "$SIF"         /opt/openmm-validation/bin/python         /work/src/validation/extract_exp06_candidate.py         "/work/${SOURCE#"$PROJECT"/}"         "/work/${CANDIDATE#"$PROJECT"/}"         "$ID"         >> "$TASKDIR/extraction.log" 2>&1
    then
        SELECTED_SOURCE="$SOURCE"
        break
    fi
done

cat "$TASKDIR/extraction.log"

if [ -z "$SELECTED_SOURCE" ] || [ ! -s "$CANDIDATE" ]; then
    echo "ERROR: candidate extraction failed for all source SDFs"
    exit 3
fi

echo "selected_source=$SELECTED_SOURCE"

# -----------------------------------------------------------
# 3. Identificar automáticamente los átomos fijos.
# -----------------------------------------------------------

"$SINGULARITY" exec     --cleanenv     --bind "$PROJECT:/work"     "$SIF"     /opt/openmm-validation/bin/python     /work/src/validation/identify_fixed_atoms.py     --project /work     --candidate "/work/${CANDIDATE#"$PROJECT"/}"     --config /work/config/validation/local_conditions.json     --output "/work/${FIXED_JSON#"$PROJECT"/}"     2>&1 | tee "$TASKDIR/fixed_atom_identification.log"

test -s "$FIXED_JSON"

# Leer del JSON el receptor seleccionado automáticamente.
PROTEIN="$(
    "$SINGULARITY" exec         --cleanenv         --bind "$PROJECT:/work"         "$SIF"         /opt/openmm-validation/bin/python         -c 'import json,sys; print(json.load(open(sys.argv[1]))["protein"])'         "/work/${FIXED_JSON#"$PROJECT"/}"
)"

echo "protein=$PROTEIN"

# -----------------------------------------------------------
# 4. Minimización restringida.
# -----------------------------------------------------------

"$SINGULARITY" exec     --cleanenv     --bind "$PROJECT:/work"     "$SIF"     /opt/openmm-validation/bin/python     /work/src/validation/minimize_restrained_complex.py     --protein "$PROTEIN"     --ligand "/work/${CANDIDATE#"$PROJECT"/}"     --fixed-json "/work/${FIXED_JSON#"$PROJECT"/}"     --outdir "/work/${MINOUT#"$PROJECT"/}"     --protein-k 1000     --warhead-k 10000     --max-iterations 3000     2>&1 | tee "$TASKDIR/minimization.log"

RESULT_JSON="$MINOUT/minimization_results.json"

test -s "$RESULT_JSON"

# -----------------------------------------------------------
# 5. Resumen individual.
# -----------------------------------------------------------

"$SINGULARITY" exec     --cleanenv     --bind "$PROJECT:/work"     "$SIF"     /opt/openmm-validation/bin/python     -     "/work/${RESULT_JSON#"$PROJECT"/}"     "$ID"     "$GROUP"     > "$TASKDIR/summary.tsv" <<'PY'
import json
import sys

path, candidate_id, group = sys.argv[1:4]

data = json.load(open(path))

columns = [
    "candidate_id",
    "group",
    "status",
    "initial_energy_kj_mol",
    "final_energy_kj_mol",
    "energy_change_kj_mol",
    "initial_severe_contacts_lt2A",
    "final_severe_contacts_lt2A",
    "initial_min_distance_A",
    "final_min_distance_A",
    "warhead_rmsd_A",
    "ligand_heavy_rmsd_A",
]

values = [
    candidate_id,
    group,
    "PASS",
    data["initial_energy_kj_mol"],
    data["final_energy_kj_mol"],
    data["energy_change_kj_mol"],
    data["initial_severe_contacts_lt2A"],
    data["final_severe_contacts_lt2A"],
    data["initial_min_protein_ligand_distance_A"],
    data["final_min_protein_ligand_distance_A"],
    data["warhead_rmsd_A"],
    data["ligand_heavy_rmsd_A"],
]

print("\t".join(columns))
print("\t".join(map(str, values)))
PY

{
    printf 'candidate_id\tgroup\tstatus\texit_code\n'
    printf '%s\t%s\tPASS\t0\n'         "$ID" "$GROUP"
} > "$TASKDIR/status.tsv"

echo
cat "$TASKDIR/summary.tsv"

echo
echo "TASK_STATUS=OK"
echo "candidate_id=$ID"
