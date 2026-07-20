#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


ROOT = Path.cwd()

RUNROOT = (
    ROOT
    / "artifacts/phase0_exp04_seed_replicates"
    / "x0874_x1093"
)

EVALROOT = (
    ROOT
    / "artifacts/reports/phase0_official_eval"
    / "exp04_seed_replicates"
    / "x0874_x1093"
)

SEEDS = [1101, 2202, 3303, 4404, 5505]
LAMBDAS = [0.0, 50.0, 100.0]

METRICS = [
    "shape_tanimoto_dist_to_B",
    "shape_protrude_dist_to_B",
    "shape_tanimoto_dist_to_A",
    "shape_protrude_dist_to_A",
    "n_heavy",
    "best_fixed_geom_rmsd_A",
]


def as_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def metric_file(seed, lam):
    outdir = EVALROOT / f"seed_{seed}" / f"lambda_{lam}"

    expected = (
        outdir
        / f"exp04_x0874_x1093_seed_{seed}_lambda_{lam}"
          "_official_phase0_metrics.tsv"
    )

    if expected.exists():
        return expected

    candidates = sorted(outdir.glob("*_official_phase0_metrics.tsv"))

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one metrics TSV in {outdir}; found {candidates}"
        )

    return candidates[0]


def sdf_file(seed, lam):
    return (
        RUNROOT
        / f"seed_{seed}"
        / f"lambda_{lam}"
        / f"exp04_x0874_x1093_seed_{seed}_lambda_{lam}_n10.sdf"
    )


def count_sdf_records(path):
    count = 0

    with path.open(errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1

    return count


def summarize_condition(seed, lam):
    path = metric_file(seed, lam)

    with path.open() as handle:
        rows_all = list(csv.DictReader(handle, delimiter="\t"))

    valid = [
        row
        for row in rows_all
        if row.get("valid_rdkit_read") == "TRUE"
    ]

    result = {
        "seed": seed,
        "lambda_global": lam,
        "requested": 10,
        "sdf_records": count_sdf_records(sdf_file(seed, lam)),
        "evaluator_records": len(rows_all),
        "valid": len(valid),
        "sanitize_ok": sum(
            row.get("sanitize_status") == "OK"
            or row.get("sanitize_ok") == "TRUE"
            for row in valid
        ),
        "local_pass": sum(
            row.get("local_pass_all_atoms_0p2A") == "TRUE"
            for row in valid
        ),
        "B_better_Tani": sum(
            row.get("b_closer_than_A_shape_tanimoto") == "TRUE"
            for row in valid
        ),
        "B_better_Protrude": sum(
            row.get("b_closer_than_A_shape_protrude") == "TRUE"
            for row in valid
        ),
        "strict_dual": sum(
            row.get("dual_candidate_vs_A_strict") == "TRUE"
            for row in valid
        ),
    }

    for metric in METRICS:
        values = []
        for row in valid:
            value = as_float(row.get(metric))
            if value is not None:
                values.append(value)

        result[f"{metric}_mean"] = mean(values) if values else None
        result[f"{metric}_median"] = median(values) if values else None
        result[f"{metric}_sd"] = (
            stdev(values) if len(values) > 1 else 0.0 if values else None
        )

    return result


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


condition_rows = [
    summarize_condition(seed, lam)
    for seed in SEEDS
    for lam in LAMBDAS
]

condition_fields = [
    "seed",
    "lambda_global",
    "requested",
    "sdf_records",
    "evaluator_records",
    "valid",
    "sanitize_ok",
    "local_pass",
    "B_better_Tani",
    "B_better_Protrude",
    "strict_dual",
]

for metric in METRICS:
    condition_fields.extend([
        f"{metric}_mean",
        f"{metric}_median",
        f"{metric}_sd",
    ])

seed_summary_path = EVALROOT / "exp04_seed_level_summary.tsv"
write_tsv(seed_summary_path, condition_rows, condition_fields)

###############################################################################
# Paired differences within each seed
###############################################################################

lookup = {
    (row["seed"], row["lambda_global"]): row
    for row in condition_rows
}

paired_rows = []

for seed in SEEDS:
    baseline = lookup[(seed, 0.0)]

    for lam in [50.0, 100.0]:
        guided = lookup[(seed, lam)]

        paired_rows.append({
            "seed": seed,
            "lambda_global": lam,
            "baseline_valid": baseline["valid"],
            "guided_valid": guided["valid"],

            "ShapeTaniB_baseline":
                baseline["shape_tanimoto_dist_to_B_mean"],
            "ShapeTaniB_guided":
                guided["shape_tanimoto_dist_to_B_mean"],
            "delta_ShapeTaniB":
                guided["shape_tanimoto_dist_to_B_mean"]
                - baseline["shape_tanimoto_dist_to_B_mean"],

            "ProtrudeB_baseline":
                baseline["shape_protrude_dist_to_B_mean"],
            "ProtrudeB_guided":
                guided["shape_protrude_dist_to_B_mean"],
            "delta_ProtrudeB":
                guided["shape_protrude_dist_to_B_mean"]
                - baseline["shape_protrude_dist_to_B_mean"],

            "n_heavy_baseline":
                baseline["n_heavy_mean"],
            "n_heavy_guided":
                guided["n_heavy_mean"],
            "delta_n_heavy":
                guided["n_heavy_mean"] - baseline["n_heavy_mean"],

            "local_pass_baseline":
                baseline["local_pass"],
            "local_pass_guided":
                guided["local_pass"],
        })

paired_fields = [
    "seed",
    "lambda_global",
    "baseline_valid",
    "guided_valid",
    "ShapeTaniB_baseline",
    "ShapeTaniB_guided",
    "delta_ShapeTaniB",
    "ProtrudeB_baseline",
    "ProtrudeB_guided",
    "delta_ProtrudeB",
    "n_heavy_baseline",
    "n_heavy_guided",
    "delta_n_heavy",
    "local_pass_baseline",
    "local_pass_guided",
]

paired_path = EVALROOT / "exp04_paired_seed_deltas.tsv"
write_tsv(paired_path, paired_rows, paired_fields)

###############################################################################
# Summary across seeds: seed is the replicate unit
###############################################################################

across_seed_rows = []

for lam in LAMBDAS:
    subset = [
        row for row in condition_rows
        if row["lambda_global"] == lam
    ]

    output = {
        "lambda_global": lam,
        "n_seeds": len(subset),
        "requested_total": sum(row["requested"] for row in subset),
        "sdf_records_total": sum(row["sdf_records"] for row in subset),
        "valid_total": sum(row["valid"] for row in subset),
        "local_pass_total": sum(row["local_pass"] for row in subset),
        "B_better_Tani_total": sum(
            row["B_better_Tani"] for row in subset
        ),
        "B_better_Protrude_total": sum(
            row["B_better_Protrude"] for row in subset
        ),
        "strict_dual_total": sum(row["strict_dual"] for row in subset),
    }

    for metric in [
        "shape_tanimoto_dist_to_B_mean",
        "shape_protrude_dist_to_B_mean",
        "shape_tanimoto_dist_to_A_mean",
        "shape_protrude_dist_to_A_mean",
        "n_heavy_mean",
        "best_fixed_geom_rmsd_A_mean",
    ]:
        values = [
            row[metric]
            for row in subset
            if row[metric] is not None
        ]

        output[f"{metric}_across_seed_mean"] = mean(values)
        output[f"{metric}_across_seed_sd"] = (
            stdev(values) if len(values) > 1 else 0.0
        )
        output[f"{metric}_across_seed_min"] = min(values)
        output[f"{metric}_across_seed_max"] = max(values)

    across_seed_rows.append(output)

across_fields = list(across_seed_rows[0].keys())

across_path = EVALROOT / "exp04_across_seed_summary.tsv"
write_tsv(across_path, across_seed_rows, across_fields)

###############################################################################
# Paired-effect summary and consistency
###############################################################################

effect_rows = []

for lam in [50.0, 100.0]:
    subset = [
        row for row in paired_rows
        if row["lambda_global"] == lam
    ]

    for metric in ["delta_ShapeTaniB", "delta_ProtrudeB"]:
        values = [row[metric] for row in subset]

        effect_rows.append({
            "lambda_global": lam,
            "metric": metric,
            "n_seeds": len(values),
            "paired_delta_mean": mean(values),
            "paired_delta_median": median(values),
            "paired_delta_sd": stdev(values),
            "n_seeds_improved": sum(value < 0 for value in values),
            "n_seeds_worsened": sum(value > 0 for value in values),
            "minimum_delta": min(values),
            "maximum_delta": max(values),
        })

effect_fields = [
    "lambda_global",
    "metric",
    "n_seeds",
    "paired_delta_mean",
    "paired_delta_median",
    "paired_delta_sd",
    "n_seeds_improved",
    "n_seeds_worsened",
    "minimum_delta",
    "maximum_delta",
]

effect_path = EVALROOT / "exp04_paired_effect_summary.tsv"
write_tsv(effect_path, effect_rows, effect_fields)

print("EXP04_AGGREGATION_DONE")
print(f"seed_level={seed_summary_path}")
print(f"paired_deltas={paired_path}")
print(f"across_seed={across_path}")
print(f"paired_effects={effect_path}")
