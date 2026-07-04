#!/usr/bin/env python3

import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path


MASTER = Path(
    "artifacts/phase0_exp06_full_lambda_grid/"
    "experimental_design_master_75_canonical_x0434.tsv"
)

ROOT = Path(
    "artifacts/reports/phase0_official_eval/"
    "exp06_full_lambda_grid_canonical_x0434"
)

LAMBDAS = ["0.0", "20.0", "50.0", "100.0", "200.0"]

METRICS = [
    "shape_tanimoto_dist_to_B",
    "shape_protrude_dist_to_B",
    "shape_tanimoto_dist_to_A",
    "shape_protrude_dist_to_A",
    "best_fixed_geom_rmsd_A",
    "n_heavy",
]


def true_value(value):
    return str(value).strip().upper() == "TRUE"


def number(value):
    try:
        result = float(value)
    except Exception:
        return None

    if math.isnan(result):
        return None

    return result


def stats(values):
    if not values:
        return {
            "mean": "",
            "median": "",
            "sd": "",
            "minimum": "",
            "maximum": "",
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum": min(values),
        "maximum": max(values),
    }


def file_hash(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


with MASTER.open() as handle:
    design = list(csv.DictReader(handle, delimiter="\t"))

conditions = []

for item in design:
    directory = (
        ROOT
        / item["pair_id"]
        / ("seed_" + item["seed"])
        / ("lambda_" + item["lambda_global"])
    )

    metrics_files = sorted(
        directory.glob("*_official_phase0_metrics.tsv")
    )

    if len(metrics_files) != 1:
        raise RuntimeError(
            "Expected one metrics TSV in {}; found {}".format(
                directory,
                metrics_files,
            )
        )

    with metrics_files[0].open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    valid_rows = [
        row for row in rows
        if true_value(row.get("valid_rdkit_read"))
    ]

    condition = {
        "pair_id": item["pair_id"],
        "A_local": item["A_local"],
        "B_global": item["B_global"],
        "seed": item["seed"],
        "lambda_global": item["lambda_global"],
        "source": item["source"],
        "requested": int(item["n_samples_requested"]),
        "n_records": len(rows),
        "n_valid": len(valid_rows),
        "n_sanitize_ok": sum(
            row.get("sanitize_status") == "OK"
            for row in rows
        ),
        "n_local_pass": sum(
            true_value(
                row.get("local_pass_all_atoms_0p2A")
            )
            for row in rows
        ),
        "n_B_closer_Tani": sum(
            true_value(
                row.get(
                    "b_closer_than_A_shape_tanimoto"
                )
            )
            for row in rows
        ),
        "n_B_closer_Protrude": sum(
            true_value(
                row.get(
                    "b_closer_than_A_shape_protrude"
                )
            )
            for row in rows
        ),
        "n_strict_dual": sum(
            true_value(
                row.get("dual_candidate_vs_A_strict")
            )
            for row in rows
        ),
        "sdf_path": item["sdf_path"],
    }

    for metric in METRICS:
        values = []

        for row in valid_rows:
            value = number(row.get(metric))
            if value is not None:
                values.append(value)

        metric_stats = stats(values)

        for suffix in [
            "mean",
            "median",
            "sd",
            "minimum",
            "maximum",
        ]:
            condition[
                metric + "_" + suffix
            ] = metric_stats[suffix]

    conditions.append(condition)


condition_path = ROOT / "exp06_condition_level_summary.tsv"

with condition_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(conditions[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(conditions)


###############################################################################
# Across-seed summaries
###############################################################################

grouped = defaultdict(list)

for row in conditions:
    grouped[
        (row["pair_id"], row["lambda_global"])
    ].append(row)

across_rows = []

for key in sorted(
    grouped,
    key=lambda x: (x[0], float(x[1]))
):
    rows = grouped[key]

    result = {
        "pair_id": key[0],
        "lambda_global": key[1],
        "n_seeds": len(rows),
        "requested_total": sum(
            row["requested"] for row in rows
        ),
        "records_total": sum(
            row["n_records"] for row in rows
        ),
        "valid_total": sum(
            row["n_valid"] for row in rows
        ),
        "local_pass_total": sum(
            row["n_local_pass"] for row in rows
        ),
        "B_closer_Tani_total": sum(
            row["n_B_closer_Tani"] for row in rows
        ),
        "B_closer_Protrude_total": sum(
            row["n_B_closer_Protrude"]
            for row in rows
        ),
        "strict_dual_total": sum(
            row["n_strict_dual"] for row in rows
        ),
    }

    for metric in METRICS:
        seed_means = [
            float(row[metric + "_mean"])
            for row in rows
            if row[metric + "_mean"] != ""
        ]

        metric_stats = stats(seed_means)

        result[metric + "_seedmean_mean"] = (
            metric_stats["mean"]
        )
        result[metric + "_seedmean_sd"] = (
            metric_stats["sd"]
        )
        result[metric + "_seedmean_min"] = (
            metric_stats["minimum"]
        )
        result[metric + "_seedmean_max"] = (
            metric_stats["maximum"]
        )

    across_rows.append(result)


across_path = ROOT / "exp06_across_seed_summary.tsv"

with across_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(across_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(across_rows)


###############################################################################
# Paired changes against seed-matched lambda=0
###############################################################################

lookup = {
    (
        row["pair_id"],
        row["seed"],
        row["lambda_global"],
    ): row
    for row in conditions
}

paired_rows = []

for pair_id in sorted(set(
    row["pair_id"] for row in conditions
)):
    seeds = sorted(set(
        row["seed"]
        for row in conditions
        if row["pair_id"] == pair_id
    ))

    for seed in seeds:
        baseline = lookup[(pair_id, seed, "0.0")]

        for lambda_value in LAMBDAS[1:]:
            guided = lookup[
                (pair_id, seed, lambda_value)
            ]

            paired = {
                "pair_id": pair_id,
                "seed": seed,
                "lambda_global": lambda_value,
                "baseline_valid": baseline["n_valid"],
                "guided_valid": guided["n_valid"],
                "baseline_local_pass":
                    baseline["n_local_pass"],
                "guided_local_pass":
                    guided["n_local_pass"],
            }

            for metric in METRICS:
                baseline_value = float(
                    baseline[metric + "_mean"]
                )
                guided_value = float(
                    guided[metric + "_mean"]
                )

                paired[metric + "_baseline"] = (
                    baseline_value
                )
                paired[metric + "_guided"] = (
                    guided_value
                )
                paired["delta_" + metric] = (
                    guided_value - baseline_value
                )

            paired_rows.append(paired)


paired_path = (
    ROOT / "exp06_paired_seed_deltas_vs_lambda0.tsv"
)

with paired_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(paired_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(paired_rows)


###############################################################################
# Paired-effect summary; seed is the replicate unit
###############################################################################

effect_rows = []

for pair_id in sorted(set(
    row["pair_id"] for row in paired_rows
)):
    for lambda_value in LAMBDAS[1:]:
        selected = [
            row for row in paired_rows
            if row["pair_id"] == pair_id
            and row["lambda_global"] == lambda_value
        ]

        for metric in [
            "shape_tanimoto_dist_to_B",
            "shape_protrude_dist_to_B",
            "shape_tanimoto_dist_to_A",
            "shape_protrude_dist_to_A",
            "n_heavy",
        ]:
            deltas = [
                float(row["delta_" + metric])
                for row in selected
            ]

            delta_stats = stats(deltas)

            effect_rows.append({
                "pair_id": pair_id,
                "lambda_global": lambda_value,
                "metric": metric,
                "n_seeds": len(deltas),
                "paired_delta_mean":
                    delta_stats["mean"],
                "paired_delta_median":
                    delta_stats["median"],
                "paired_delta_sd":
                    delta_stats["sd"],
                "minimum_delta":
                    delta_stats["minimum"],
                "maximum_delta":
                    delta_stats["maximum"],
                "n_negative": sum(
                    delta < 0 for delta in deltas
                ),
                "n_positive": sum(
                    delta > 0 for delta in deltas
                ),
                "n_zero": sum(
                    delta == 0 for delta in deltas
                ),
            })


effect_path = ROOT / "exp06_paired_effect_summary.tsv"

with effect_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(effect_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(effect_rows)


###############################################################################
# Check that duplicated x0434 lambda=0 controls are exactly identical
###############################################################################

identity_rows = []

for seed in ["1101", "2202", "3303", "4404", "5505"]:
    first = lookup[
        ("x0434_x1093", seed, "0.0")
    ]["sdf_path"]

    second = lookup[
        ("x0434_x2193", seed, "0.0")
    ]["sdf_path"]

    hash_first = file_hash(first)
    hash_second = file_hash(second)

    identity_rows.append({
        "seed": seed,
        "x0434_x1093_lambda0_sha256": hash_first,
        "x0434_x2193_lambda0_sha256": hash_second,
        "byte_identical": (
            "TRUE" if hash_first == hash_second
            else "FALSE"
        ),
    })


identity_path = ROOT / "exp06_x0434_lambda0_identity.tsv"

with identity_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(identity_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(identity_rows)


###############################################################################
# Compact terminal output
###############################################################################

print("EXP06_AGGREGATION_DONE")

print()
print("ACROSS-SEED DOSE RESPONSE")
print(
    "pair\tlambda\trecords/requested\tlocal\tstrict\t"
    "ShapeTaniB_mean\tShapeTaniB_sd\t"
    "ProtrudeB_mean\tProtrudeB_sd\tnHeavy_mean"
)

for row in across_rows:
    print(
        "{}\t{}\t{}/{}\t{}/{}\t{}\t"
        "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.3f}".format(
            row["pair_id"],
            row["lambda_global"],
            row["records_total"],
            row["requested_total"],
            row["local_pass_total"],
            row["valid_total"],
            row["strict_dual_total"],
            float(
                row[
                    "shape_tanimoto_dist_to_B_"
                    "seedmean_mean"
                ]
            ),
            float(
                row[
                    "shape_tanimoto_dist_to_B_"
                    "seedmean_sd"
                ]
            ),
            float(
                row[
                    "shape_protrude_dist_to_B_"
                    "seedmean_mean"
                ]
            ),
            float(
                row[
                    "shape_protrude_dist_to_B_"
                    "seedmean_sd"
                ]
            ),
            float(
                row[
                    "n_heavy_seedmean_mean"
                ]
            ),
        )
    )

print()
print("PAIRED EFFECTS VS LAMBDA=0")
print(
    "pair\tlambda\tmetric\tmean_delta\tsd\t"
    "negative_seeds/5\tmin\tmax"
)

for row in effect_rows:
    if row["metric"] not in [
        "shape_tanimoto_dist_to_B",
        "shape_protrude_dist_to_B",
    ]:
        continue

    print(
        "{}\t{}\t{}\t{:.6f}\t{:.6f}\t"
        "{}/{}\t{:.6f}\t{:.6f}".format(
            row["pair_id"],
            row["lambda_global"],
            row["metric"],
            float(row["paired_delta_mean"]),
            float(row["paired_delta_sd"]),
            row["n_negative"],
            row["n_seeds"],
            float(row["minimum_delta"]),
            float(row["maximum_delta"]),
        )
    )

print()
print("X0434 DUPLICATED LAMBDA=0 CONTROL")
print(
    "byte_identical_seeds={}/5".format(
        sum(
            row["byte_identical"] == "TRUE"
            for row in identity_rows
        )
    )
)

print()
print("OUTPUTS")
print(condition_path)
print(across_path)
print(paired_path)
print(effect_path)
print(identity_path)
