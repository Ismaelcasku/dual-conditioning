#!/usr/bin/env python3

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(
    "artifacts/reports/phase0_official_eval/"
    "exp05_x0434_lambda50_pilot"
)

TARGETS = ["x1093", "x2193"]
LAMBDAS = ["0.0", "50.0"]

METRICS = [
    "shape_tanimoto_dist_to_A",
    "shape_tanimoto_dist_to_B",
    "shape_protrude_dist_to_A",
    "shape_protrude_dist_to_B",
    "best_fixed_geom_rmsd_A",
    "n_heavy",
]


def as_float(value):
    try:
        number = float(value)
    except Exception:
        return None

    if math.isnan(number):
        return None

    return number


def is_true(value):
    return str(value).strip().upper() == "TRUE"


def summarize(values):
    if not values:
        return {
            "mean": "",
            "median": "",
            "sd": "",
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


summaries = []

for target in TARGETS:
    for lambda_value in LAMBDAS:
        directory = ROOT / target / ("lambda_" + lambda_value)

        files = sorted(
            directory.glob("*_official_phase0_metrics.tsv")
        )

        if len(files) != 1:
            raise RuntimeError(
                "Expected one metrics TSV in {}; found {}".format(
                    directory,
                    files,
                )
            )

        with files[0].open() as handle:
            rows = list(
                csv.DictReader(handle, delimiter="\t")
            )

        valid_rows = [
            row for row in rows
            if is_true(row.get("valid_rdkit_read"))
        ]

        row_summary = {
            "B_global": target,
            "lambda_global": lambda_value,
            "n_sdf_records": len(rows),
            "n_valid": len(valid_rows),
            "n_sanitize_ok": sum(
                row.get("sanitize_status") == "OK"
                for row in rows
            ),
            "n_local_pass": sum(
                is_true(row.get("local_pass_all_atoms_0p2A"))
                for row in rows
            ),
            "n_B_closer_Tani": sum(
                is_true(
                    row.get(
                        "b_closer_than_A_shape_tanimoto"
                    )
                )
                for row in rows
            ),
            "n_B_closer_Protrude": sum(
                is_true(
                    row.get(
                        "b_closer_than_A_shape_protrude"
                    )
                )
                for row in rows
            ),
            "n_strict_dual": sum(
                is_true(
                    row.get("dual_candidate_vs_A_strict")
                )
                for row in rows
            ),
        }

        for metric in METRICS:
            values = []

            for row in valid_rows:
                value = as_float(row.get(metric))

                if value is not None:
                    values.append(value)

            stats = summarize(values)

            row_summary[metric + "_mean"] = stats["mean"]
            row_summary[metric + "_median"] = stats["median"]
            row_summary[metric + "_sd"] = stats["sd"]

        summaries.append(row_summary)


summary_path = ROOT / "exp05_pilot_condition_summary.tsv"

summary_fields = list(summaries[0].keys())

with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=summary_fields,
        delimiter="\t",
    )

    writer.writeheader()
    writer.writerows(summaries)


paired_rows = []

for target in TARGETS:
    baseline = next(
        row for row in summaries
        if row["B_global"] == target
        and row["lambda_global"] == "0.0"
    )

    guided = next(
        row for row in summaries
        if row["B_global"] == target
        and row["lambda_global"] == "50.0"
    )

    paired = {
        "B_global": target,
        "baseline_records": baseline["n_sdf_records"],
        "guided_records": guided["n_sdf_records"],
        "baseline_local_pass": baseline["n_local_pass"],
        "guided_local_pass": guided["n_local_pass"],
    }

    for metric in [
        "shape_tanimoto_dist_to_B",
        "shape_protrude_dist_to_B",
        "shape_tanimoto_dist_to_A",
        "shape_protrude_dist_to_A",
        "best_fixed_geom_rmsd_A",
        "n_heavy",
    ]:
        baseline_value = float(
            baseline[metric + "_mean"]
        )

        guided_value = float(
            guided[metric + "_mean"]
        )

        paired[metric + "_baseline"] = baseline_value
        paired[metric + "_guided"] = guided_value
        paired["delta_" + metric] = (
            guided_value - baseline_value
        )

    paired_rows.append(paired)


paired_path = ROOT / "exp05_pilot_paired_deltas.tsv"

paired_fields = list(paired_rows[0].keys())

with paired_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=paired_fields,
        delimiter="\t",
    )

    writer.writeheader()
    writer.writerows(paired_rows)


print("EXP05_PILOT_AGGREGATION_DONE")
print("summary={}".format(summary_path))
print("paired={}".format(paired_path))

print()
print("CORE RESULTS")
print(
    "B_global\tlambda\tn_valid\tlocal_pass\t"
    "ShapeTaniB_mean\tProtrudeB_mean\t"
    "ShapeTaniA_mean\tProtrudeA_mean\t"
    "RMSD_fixed_mean\tnHeavy_mean"
)

for row in summaries:
    print(
        "{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t"
        "{:.6f}\t{:.6f}\t{:.6f}\t{:.3f}".format(
            row["B_global"],
            row["lambda_global"],
            row["n_valid"],
            row["n_local_pass"],
            float(
                row[
                    "shape_tanimoto_dist_to_B_mean"
                ]
            ),
            float(
                row[
                    "shape_protrude_dist_to_B_mean"
                ]
            ),
            float(
                row[
                    "shape_tanimoto_dist_to_A_mean"
                ]
            ),
            float(
                row[
                    "shape_protrude_dist_to_A_mean"
                ]
            ),
            float(
                row[
                    "best_fixed_geom_rmsd_A_mean"
                ]
            ),
            float(row["n_heavy_mean"]),
        )
    )

print()
print("PAIRED DELTAS: lambda50 minus lambda0")
print("Negative B-distance delta = steering toward B")
print(
    "B_global\tdelta_ShapeTaniB\tdelta_ProtrudeB\t"
    "delta_ShapeTaniA\tdelta_ProtrudeA\t"
    "delta_RMSD_fixed\tdelta_nHeavy"
)

for row in paired_rows:
    print(
        "{}\t{:.6f}\t{:.6f}\t{:.6f}\t"
        "{:.6f}\t{:.6f}\t{:.3f}".format(
            row["B_global"],
            row[
                "delta_shape_tanimoto_dist_to_B"
            ],
            row[
                "delta_shape_protrude_dist_to_B"
            ],
            row[
                "delta_shape_tanimoto_dist_to_A"
            ],
            row[
                "delta_shape_protrude_dist_to_A"
            ],
            row[
                "delta_best_fixed_geom_rmsd_A"
            ],
            row["delta_n_heavy"],
        )
    )
