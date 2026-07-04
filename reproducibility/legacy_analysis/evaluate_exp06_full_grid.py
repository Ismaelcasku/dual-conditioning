#!/usr/bin/env python3

import csv
import subprocess
import sys
from pathlib import Path


MANIFEST = Path(
    "artifacts/phase0_exp06_full_lambda_grid/"
    "experimental_design_master_75.tsv"
)

EVALUATOR = Path(
    "codes/utils/phase0_official_evaluator.py"
)

OUTROOT = Path(
    "artifacts/reports/phase0_official_eval/"
    "exp06_full_lambda_grid"
)

STATUS = OUTROOT / "evaluation_status_75.tsv"

A_SDF = {
    "x0874": (
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
        "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_ligand.sdf"
    ),
    "x0434": (
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
        "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_ligand.sdf"
    ),
}


OUTROOT.mkdir(parents=True, exist_ok=True)

with MANIFEST.open() as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if len(rows) != 75:
    raise SystemExit(
        "Expected 75 conditions; found {}".format(len(rows))
    )

status_rows = []

for number, row in enumerate(rows, start=1):
    pair_id = row["pair_id"]
    seed = row["seed"]
    lambda_value = row["lambda_global"]
    experiment_id = row["experiment_id"]

    outdir = (
        OUTROOT
        / pair_id
        / ("seed_" + seed)
        / ("lambda_" + lambda_value)
    )

    outdir.mkdir(parents=True, exist_ok=True)

    metrics = (
        outdir
        / (
            experiment_id
            + "_official_phase0_metrics.tsv"
        )
    )

    summary = (
        outdir
        / (
            experiment_id
            + "_official_phase0_summary.json"
        )
    )

    if metrics.is_file() and metrics.stat().st_size > 0:
        state = "REUSED_EXISTING_EVAL"
    else:
        command = [
            sys.executable,
            str(EVALUATOR),
            "--experiment_id",
            experiment_id,
            "--mode",
            "exp06_full_lambda_grid",
            "--lambda_global",
            lambda_value,
            "--lambda_local",
            "1.0",
            "--A_sdf",
            A_SDF[row["A_local"]],
            "--B_sdf",
            row["B_ligand"],
            "--gen_sdf",
            row["sdf_path"],
            "--fixed_json",
            row["fixed_json"],
            "--out_dir",
            str(outdir),
        ]

        print(
            "[{}/75] {} seed={} lambda={}".format(
                number,
                pair_id,
                seed,
                lambda_value,
            ),
            flush=True,
        )

        subprocess.run(command, check=True)
        state = "EVALUATED"

    if not metrics.is_file() or metrics.stat().st_size == 0:
        raise RuntimeError(
            "Missing metrics output: {}".format(metrics)
        )

    if not summary.is_file() or summary.stat().st_size == 0:
        raise RuntimeError(
            "Missing summary output: {}".format(summary)
        )

    status_rows.append({
        "pair_id": pair_id,
        "seed": seed,
        "lambda_global": lambda_value,
        "source": row["source"],
        "experiment_id": experiment_id,
        "state": state,
        "metrics_tsv": str(metrics),
        "summary_json": str(summary),
    })


with STATUS.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(status_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(status_rows)

print("EXP06_OFFICIAL_EVALUATION_DONE")
print("conditions={}".format(len(status_rows)))
print("status={}".format(STATUS))
