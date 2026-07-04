#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path


MANIFEST = Path(
    "artifacts/phase0_exp06_full_lambda_grid/"
    "experimental_design_master_75.tsv"
)

OUT = Path(
    "artifacts/phase0_exp06_full_lambda_grid/"
    "generation_yield_75.tsv"
)


def count_sdf_records(path):
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


with MANIFEST.open() as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if len(rows) != 75:
    raise SystemExit(
        "Expected 75 manifest rows; found {}".format(len(rows))
    )

output_rows = []
missing = []

for row in rows:
    sdf = Path(row["sdf_path"])

    if not sdf.is_file() or sdf.stat().st_size == 0:
        missing.append(str(sdf))
        records = 0
    else:
        records = count_sdf_records(sdf)

    output_rows.append({
        "pair_id": row["pair_id"],
        "seed": row["seed"],
        "lambda_global": row["lambda_global"],
        "source": row["source"],
        "requested": row["n_samples_requested"],
        "sdf_records": records,
        "sdf_path": row["sdf_path"],
    })

if missing:
    raise SystemExit(
        "Missing/empty SDF files:\n" + "\n".join(missing)
    )

with OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(output_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(output_rows)

groups = defaultdict(lambda: [0, 0, 0])

for row in output_rows:
    key = (row["pair_id"], row["lambda_global"])
    groups[key][0] += int(row["requested"])
    groups[key][1] += int(row["sdf_records"])
    groups[key][2] += 1

print("PAIR\tLAMBDA\tSEEDS\tRECORDS/REQUESTED\tYIELD")

for key in sorted(
    groups,
    key=lambda x: (x[0], float(x[1]))
):
    requested, records, seeds = groups[key]

    print(
        "{}\t{}\t{}\t{}/{}\t{:.1f}%".format(
            key[0],
            key[1],
            seeds,
            records,
            requested,
            100.0 * records / requested,
        )
    )

print()
print("TOTAL_RECORDS={}".format(
    sum(int(row["sdf_records"]) for row in output_rows)
))
print("OUTPUT={}".format(OUT))
