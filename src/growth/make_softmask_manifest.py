#!/usr/bin/env python3
"""Build the final centered, order-safe Phase-5 soft-mask manifest."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

SEEDS = [1101, 2202, 3303, 4404, 5505, 6606, 7707, 8808, 9909, 1010]
LAMBDA_GLOBAL = 20.0
VARIANTS = [("greedy", 1), ("beam", 3)]
RHO_VALUES = "1.0,0.75,0.5,0.25,0.0"
ADD_N = 4
OUT_ROOT = "artifacts/phase5_softmask"

PAIRS = [
    {"pair_id": "x0434_x2193", "A_local": "x0434", "B_global": "x2193"},
    {"pair_id": "x0874_x1093", "A_local": "x0874", "B_global": "x1093"},
]


def resolve_one(pattern: str, description: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"ERROR: no se encontró {description} con patrón: {pattern}")
    if len(matches) > 1:
        raise SystemExit(f"ERROR: {description} ambiguo:\n  " + "\n  ".join(matches))
    return matches[0]


def resolve_pair_paths(pair):
    a_complex = resolve_one(
        f"data/mpro/prepared/silvr_xchem_hits/{pair['A_local']}__*/*_complex.pdb",
        f"complex de {pair['A_local']}",
    )
    b_ligand = resolve_one(
        f"data/mpro/prepared/silvr_xchem_hits/{pair['B_global']}__*/*_ligand.sdf",
        f"ligando de {pair['B_global']}",
    )
    return a_complex, b_ligand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="artifacts/phase5_softmask/softmask_manifest.tsv"
    )
    parser.add_argument("--pilot", type=int, default=None)
    args = parser.parse_args()

    seeds = SEEDS[: args.pilot] if args.pilot else SEEDS
    rows = []
    task_id = 0
    for pair in PAIRS:
        a_complex, b_ligand = resolve_pair_paths(pair)
        for seed in seeds:
            for variant, k_beams in VARIANTS:
                out_dir = (
                    f"{OUT_ROOT}/{pair['pair_id']}/variant_{variant}/seed_{seed}"
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "pair_id": pair["pair_id"],
                        "A_local": pair["A_local"],
                        "B_global": pair["B_global"],
                        "variant": variant,
                        "k_beams": k_beams,
                        "seed": seed,
                        "lambda_global": LAMBDA_GLOBAL,
                        "add_n": ADD_N,
                        "rho_values": RHO_VALUES,
                        "experiment_id": f"softmask_{pair['pair_id']}_{variant}_seed{seed}",
                        "out_dir": out_dir,
                        "A_complex": a_complex,
                        "B_ligand": b_ligand,
                    }
                )
                task_id += 1

    fields = [
        "task_id",
        "pair_id",
        "A_local",
        "B_global",
        "variant",
        "k_beams",
        "seed",
        "lambda_global",
        "add_n",
        "rho_values",
        "experiment_id",
        "out_dir",
        "A_complex",
        "B_ligand",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {len(rows)} trayectorias en {out}")
    print(f"     semillas={len(seeds)} array=0-{len(rows)-1}")
    print(f"     add_n={ADD_N} rho={RHO_VALUES}")


if __name__ == "__main__":
    main()
