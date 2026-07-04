#!/usr/bin/env python3

import csv
from pathlib import Path


SEEDS = [1101, 2202, 3303, 4404, 5505]
LAMBDAS = ["0.0", "20.0", "50.0", "100.0", "200.0"]
N_SAMPLES = 10

OUTROOT = Path("artifacts/phase0_exp06_full_lambda_grid")
OUTROOT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    {
        "pair_id": "x0874_x1093",
        "A_local": "x0874",
        "B_global": "x1093",
        "A_complex": (
            "data/mpro/prepared/silvr_xchem_hits/"
            "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
            "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_complex.pdb"
        ),
        "B_ligand": (
            "data/mpro/prepared/silvr_xchem_hits/"
            "x1093__5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20/"
            "5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20_ligand.sdf"
        ),
        "fixed_json": (
            "data/mpro/manifests/"
            "b0_x0874_T54_fixed_atoms.json"
        ),
        "fixed_atoms": "C02 C04 C05 C06 C07 C08 C09",
        "reuse_lambdas": {"0.0", "50.0", "100.0"},
    },
    {
        "pair_id": "x0434_x1093",
        "A_local": "x0434",
        "B_global": "x1093",
        "A_complex": (
            "data/mpro/prepared/silvr_xchem_hits/"
            "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
            "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_complex.pdb"
        ),
        "B_ligand": (
            "data/mpro/prepared/silvr_xchem_hits/"
            "x1093__5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20/"
            "5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20_ligand.sdf"
        ),
        "fixed_json": (
            "artifacts/fixed_atom_screen/x0434_blind/"
            "x0434_fixed_anchor_EVALUATOR_V2.json"
        ),
        "fixed_atoms": "C4 C5 C6 N C1 C2 C3",
        "reuse_lambdas": set(),
    },
    {
        "pair_id": "x0434_x2193",
        "A_local": "x0434",
        "B_global": "x2193",
        "A_complex": (
            "data/mpro/prepared/silvr_xchem_hits/"
            "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
            "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_complex.pdb"
        ),
        "B_ligand": (
            "data/mpro/prepared/silvr_xchem_hits/"
            "x2193__5RHD_Mpro-x2193_AAR-POS-5507155c-1/"
            "5RHD_Mpro-x2193_AAR-POS-5507155c-1_ligand.sdf"
        ),
        "fixed_json": (
            "artifacts/fixed_atom_screen/x0434_blind/"
            "x0434_fixed_anchor_EVALUATOR_V2.json"
        ),
        "fixed_atoms": "C4 C5 C6 N C1 C2 C3",
        "reuse_lambdas": set(),
    },
]


MASTER_FIELDS = [
    "grid_id",
    "pair_id",
    "A_local",
    "B_global",
    "seed",
    "lambda_global",
    "n_samples_requested",
    "source",
    "experiment_id",
    "sdf_path",
    "A_complex",
    "B_ligand",
    "fixed_json",
    "fixed_atoms",
]

SUBMISSION_FIELDS = [
    "task_id",
    "pair_id",
    "A_local",
    "B_global",
    "seed",
    "lambda_global",
    "n_samples_requested",
    "experiment_id",
    "out_sdf",
    "A_complex",
    "B_ligand",
    "fixed_json",
    "fixed_atoms",
]


master_rows = []
submission_rows = []
missing_reused = []

grid_id = 0
task_id = 0

for pair in PAIRS:
    for seed in SEEDS:
        for lambda_value in LAMBDAS:

            pair_id = pair["pair_id"]

            if (
                pair_id == "x0874_x1093"
                and lambda_value in pair["reuse_lambdas"]
            ):
                source = "reuse_exp04"

                experiment_id = (
                    "exp04_x0874_x1093_"
                    "seed_{}_lambda_{}".format(
                        seed,
                        lambda_value,
                    )
                )

                sdf_path = Path(
                    "artifacts/phase0_exp04_seed_replicates/"
                    "x0874_x1093/"
                    "seed_{}/lambda_{}/"
                    "{}_n10.sdf".format(
                        seed,
                        lambda_value,
                        experiment_id,
                    )
                )

                if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
                    missing_reused.append(str(sdf_path))

            else:
                source = "generate_exp06"

                experiment_id = (
                    "exp06_{}_seed_{}_lambda_{}".format(
                        pair_id,
                        seed,
                        lambda_value,
                    )
                )

                sdf_path = (
                    OUTROOT
                    / pair_id
                    / "seed_{}".format(seed)
                    / "lambda_{}".format(lambda_value)
                    / "{}_n10.sdf".format(experiment_id)
                )

                submission_rows.append({
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "A_local": pair["A_local"],
                    "B_global": pair["B_global"],
                    "seed": seed,
                    "lambda_global": lambda_value,
                    "n_samples_requested": N_SAMPLES,
                    "experiment_id": experiment_id,
                    "out_sdf": str(sdf_path),
                    "A_complex": pair["A_complex"],
                    "B_ligand": pair["B_ligand"],
                    "fixed_json": pair["fixed_json"],
                    "fixed_atoms": pair["fixed_atoms"],
                })

                task_id += 1

            master_rows.append({
                "grid_id": grid_id,
                "pair_id": pair_id,
                "A_local": pair["A_local"],
                "B_global": pair["B_global"],
                "seed": seed,
                "lambda_global": lambda_value,
                "n_samples_requested": N_SAMPLES,
                "source": source,
                "experiment_id": experiment_id,
                "sdf_path": str(sdf_path),
                "A_complex": pair["A_complex"],
                "B_ligand": pair["B_ligand"],
                "fixed_json": pair["fixed_json"],
                "fixed_atoms": pair["fixed_atoms"],
            })

            grid_id += 1


if missing_reused:
    raise SystemExit(
        "Missing reused Exp04 files:\n" + "\n".join(missing_reused)
    )


master_path = OUTROOT / "experimental_design_master_75.tsv"
submission_path = OUTROOT / "submission_manifest_60.tsv"

with master_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=MASTER_FIELDS,
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(master_rows)

with submission_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=SUBMISSION_FIELDS,
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(submission_rows)


assert len(master_rows) == 75
assert len(submission_rows) == 60

print("EXP06_MANIFEST_DONE")
print("full_grid_conditions={}".format(len(master_rows)))
print("reused_exp04_conditions={}".format(
    sum(row["source"] == "reuse_exp04" for row in master_rows)
))
print("new_exp06_conditions={}".format(len(submission_rows)))
print("master={}".format(master_path))
print("submission={}".format(submission_path))
