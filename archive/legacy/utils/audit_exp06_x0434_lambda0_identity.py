#!/usr/bin/env python3

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem


ROOT = Path("artifacts/phase0_exp06_full_lambda_grid")
SUMMARY = Path(
    "artifacts/reports/phase0_official_eval/"
    "exp06_full_lambda_grid/"
    "exp06_condition_level_summary.tsv"
)

SEEDS = ["1101", "2202", "3303", "4404", "5505"]


def read_records(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        sanitize=False,
        removeHs=False,
    )

    return [mol for mol in supplier]


def canonical_smiles(mol):
    if mol is None:
        return "<READ_FAIL>"

    copy = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(copy)
    except Exception:
        pass

    try:
        return Chem.MolToSmiles(
            copy,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return "<SMILES_FAIL>"


def atom_symbols(mol):
    if mol is None:
        return []

    return [
        atom.GetSymbol()
        for atom in mol.GetAtoms()
    ]


def coordinates(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()

    return np.asarray([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in range(mol.GetNumAtoms())
    ], dtype=float)


def compare_files(first_path, second_path):
    first = read_records(first_path)
    second = read_records(second_path)

    smiles_first = [canonical_smiles(mol) for mol in first]
    smiles_second = [canonical_smiles(mol) for mol in second]

    same_smiles_sequence = smiles_first == smiles_second
    same_smiles_multiset = (
        Counter(smiles_first) == Counter(smiles_second)
    )

    n_exact_graph_order = 0
    coordinate_rmsds = []

    for mol_a, mol_b in zip(first, second):
        if mol_a is None or mol_b is None:
            continue

        if atom_symbols(mol_a) != atom_symbols(mol_b):
            continue

        coord_a = coordinates(mol_a)
        coord_b = coordinates(mol_b)

        if (
            coord_a is None
            or coord_b is None
            or coord_a.shape != coord_b.shape
        ):
            continue

        n_exact_graph_order += 1

        rmsd = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (coord_a - coord_b) ** 2,
                        axis=1,
                    )
                )
            )
        )

        coordinate_rmsds.append(rmsd)

    return {
        "records_first": len(first),
        "records_second": len(second),
        "same_smiles_sequence": same_smiles_sequence,
        "same_smiles_multiset": same_smiles_multiset,
        "matched_atom_order_records": n_exact_graph_order,
        "maximum_coordinate_rmsd_A": (
            max(coordinate_rmsds)
            if coordinate_rmsds
            else ""
        ),
        "mean_coordinate_rmsd_A": (
            float(np.mean(coordinate_rmsds))
            if coordinate_rmsds
            else ""
        ),
    }


summary_lookup = {}

with SUMMARY.open() as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row["lambda_global"] != "0.0":
            continue

        if row["pair_id"] not in [
            "x0434_x1093",
            "x0434_x2193",
        ]:
            continue

        summary_lookup[
            (row["pair_id"], row["seed"])
        ] = row


output_rows = []

for seed in SEEDS:
    first = (
        ROOT
        / "x0434_x1093"
        / ("seed_" + seed)
        / "lambda_0.0"
        / (
            "exp06_x0434_x1093_seed_"
            + seed
            + "_lambda_0.0_n10.sdf"
        )
    )

    second = (
        ROOT
        / "x0434_x2193"
        / ("seed_" + seed)
        / "lambda_0.0"
        / (
            "exp06_x0434_x2193_seed_"
            + seed
            + "_lambda_0.0_n10.sdf"
        )
    )

    result = compare_files(first, second)

    first_summary = summary_lookup[
        ("x0434_x1093", seed)
    ]

    second_summary = summary_lookup[
        ("x0434_x2193", seed)
    ]

    result.update({
        "seed": seed,
        "delta_n_heavy_mean": (
            float(
                second_summary["n_heavy_mean"]
            )
            - float(
                first_summary["n_heavy_mean"]
            )
        ),
        "delta_ShapeTaniA_mean": (
            float(
                second_summary[
                    "shape_tanimoto_dist_to_A_mean"
                ]
            )
            - float(
                first_summary[
                    "shape_tanimoto_dist_to_A_mean"
                ]
            )
        ),
        "delta_ProtrudeA_mean": (
            float(
                second_summary[
                    "shape_protrude_dist_to_A_mean"
                ]
            )
            - float(
                first_summary[
                    "shape_protrude_dist_to_A_mean"
                ]
            )
        ),
    })

    output_rows.append(result)


output = Path(
    "artifacts/reports/phase0_official_eval/"
    "exp06_full_lambda_grid/"
    "exp06_x0434_lambda0_structural_identity.tsv"
)

fields = [
    "seed",
    "records_first",
    "records_second",
    "same_smiles_sequence",
    "same_smiles_multiset",
    "matched_atom_order_records",
    "maximum_coordinate_rmsd_A",
    "mean_coordinate_rmsd_A",
    "delta_n_heavy_mean",
    "delta_ShapeTaniA_mean",
    "delta_ProtrudeA_mean",
]

with output.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(output_rows)


print("X0434 LAMBDA=0 STRUCTURAL IDENTITY AUDIT")
print(
    "seed\trecords\tSMILES_sequence\tSMILES_multiset\t"
    "coord_matches\tmax_coord_RMSD\t"
    "delta_nHeavy\tdelta_TaniA\tdelta_ProtrudeA"
)

for row in output_rows:
    print(
        "{}\t{}/{}\t{}\t{}\t{}\t{}\t"
        "{:.6f}\t{:.6f}\t{:.6f}".format(
            row["seed"],
            row["records_first"],
            row["records_second"],
            row["same_smiles_sequence"],
            row["same_smiles_multiset"],
            row["matched_atom_order_records"],
            row["maximum_coordinate_rmsd_A"],
            row["delta_n_heavy_mean"],
            row["delta_ShapeTaniA_mean"],
            row["delta_ProtrudeA_mean"],
        )
    )

print()
print("output={}".format(output))
