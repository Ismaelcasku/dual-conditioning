#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDConfig
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdMolDescriptors,
    rdShapeHelpers,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import the exact DiffSBDD postprocessing implementation.
DIFFSBDD_ROOT = Path("codes/vendor/diffsbdd").resolve()
sys.path.insert(0, str(DIFFSBDD_ROOT))

from analysis.molecule_builder import process_molecule

sys.path.append(
    os.path.join(RDConfig.RDContribDir, "SA_Score")
)
import sascorer


REFERENCE_A = {
    "x0434": (
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
        "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_ligand.sdf"
    ),
    "x0874": (
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
        "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_ligand.sdf"
    ),
}

REFERENCE_B = {
    "x1093": (
        "data/mpro/prepared/silvr_xchem_hits/"
        "x1093__5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20/"
        "5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20_ligand.sdf"
    ),
    "x2193": (
        "data/mpro/prepared/silvr_xchem_hits/"
        "x2193__5RHD_Mpro-x2193_AAR-POS-5507155c-1/"
        "5RHD_Mpro-x2193_AAR-POS-5507155c-1_ligand.sdf"
    ),
}

# Zero-based indices in the reference ligands.
FIXED_INDICES = {
    "x0434": [1, 2, 3, 7, 10, 12, 13],
    "x0874": [2, 3, 4, 5, 6, 7, 8],
}

TIERS = [
    "diffsbdd_native",
    "anchor_largest",
    "anchor_lf80_extended",
    "anchor_lf90_extended",
    "anchor_lf80_refsize",
]


def load_reference(path):
    mol = Chem.MolFromMolFile(
        str(path),
        sanitize=False,
        removeHs=False,
    )

    if mol is None:
        raise RuntimeError(
            f"Could not load reference molecule: {path}"
        )

    Chem.SanitizeMol(mol)

    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            f"Reference has no conformer: {path}"
        )

    return mol


def xyz(mol, atom_index):
    position = mol.GetConformer().GetAtomPosition(atom_index)

    return (
        float(position.x),
        float(position.y),
        float(position.z),
    )


def distance(first, second):
    return math.sqrt(sum(
        (a - b) ** 2
        for a, b in zip(first, second)
    ))


def anchor_match(reference, generated, fixed_indices):
    candidates = []

    for fixed_position, reference_index in enumerate(fixed_indices):
        reference_atom = reference.GetAtomWithIdx(reference_index)
        reference_xyz = xyz(reference, reference_index)

        for generated_index, generated_atom in enumerate(
            generated.GetAtoms()
        ):
            if (
                generated_atom.GetAtomicNum()
                != reference_atom.GetAtomicNum()
            ):
                continue

            candidates.append((
                distance(
                    reference_xyz,
                    xyz(generated, generated_index),
                ),
                fixed_position,
                generated_index,
            ))

    candidates.sort()

    assigned_fixed = set()
    assigned_generated = set()
    assignments = []

    for current_distance, fixed_position, generated_index in candidates:
        if fixed_position in assigned_fixed:
            continue

        if generated_index in assigned_generated:
            continue

        assigned_fixed.add(fixed_position)
        assigned_generated.add(generated_index)

        assignments.append((
            fixed_position,
            generated_index,
            current_distance,
        ))

        if len(assignments) == len(fixed_indices):
            break

    distances = [
        assignment[2]
        for assignment in assignments
    ]

    matched = len(assignments)
    maximum_distance = max(distances) if distances else None

    preserved = (
        matched == len(fixed_indices)
        and maximum_distance is not None
        and maximum_distance <= 0.20
    )

    return {
        "fixed_atoms_expected": len(fixed_indices),
        "fixed_atoms_matched": matched,
        "fixed_match_max_distance": maximum_distance,
        "anchor_preserved": preserved,
    }


def shape_metrics(mol, reference_a, reference_b):
    tani_a = float(
        rdShapeHelpers.ShapeTanimotoDist(
            mol,
            reference_a,
            ignoreHs=True,
        )
    )

    tani_b = float(
        rdShapeHelpers.ShapeTanimotoDist(
            mol,
            reference_b,
            ignoreHs=True,
        )
    )

    protrude_a = float(
        rdShapeHelpers.ShapeProtrudeDist(
            mol,
            reference_a,
            ignoreHs=True,
        )
    )

    protrude_b = float(
        rdShapeHelpers.ShapeProtrudeDist(
            mol,
            reference_b,
            ignoreHs=True,
        )
    )

    return {
        "parent_tanimoto_A": tani_a,
        "parent_tanimoto_B": tani_b,
        "parent_protrude_A": protrude_a,
        "parent_protrude_B": protrude_b,
        "parent_strict_dual": (
            tani_b < tani_a
            and protrude_b < protrude_a
        ),
    }


def chemistry_metrics(mol):
    descriptor_mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(descriptor_mol)

    smiles = Chem.MolToSmiles(
        descriptor_mol,
        canonical=True,
        isomericSmiles=True,
    )

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=descriptor_mol,
        includeChirality=True,
    )

    molecular_weight = float(
        Descriptors.MolWt(descriptor_mol)
    )

    logp = float(
        Crippen.MolLogP(descriptor_mol)
    )

    tpsa = float(
        rdMolDescriptors.CalcTPSA(descriptor_mol)
    )

    hbd = int(
        Lipinski.NumHDonors(descriptor_mol)
    )

    hba = int(
        Lipinski.NumHAcceptors(descriptor_mol)
    )

    return {
        "parent_smiles": smiles,
        "parent_scaffold": scaffold,
        "parent_nheavy":
            descriptor_mol.GetNumHeavyAtoms(),
        "parent_qed":
            float(QED.qed(descriptor_mol)),
        "parent_sa":
            float(sascorer.calculateScore(descriptor_mol)),
        "parent_mw": molecular_weight,
        "parent_logp": logp,
        "parent_tpsa": tpsa,
        "parent_hbd": hbd,
        "parent_hba": hba,
        "parent_rings": int(
            rdMolDescriptors.CalcNumRings(
                descriptor_mol
            )
        ),
        "parent_formal_charge": int(sum(
            atom.GetFormalCharge()
            for atom in descriptor_mol.GetAtoms()
        )),
        "parent_ro5": (
            molecular_weight <= 500
            and logp <= 5
            and hbd <= 5
            and hba <= 10
        ),
    }


def mean(values):
    clean = []

    for value in values:
        if value in ("", None):
            continue

        value = float(value)

        if math.isfinite(value):
            clean.append(value)

    return statistics.mean(clean) if clean else ""


def write_tsv(path, rows, fields):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def set_properties(mol, row, tier):
    output = Chem.Mol(mol)

    output.SetProp(
        "_Name",
        (
            f"{row['pair_id']}_"
            f"seed{row['seed']}_"
            f"lambda{row['lambda_global']}_"
            f"sample{row['sample']}"
        ),
    )

    properties = {
        "pair_id": row["pair_id"],
        "A_local": row["A_local"],
        "B_global": row["B_global"],
        "seed": row["seed"],
        "lambda_global": row["lambda_global"],
        "sample": row["sample"],
        "postprocessing_tier": tier,
        "raw_heavy_atoms": row["raw_nheavy"],
        "parent_heavy_atoms": row["parent_nheavy"],
        "parent_heavy_fraction":
            row["parent_heavy_fraction"],
        "anchor_preserved": row["anchor_preserved"],
        "parent_tanimoto_B":
            row["parent_tanimoto_B"],
        "parent_protrude_B":
            row["parent_protrude_B"],
        "parent_qed": row["parent_qed"],
        "parent_sa": row["parent_sa"],
    }

    for name, value in properties.items():
        output.SetProp(name, str(value))

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audit_path = Path(args.audit)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open() as handle:
        manifest = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    if len(manifest) != 75:
        raise RuntimeError(
            f"Expected 75 manifest rows; found {len(manifest)}"
        )

    with audit_path.open() as handle:
        audit_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    audit_index = {
        (
            row["pair_id"],
            row["seed"],
            row["lambda_global"],
            row["sample"],
        ): row
        for row in audit_rows
    }

    references_a = {
        name: load_reference(Path(path))
        for name, path in REFERENCE_A.items()
    }

    references_b = {
        name: load_reference(Path(path))
        for name, path in REFERENCE_B.items()
    }

    per_molecule = []
    accepted_molecules = defaultdict(list)

    for condition_number, condition in enumerate(
        manifest,
        start=1,
    ):
        pair_id = condition["pair_id"]
        seed = condition["seed"]
        lambda_global = condition["lambda_global"]
        sdf_path = Path(condition["release_sdf_path"])

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=False,
            removeHs=False,
        )

        records = list(supplier)

        print(
            f"[{condition_number}/75] "
            f"{pair_id} seed={seed} "
            f"lambda={lambda_global} "
            f"records={len(records)}",
            flush=True,
        )

        for sample, raw_mol in enumerate(records, start=1):
            audit_row = audit_index.get(
                (
                    pair_id,
                    seed,
                    lambda_global,
                    str(sample),
                ),
                {},
            )

            row = {
                "pair_id": pair_id,
                "A_local": condition["A_local"],
                "B_global": condition["B_global"],
                "seed": seed,
                "lambda_global": lambda_global,
                "sample": sample,
                "raw_sdf": str(sdf_path),
                "raw_status": "OK",
                "native_status": "NOT_PROCESSED",
                "raw_nheavy": "",
                "raw_n_heavy_fragments":
                    audit_row.get(
                        "n_heavy_fragments",
                        "",
                    ),
            }

            for tier in TIERS:
                row[f"accepted_{tier}"] = False

            if raw_mol is None:
                row["raw_status"] = "READ_FAIL"
                per_molecule.append(row)
                continue

            row["raw_nheavy"] = raw_mol.GetNumHeavyAtoms()

            try:
                native = process_molecule(
                    raw_mol,
                    add_hydrogens=False,
                    sanitize=True,
                    relax_iter=0,
                    largest_frag=True,
                )
            except Exception as error:
                row["native_status"] = (
                    f"PROCESS_EXCEPTION:"
                    f"{type(error).__name__}"
                )
                per_molecule.append(row)
                continue

            if native is None:
                row["native_status"] = "PROCESS_FAILED"
                per_molecule.append(row)
                continue

            if native.GetNumConformers() == 0:
                row["native_status"] = "NO_CONFORMER"
                per_molecule.append(row)
                continue

            try:
                Chem.SanitizeMol(native)

                row.update(
                    chemistry_metrics(native)
                )

                row.update(
                    shape_metrics(
                        native,
                        references_a[
                            condition["A_local"]
                        ],
                        references_b[
                            condition["B_global"]
                        ],
                    )
                )

                row.update(
                    anchor_match(
                        references_a[
                            condition["A_local"]
                        ],
                        native,
                        FIXED_INDICES[
                            condition["A_local"]
                        ],
                    )
                )

            except Exception as error:
                row["native_status"] = (
                    f"METRIC_EXCEPTION:"
                    f"{type(error).__name__}"
                )
                per_molecule.append(row)
                continue

            row["native_status"] = "OK"

            raw_nheavy = int(row["raw_nheavy"])
            parent_nheavy = int(row["parent_nheavy"])

            row["parent_heavy_fraction"] = (
                parent_nheavy / raw_nheavy
                if raw_nheavy else 0.0
            )

            accepted = ["diffsbdd_native"]

            if row["anchor_preserved"]:
                accepted.append("anchor_largest")

                if (
                    row["parent_heavy_fraction"] >= 0.80
                    and 11 <= parent_nheavy <= 25
                ):
                    accepted.append(
                        "anchor_lf80_extended"
                    )

                if (
                    row["parent_heavy_fraction"] >= 0.90
                    and 11 <= parent_nheavy <= 25
                ):
                    accepted.append(
                        "anchor_lf90_extended"
                    )

                if (
                    row["parent_heavy_fraction"] >= 0.80
                    and 11 <= parent_nheavy <= 19
                ):
                    accepted.append(
                        "anchor_lf80_refsize"
                    )

            for tier in accepted:
                row[f"accepted_{tier}"] = True

                key = (
                    tier,
                    pair_id,
                    seed,
                    lambda_global,
                )

                accepted_molecules[key].append(
                    set_properties(
                        native,
                        row,
                        tier,
                    )
                )

            per_molecule.append(row)

    molecule_fields = [
        "pair_id",
        "A_local",
        "B_global",
        "seed",
        "lambda_global",
        "sample",
        "raw_sdf",
        "raw_status",
        "native_status",
        "raw_nheavy",
        "raw_n_heavy_fragments",
        "parent_nheavy",
        "parent_heavy_fraction",
        "fixed_atoms_expected",
        "fixed_atoms_matched",
        "fixed_match_max_distance",
        "anchor_preserved",
        "parent_smiles",
        "parent_scaffold",
        "parent_qed",
        "parent_sa",
        "parent_mw",
        "parent_logp",
        "parent_tpsa",
        "parent_hbd",
        "parent_hba",
        "parent_rings",
        "parent_formal_charge",
        "parent_ro5",
        "parent_tanimoto_A",
        "parent_tanimoto_B",
        "parent_protrude_A",
        "parent_protrude_B",
        "parent_strict_dual",
    ] + [
        f"accepted_{tier}"
        for tier in TIERS
    ]

    write_tsv(
        out_dir / "postprocessing_per_molecule.tsv",
        per_molecule,
        molecule_fields,
    )

    generated_manifest = []

    for condition in manifest:
        for tier in TIERS:
            key = (
                tier,
                condition["pair_id"],
                condition["seed"],
                condition["lambda_global"],
            )

            molecules = accepted_molecules.get(
                key,
                [],
            )

            lambda_label = (
                condition["lambda_global"]
                .rstrip("0")
                .rstrip(".")
            )

            output_sdf = (
                out_dir
                / "sdf"
                / tier
                / condition["pair_id"]
                / f"seed_{condition['seed']}"
                / f"lambda_{lambda_label}"
                / "postprocessed.sdf"
            )

            output_sdf.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if molecules:
                writer = Chem.SDWriter(
                    str(output_sdf)
                )

                for mol in molecules:
                    writer.write(mol)

                writer.close()
            else:
                output_sdf.write_text("")

            generated_manifest.append({
                "tier": tier,
                "pair_id": condition["pair_id"],
                "A_local": condition["A_local"],
                "B_global": condition["B_global"],
                "seed": condition["seed"],
                "lambda_global":
                    condition["lambda_global"],
                "n_requested":
                    condition["n_samples_requested"],
                "n_accepted": len(molecules),
                "postprocessed_sdf":
                    str(output_sdf),
            })

    write_tsv(
        out_dir / "postprocessed_manifest.tsv",
        generated_manifest,
        [
            "tier",
            "pair_id",
            "A_local",
            "B_global",
            "seed",
            "lambda_global",
            "n_requested",
            "n_accepted",
            "postprocessed_sdf",
        ],
    )

    summaries = []

    pair_ids = sorted({
        row["pair_id"]
        for row in manifest
    })

    lambdas = sorted(
        {
            row["lambda_global"]
            for row in manifest
        },
        key=float,
    )

    for tier in TIERS:
        for pair_id in pair_ids:
            for lambda_global in lambdas:
                conditions = [
                    row
                    for row in manifest
                    if (
                        row["pair_id"] == pair_id
                        and row["lambda_global"]
                        == lambda_global
                    )
                ]

                requested = sum(
                    int(row["n_samples_requested"])
                    for row in conditions
                )

                available = [
                    row
                    for row in per_molecule
                    if (
                        row["pair_id"] == pair_id
                        and row["lambda_global"]
                        == lambda_global
                    )
                ]

                accepted = [
                    row
                    for row in available
                    if row.get(
                        f"accepted_{tier}",
                        False,
                    )
                ]

                smiles = [
                    row["parent_smiles"]
                    for row in accepted
                ]

                summaries.append({
                    "tier": tier,
                    "pair_id": pair_id,
                    "lambda_global": lambda_global,
                    "n_requested": requested,
                    "n_raw_records": len(available),
                    "n_accepted": len(accepted),
                    "yield_requested": (
                        len(accepted) / requested
                        if requested else 0.0
                    ),
                    "unique_rate": (
                        len(set(smiles)) / len(smiles)
                        if smiles else 0.0
                    ),
                    "mean_parent_fraction": mean([
                        row.get(
                            "parent_heavy_fraction"
                        )
                        for row in accepted
                    ]),
                    "mean_parent_nheavy": mean([
                        row.get("parent_nheavy")
                        for row in accepted
                    ]),
                    "mean_qed": mean([
                        row.get("parent_qed")
                        for row in accepted
                    ]),
                    "mean_sa": mean([
                        row.get("parent_sa")
                        for row in accepted
                    ]),
                    "mean_logp": mean([
                        row.get("parent_logp")
                        for row in accepted
                    ]),
                    "mean_tpsa": mean([
                        row.get("parent_tpsa")
                        for row in accepted
                    ]),
                    "mean_tanimoto_A": mean([
                        row.get("parent_tanimoto_A")
                        for row in accepted
                    ]),
                    "mean_tanimoto_B": mean([
                        row.get("parent_tanimoto_B")
                        for row in accepted
                    ]),
                    "mean_protrude_A": mean([
                        row.get("parent_protrude_A")
                        for row in accepted
                    ]),
                    "mean_protrude_B": mean([
                        row.get("parent_protrude_B")
                        for row in accepted
                    ]),
                    "strict_dual_count": sum(
                        bool(
                            row.get(
                                "parent_strict_dual"
                            )
                        )
                        for row in accepted
                    ),
                })

    summary_fields = list(
        summaries[0].keys()
    )

    write_tsv(
        out_dir / "postprocessing_across_seed_summary.tsv",
        summaries,
        summary_fields,
    )

    metadata = {
        "schema":
            "exp06_diffsbdd_native_anchor_aware_v1",
        "diffsbdd_native_call": {
            "add_hydrogens": False,
            "sanitize": True,
            "largest_frag": True,
            "relax_iter": 0,
        },
        "diffsbdd_native_selection": (
            "Largest connected fragment by total atom count, "
            "exactly as implemented by "
            "analysis.molecule_builder.process_molecule."
        ),
        "primary_dual_conditioning_tier":
            "anchor_lf80_extended",
        "anchor_tolerance_angstrom": 0.20,
        "tiers": {
            "diffsbdd_native": (
                "Exact native largest-fragment filtering."
            ),
            "anchor_largest": (
                "Native largest fragment must preserve all "
                "fixed warhead atoms."
            ),
            "anchor_lf80_extended": (
                "anchor_largest, parent fraction >=0.80, "
                "11-25 heavy atoms."
            ),
            "anchor_lf90_extended": (
                "anchor_largest, parent fraction >=0.90, "
                "11-25 heavy atoms."
            ),
            "anchor_lf80_refsize": (
                "anchor_largest, parent fraction >=0.80, "
                "11-19 heavy atoms."
            ),
        },
        "uff_relaxation": False,
        "new_bonds_inferred": False,
    }

    (
        out_dir / "postprocessing_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("POSTPROCESSING SUMMARY")
    print(
        "tier\tpair\tlambda\taccepted/requested\t"
        "yield\tparent_fraction\tnHeavy\tQED\tSA\t"
        "TaniB\tProtrudeB\tstrict"
    )

    for row in summaries:
        if row["tier"] not in {
            "diffsbdd_native",
            "anchor_lf80_extended",
            "anchor_lf90_extended",
            "anchor_lf80_refsize",
        }:
            continue

        def fmt(value):
            if value == "":
                return "NA"

            return f"{float(value):.3f}"

        print(
            "{}\t{}\t{}\t{}/{}\t{:.3f}\t{}\t{}\t"
            "{}\t{}\t{}\t{}\t{}".format(
                row["tier"],
                row["pair_id"],
                row["lambda_global"],
                row["n_accepted"],
                row["n_requested"],
                row["yield_requested"],
                fmt(row["mean_parent_fraction"]),
                fmt(row["mean_parent_nheavy"]),
                fmt(row["mean_qed"]),
                fmt(row["mean_sa"]),
                fmt(row["mean_tanimoto_B"]),
                fmt(row["mean_protrude_B"]),
                row["strict_dual_count"],
            )
        )

    print()
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
