#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


EXPECTED_LAMBDAS = {"0.0", "20.0", "50.0", "100.0", "200.0"}

PARENT_METRICS = [
    "n_heavy_parent",
    "mol_wt_parent",
    "qed_parent",
    "sa_score_parent",
    "logp_parent",
    "tpsa_parent",
    "hbd_parent",
    "hba_parent",
    "rotatable_bonds_parent",
    "n_rings_parent",
    "n_aromatic_rings_parent",
    "fraction_csp3_parent",
    "formal_charge_parent",
    "n_charged_atoms_parent",
    "ro5_violations_parent",
]

FULL_RECORD_METRICS = [
    "n_heavy_full",
    "n_fragments",
    "parent_heavy_fraction",
]


def bool_text(value):
    return "TRUE" if bool(value) else "FALSE"


def safe_float(value):
    try:
        result = float(value)
    except Exception:
        return None

    if math.isnan(result):
        return None

    return result


def describe(values):
    clean = [
        float(value)
        for value in values
        if value is not None
        and not math.isnan(float(value))
    ]

    if not clean:
        return {
            "mean": "",
            "median": "",
            "sd": "",
            "minimum": "",
            "maximum": "",
        }

    return {
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "minimum": min(clean),
        "maximum": max(clean),
    }


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()

        for source in rows:
            row = {}

            for field in fields:
                value = source.get(field, "")

                if isinstance(value, bool):
                    value = bool_text(value)
                elif isinstance(value, float) and math.isnan(value):
                    value = ""

                row[field] = value

            writer.writerow(row)


def load_sa_scorer(project_root):
    sa_dir = (
        project_root
        / "codes"
        / "vendor"
        / "diffsbdd"
        / "analysis"
        / "SA_Score"
    )

    if not sa_dir.is_dir():
        raise SystemExit(
            "SA_Score directory not found: {}".format(sa_dir)
        )

    sys.path.insert(0, str(sa_dir))

    try:
        import sascorer
    except Exception as exc:
        raise SystemExit(
            "Could not import sascorer: {}".format(repr(exc))
        )

    return sascorer


def sanitize_molecule(mol):
    if mol is None:
        return None, "READ_FAIL", "RDKit supplier returned None"

    copied = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(copied)
        return copied, "OK", ""
    except Exception as exc:
        return copied, "WARN", repr(exc).replace("\t", " ").replace("\n", " ")


def select_parent(full_mol):
    fragments = list(
        Chem.GetMolFrags(
            full_mol,
            asMols=True,
            sanitizeFrags=False,
        )
    )

    if not fragments:
        raise ValueError("No molecular fragments found")

    heavy_counts = [
        fragment.GetNumHeavyAtoms()
        for fragment in fragments
    ]

    parent_index = max(
        range(len(fragments)),
        key=lambda index: (
            heavy_counts[index],
            fragments[index].GetNumAtoms(),
            -index,
        ),
    )

    parent_with_h = Chem.Mol(fragments[parent_index])
    Chem.SanitizeMol(parent_with_h)

    # Frozen convention for physicochemical descriptors:
    # largest connected fragment, explicit H atoms removed.
    parent = Chem.RemoveHs(parent_with_h)
    Chem.SanitizeMol(parent)

    total_heavy = full_mol.GetNumHeavyAtoms()
    parent_heavy = parent.GetNumHeavyAtoms()

    parent_fraction = (
        float(parent_heavy) / total_heavy
        if total_heavy > 0
        else 0.0
    )

    return parent, fragments, parent_fraction


def canonical_smiles(mol):
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def scaffold_smiles(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""

    return canonical_smiles(scaffold)


def parent_descriptors(parent, sa_scorer):
    atoms = list(parent.GetAtoms())
    charges = [atom.GetFormalCharge() for atom in atoms]
    radicals = [
        atom.GetNumRadicalElectrons()
        for atom in atoms
    ]

    molecular_weight = float(Descriptors.MolWt(parent))
    logp = float(Crippen.MolLogP(parent))
    tpsa = float(rdMolDescriptors.CalcTPSA(parent))
    hbd = int(Lipinski.NumHDonors(parent))
    hba = int(Lipinski.NumHAcceptors(parent))

    ro5_violations = int(sum([
        molecular_weight > 500.0,
        logp > 5.0,
        hbd > 5,
        hba > 10,
    ]))

    return {
        "canonical_smiles_parent": canonical_smiles(parent),
        "murcko_scaffold_parent": scaffold_smiles(parent),
        "formula_parent": rdMolDescriptors.CalcMolFormula(parent),
        "elements_parent": ",".join(sorted(set(
            atom.GetSymbol()
            for atom in atoms
        ))),
        "n_heavy_parent": int(parent.GetNumHeavyAtoms()),
        "mol_wt_parent": molecular_weight,
        "qed_parent": float(QED.qed(parent)),
        "sa_score_parent": float(sa_scorer.calculateScore(parent)),
        "logp_parent": logp,
        "tpsa_parent": tpsa,
        "hbd_parent": hbd,
        "hba_parent": hba,
        "rotatable_bonds_parent": int(
            Lipinski.NumRotatableBonds(parent)
        ),
        "n_rings_parent": int(
            rdMolDescriptors.CalcNumRings(parent)
        ),
        "n_aromatic_rings_parent": int(
            rdMolDescriptors.CalcNumAromaticRings(parent)
        ),
        "fraction_csp3_parent": float(
            rdMolDescriptors.CalcFractionCSP3(parent)
        ),
        "formal_charge_parent": int(sum(charges)),
        "n_charged_atoms_parent": int(sum(
            charge != 0
            for charge in charges
        )),
        "max_abs_atom_formal_charge_parent": int(
            max([abs(charge) for charge in charges] or [0])
        ),
        "radical_electrons_parent": int(sum(radicals)),
        "ro5_violations_parent": ro5_violations,
        "passes_ro5_parent": ro5_violations == 0,
    }


def preflight_manifest(rows):
    if len(rows) != 75:
        raise SystemExit(
            "Expected 75 manifest rows; found {}".format(len(rows))
        )

    grouped = defaultdict(list)

    for row in rows:
        key = (row["pair_id"], row["seed"])
        grouped[key].append(row)

        sdf = Path(row["release_sdf_path"])

        if not sdf.is_file() or sdf.stat().st_size == 0:
            raise SystemExit(
                "Missing or empty SDF: {}".format(sdf)
            )

    pairs = sorted(set(
        row["pair_id"]
        for row in rows
    ))

    seeds = sorted(set(
        row["seed"]
        for row in rows
    ))

    if len(pairs) != 3:
        raise SystemExit(
            "Expected 3 pairs; found {}".format(pairs)
        )

    if len(seeds) != 5:
        raise SystemExit(
            "Expected 5 seeds; found {}".format(seeds)
        )

    problems = []

    for key, group in sorted(grouped.items()):
        lambdas = {
            row["lambda_global"]
            for row in group
        }

        if lambdas != EXPECTED_LAMBDAS:
            problems.append(
                "{} seed {}: lambdas={}".format(
                    key[0],
                    key[1],
                    sorted(lambdas),
                )
            )

        if len(group) != 5:
            problems.append(
                "{} seed {}: rows={}".format(
                    key[0],
                    key[1],
                    len(group),
                )
            )

    if problems:
        raise SystemExit(
            "Incomplete factorial design:\n"
            + "\n".join(problems)
        )

    print("MANIFEST_PREFLIGHT_OK")
    print("pairs={}".format(",".join(pairs)))
    print("seeds={}".format(",".join(seeds)))
    print("conditions={}".format(len(rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--project_root", default=".")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    project_root = Path(args.project_root).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open() as handle:
        design = list(
            csv.DictReader(handle, delimiter="\t")
        )

    preflight_manifest(design)
    sa_scorer = load_sa_scorer(project_root)

    molecule_rows = []
    condition_rows = []
    duplicate_rows = []
    red_flag_rows = []

    for condition_number, condition in enumerate(design, start=1):
        sdf_path = Path(condition["release_sdf_path"])

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=False,
            removeHs=False,
        )

        records = list(supplier)
        current_rows = []

        print(
            "[{}/75] {} seed={} lambda={} records={}".format(
                condition_number,
                condition["pair_id"],
                condition["seed"],
                condition["lambda_global"],
                len(records),
            ),
            flush=True,
        )

        for sample, original in enumerate(records, start=1):
            full_mol, sanitize_status, sanitize_message = (
                sanitize_molecule(original)
            )

            row = {
                "pair_id": condition["pair_id"],
                "A_local": condition["A_local"],
                "B_global": condition["B_global"],
                "seed": condition["seed"],
                "lambda_global": condition["lambda_global"],
                "experiment_id": condition["experiment_id"],
                "sample": sample,
                "requested_per_condition": int(
                    condition["n_samples_requested"]
                ),
                "sdf_path": str(sdf_path),
                "valid_rdkit_read": original is not None,
                "sanitize_status": sanitize_status,
                "sanitize_message": sanitize_message,
                "parent_evaluation_ok": False,
                "parent_evaluation_message": "",
            }

            flags = []

            if original is None:
                flags.append("RDKit_read_fail")

            if sanitize_status != "OK":
                flags.append("sanitize_fail")

            if full_mol is not None:
                row["n_heavy_full"] = int(
                    full_mol.GetNumHeavyAtoms()
                )
                row["n_atoms_full_with_explicit_H"] = int(
                    full_mol.GetNumAtoms()
                )

            if sanitize_status == "OK":
                try:
                    parent, fragments, parent_fraction = (
                        select_parent(full_mol)
                    )

                    row["n_fragments"] = len(fragments)
                    row["is_single_fragment"] = (
                        len(fragments) == 1
                    )
                    row["parent_heavy_fraction"] = (
                        parent_fraction
                    )

                    row.update(
                        parent_descriptors(
                            parent,
                            sa_scorer,
                        )
                    )

                    row["parent_evaluation_ok"] = True

                    if len(fragments) > 1:
                        flags.append("fragmented")

                    if parent_fraction < 0.8:
                        flags.append(
                            "parent_heavy_fraction_lt_0p8"
                        )

                    if row["radical_electrons_parent"] > 0:
                        flags.append("radical_parent")

                    if abs(row["formal_charge_parent"]) > 2:
                        flags.append(
                            "absolute_parent_charge_gt_2"
                        )

                    if (
                        row[
                            "max_abs_atom_formal_charge_parent"
                        ]
                        > 1
                    ):
                        flags.append(
                            "atom_formal_charge_abs_gt_1"
                        )

                except Exception as exc:
                    row["parent_evaluation_message"] = (
                        repr(exc)
                        .replace("\t", " ")
                        .replace("\n", " ")
                    )
                    flags.append("parent_evaluation_fail")

            row["red_flags"] = ",".join(flags)
            row["has_red_flag"] = bool(flags)

            molecule_rows.append(row)
            current_rows.append(row)

            if flags:
                red_flag_rows.append(row)

        usable_rows = [
            row for row in current_rows
            if row.get("parent_evaluation_ok")
        ]

        smiles = [
            row["canonical_smiles_parent"]
            for row in usable_rows
            if row.get("canonical_smiles_parent")
        ]

        scaffolds = [
            row["murcko_scaffold_parent"]
            for row in usable_rows
            if row.get("murcko_scaffold_parent")
        ]

        for smiles_value, count in Counter(smiles).items():
            if count > 1:
                duplicate_rows.append({
                    "pair_id": condition["pair_id"],
                    "seed": condition["seed"],
                    "lambda_global": condition["lambda_global"],
                    "canonical_smiles_parent": smiles_value,
                    "count": count,
                })

        requested = int(
            condition["n_samples_requested"]
        )

        condition_summary = {
            "pair_id": condition["pair_id"],
            "A_local": condition["A_local"],
            "B_global": condition["B_global"],
            "seed": condition["seed"],
            "lambda_global": condition["lambda_global"],
            "experiment_id": condition["experiment_id"],
            "requested": requested,
            "n_records": len(current_rows),
            "output_yield": (
                len(current_rows) / requested
            ),
            "n_rdkit_valid": sum(
                row["valid_rdkit_read"]
                for row in current_rows
            ),
            "n_sanitize_ok": sum(
                row["sanitize_status"] == "OK"
                for row in current_rows
            ),
            "n_parent_ok": len(usable_rows),
            "usable_yield": (
                len(usable_rows) / requested
            ),
            "rdkit_validity_rate_records": (
                sum(
                    row["valid_rdkit_read"]
                    for row in current_rows
                ) / len(current_rows)
                if current_rows else 0.0
            ),
            "sanitize_rate_records": (
                sum(
                    row["sanitize_status"] == "OK"
                    for row in current_rows
                ) / len(current_rows)
                if current_rows else 0.0
            ),
            "n_connected": sum(
                row.get("is_single_fragment", False)
                for row in usable_rows
            ),
            "connected_rate": (
                sum(
                    row.get("is_single_fragment", False)
                    for row in usable_rows
                ) / len(usable_rows)
                if usable_rows else 0.0
            ),
            "n_unique_parent_smiles": len(set(smiles)),
            "uniqueness_rate": (
                len(set(smiles)) / len(smiles)
                if smiles else 0.0
            ),
            "n_unique_parent_scaffolds": len(
                set(scaffolds)
            ),
            "scaffold_uniqueness_rate": (
                len(set(scaffolds)) / len(scaffolds)
                if scaffolds else 0.0
            ),
            "neutral_rate": (
                sum(
                    row["formal_charge_parent"] == 0
                    for row in usable_rows
                ) / len(usable_rows)
                if usable_rows else 0.0
            ),
            "ro5_pass_rate": (
                sum(
                    row["passes_ro5_parent"]
                    for row in usable_rows
                ) / len(usable_rows)
                if usable_rows else 0.0
            ),
            "n_red_flagged": sum(
                row["has_red_flag"]
                for row in current_rows
            ),
        }

        for metric in (
            PARENT_METRICS + FULL_RECORD_METRICS
        ):
            values = [
                safe_float(row.get(metric))
                for row in usable_rows
            ]

            metric_stats = describe(values)

            for suffix, value in metric_stats.items():
                condition_summary[
                    metric + "_" + suffix
                ] = value

        condition_rows.append(condition_summary)

    molecule_fields = [
        "pair_id",
        "A_local",
        "B_global",
        "seed",
        "lambda_global",
        "experiment_id",
        "sample",
        "requested_per_condition",
        "sdf_path",
        "valid_rdkit_read",
        "sanitize_status",
        "sanitize_message",
        "parent_evaluation_ok",
        "parent_evaluation_message",
        "n_atoms_full_with_explicit_H",
        "n_heavy_full",
        "n_fragments",
        "is_single_fragment",
        "parent_heavy_fraction",
        "canonical_smiles_parent",
        "murcko_scaffold_parent",
        "formula_parent",
        "elements_parent",
        "n_heavy_parent",
        "mol_wt_parent",
        "qed_parent",
        "sa_score_parent",
        "logp_parent",
        "tpsa_parent",
        "hbd_parent",
        "hba_parent",
        "rotatable_bonds_parent",
        "n_rings_parent",
        "n_aromatic_rings_parent",
        "fraction_csp3_parent",
        "formal_charge_parent",
        "n_charged_atoms_parent",
        "max_abs_atom_formal_charge_parent",
        "radical_electrons_parent",
        "ro5_violations_parent",
        "passes_ro5_parent",
        "has_red_flag",
        "red_flags",
    ]

    write_tsv(
        out_dir / "chemical_quality_per_molecule.tsv",
        molecule_rows,
        molecule_fields,
    )

    write_tsv(
        out_dir / "chemical_quality_condition_summary.tsv",
        condition_rows,
        list(condition_rows[0].keys()),
    )

    write_tsv(
        out_dir / "within_condition_duplicate_smiles.tsv",
        duplicate_rows,
        [
            "pair_id",
            "seed",
            "lambda_global",
            "canonical_smiles_parent",
            "count",
        ],
    )

    write_tsv(
        out_dir / "chemical_red_flags.tsv",
        red_flag_rows,
        molecule_fields,
    )

    ###########################################################################
    # Across-seed summaries
    ###########################################################################

    grouped = defaultdict(list)

    for row in condition_rows:
        grouped[
            (row["pair_id"], row["lambda_global"])
        ].append(row)

    across_rows = []

    for key in sorted(
        grouped,
        key=lambda value: (
            value[0],
            float(value[1]),
        ),
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
            "parent_ok_total": sum(
                row["n_parent_ok"] for row in rows
            ),
            "connected_total": sum(
                row["n_connected"] for row in rows
            ),
            "red_flagged_total": sum(
                row["n_red_flagged"] for row in rows
            ),
        }

        for field in [
            "output_yield",
            "usable_yield",
            "rdkit_validity_rate_records",
            "sanitize_rate_records",
            "connected_rate",
            "uniqueness_rate",
            "scaffold_uniqueness_rate",
            "neutral_rate",
            "ro5_pass_rate",
        ]:
            values = [
                safe_float(row.get(field))
                for row in rows
            ]

            stats = describe(values)
            result[field + "_seedmean"] = stats["mean"]
            result[field + "_seedsd"] = stats["sd"]

        for metric in (
            PARENT_METRICS + FULL_RECORD_METRICS
        ):
            values = [
                safe_float(row.get(metric + "_mean"))
                for row in rows
            ]

            stats = describe(values)
            result[metric + "_seedmean"] = stats["mean"]
            result[metric + "_seedsd"] = stats["sd"]

        across_rows.append(result)

    write_tsv(
        out_dir / "chemical_quality_across_seed_summary.tsv",
        across_rows,
        list(across_rows[0].keys()),
    )

    ###########################################################################
    # Paired seed deltas against lambda=0
    ###########################################################################

    lookup = {
        (
            row["pair_id"],
            row["seed"],
            row["lambda_global"],
        ): row
        for row in condition_rows
    }

    paired_rows = []

    delta_fields = [
        "output_yield",
        "usable_yield",
        "connected_rate",
        "uniqueness_rate",
        "scaffold_uniqueness_rate",
        "neutral_rate",
        "ro5_pass_rate",
        "n_heavy_full_mean",
        "n_heavy_parent_mean",
        "parent_heavy_fraction_mean",
        "qed_parent_mean",
        "sa_score_parent_mean",
        "logp_parent_mean",
        "tpsa_parent_mean",
        "n_rings_parent_mean",
        "formal_charge_parent_mean",
    ]

    for pair_id in sorted(set(
        row["pair_id"]
        for row in condition_rows
    )):
        seeds = sorted(set(
            row["seed"]
            for row in condition_rows
            if row["pair_id"] == pair_id
        ))

        for seed in seeds:
            baseline_key = (pair_id, seed, "0.0")

            if baseline_key not in lookup:
                raise RuntimeError(
                    "Missing lambda=0 baseline: {}".format(
                        baseline_key
                    )
                )

            baseline = lookup[baseline_key]

            for lambda_value in [
                "20.0",
                "50.0",
                "100.0",
                "200.0",
            ]:
                guided_key = (
                    pair_id,
                    seed,
                    lambda_value,
                )

                if guided_key not in lookup:
                    raise RuntimeError(
                        "Missing guided condition: {}".format(
                            guided_key
                        )
                    )

                guided = lookup[guided_key]

                result = {
                    "pair_id": pair_id,
                    "seed": seed,
                    "lambda_global": lambda_value,
                }

                for field in delta_fields:
                    base_value = safe_float(
                        baseline.get(field)
                    )
                    guided_value = safe_float(
                        guided.get(field)
                    )

                    result[field + "_baseline"] = (
                        "" if base_value is None
                        else base_value
                    )

                    result[field + "_guided"] = (
                        "" if guided_value is None
                        else guided_value
                    )

                    result["delta_" + field] = (
                        ""
                        if base_value is None
                        or guided_value is None
                        else guided_value - base_value
                    )

                paired_rows.append(result)

    write_tsv(
        out_dir
        / "chemical_quality_paired_seed_deltas_vs_lambda0.tsv",
        paired_rows,
        list(paired_rows[0].keys()),
    )

    ###########################################################################
    # Metadata and compact report
    ###########################################################################

    metadata = {
        "schema": "exp06_chemical_quality_v2",
        "manifest": str(manifest),
        "n_conditions": len(condition_rows),
        "n_requested_total": sum(
            row["requested"]
            for row in condition_rows
        ),
        "n_records_total": len(molecule_rows),
        "descriptor_unit": (
            "Largest connected fragment after removal of explicit "
            "hydrogens; implicit valence re-derived by RDKit."
        ),
        "fragmentation_unit": (
            "Full sanitized SDF record before parent-fragment selection."
        ),
        "hydrogen_convention": (
            "SDF read with removeHs=False. Explicit H atoms retained "
            "for raw record audit, then removed from the selected parent "
            "fragment before physicochemical descriptors."
        ),
    }

    (
        out_dir / "chemical_quality_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    print()
    print("CHEMICAL_QUALITY_V2_DONE")
    print("requested_total={}".format(
        metadata["n_requested_total"]
    ))
    print("records_total={}".format(
        metadata["n_records_total"]
    ))

    print()
    print("ACROSS-SEED CHEMICAL QUALITY")
    print(
        "pair\tlambda\tyield\tusable\tconnected\t"
        "unique\tQED\tSA\tLogP\tTPSA\t"
        "nHeavy_full\tnHeavy_parent\tparent_fraction\tRo5"
    )

    for row in across_rows:
        print(
            "{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t"
            "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t"
            "{:.1f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}".format(
                row["pair_id"],
                row["lambda_global"],
                float(row["output_yield_seedmean"]),
                float(row["usable_yield_seedmean"]),
                float(row["connected_rate_seedmean"]),
                float(row["uniqueness_rate_seedmean"]),
                float(row["qed_parent_seedmean"]),
                float(row["sa_score_parent_seedmean"]),
                float(row["logp_parent_seedmean"]),
                float(row["tpsa_parent_seedmean"]),
                float(row["n_heavy_full_seedmean"]),
                float(row["n_heavy_parent_seedmean"]),
                float(
                    row[
                        "parent_heavy_fraction_seedmean"
                    ]
                ),
                float(row["ro5_pass_rate_seedmean"]),
            )
        )

    print()
    print("out_dir={}".format(out_dir))


if __name__ == "__main__":
    main()
