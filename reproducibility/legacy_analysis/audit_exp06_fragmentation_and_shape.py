#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdShapeHelpers


A_SDF = {
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

B_SDF = {
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

# Frozen zero-based RDKit indices established before target comparison.
FIXED_INDICES = {
    "x0434": [1, 2, 3, 7, 10, 12, 13],
    "x0874": [2, 3, 4, 5, 6, 7, 8],
}

EXPECTED_LAMBDAS = {"0.0", "20.0", "50.0", "100.0", "200.0"}

UNITS = ["full", "parent", "anchor"]

SHAPE_FIELDS = [
    "{}_tanimoto_A".format(unit)
    for unit in UNITS
] + [
    "{}_tanimoto_B".format(unit)
    for unit in UNITS
] + [
    "{}_protrude_A".format(unit)
    for unit in UNITS
] + [
    "{}_protrude_B".format(unit)
    for unit in UNITS
]


def bool_text(value):
    return "TRUE" if bool(value) else "FALSE"


def finite(value):
    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def describe(values):
    values = [
        float(value)
        for value in values
        if finite(value) is not None
    ]

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
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
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

                row[field] = value

            writer.writerow(row)


def load_molecule(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        sanitize=False,
        removeHs=False,
    )

    mol = supplier[0] if len(supplier) else None

    if mol is None:
        raise RuntimeError(
            "Could not read reference molecule: {}".format(path)
        )

    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)

    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            "Reference has no conformer: {}".format(path)
        )

    return mol


def sanitize_record(mol):
    if mol is None:
        return None, "READ_FAIL"

    copied = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(copied)
        return copied, "OK"
    except Exception:
        return copied, "SANITIZE_FAIL"


def atom_xyz(mol, atom_index):
    position = mol.GetConformer().GetAtomPosition(atom_index)

    return (
        float(position.x),
        float(position.y),
        float(position.z),
    )


def euclidean(first, second):
    return math.sqrt(sum(
        (a - b) ** 2
        for a, b in zip(first, second)
    ))


def fragment_data(mol):
    atom_fragments = list(
        Chem.GetMolFrags(
            mol,
            asMols=False,
            sanitizeFrags=False,
        )
    )

    fragment_molecules = list(
        Chem.GetMolFrags(
            mol,
            asMols=True,
            sanitizeFrags=False,
        )
    )

    if len(atom_fragments) != len(fragment_molecules):
        raise RuntimeError(
            "Fragment mapping and fragment molecules disagree"
        )

    heavy_counts = []

    for atom_indices in atom_fragments:
        heavy_counts.append(sum(
            mol.GetAtomWithIdx(index).GetAtomicNum() > 1
            for index in atom_indices
        ))

    heavy_fragment_ids = [
        fragment_id
        for fragment_id, heavy_count in enumerate(heavy_counts)
        if heavy_count > 0
    ]

    parent_id = max(
        heavy_fragment_ids,
        key=lambda fragment_id: (
            heavy_counts[fragment_id],
            len(atom_fragments[fragment_id]),
            -fragment_id,
        ),
    )

    atom_to_fragment = {}

    for fragment_id, atom_indices in enumerate(atom_fragments):
        for atom_index in atom_indices:
            atom_to_fragment[atom_index] = fragment_id

    return {
        "atom_fragments": atom_fragments,
        "fragment_molecules": fragment_molecules,
        "heavy_counts": heavy_counts,
        "heavy_fragment_ids": heavy_fragment_ids,
        "parent_id": parent_id,
        "atom_to_fragment": atom_to_fragment,
    }


def match_fixed_atoms(reference_a, generated, fixed_indices):
    reference_conf = reference_a.GetConformer()
    generated_conf = generated.GetConformer()

    candidate_pairs = []

    for fixed_position, reference_index in enumerate(fixed_indices):
        reference_atom = reference_a.GetAtomWithIdx(reference_index)
        reference_symbol = reference_atom.GetSymbol()

        reference_xyz = reference_conf.GetAtomPosition(reference_index)
        reference_xyz = (
            reference_xyz.x,
            reference_xyz.y,
            reference_xyz.z,
        )

        for generated_index in range(generated.GetNumAtoms()):
            generated_atom = generated.GetAtomWithIdx(generated_index)

            if generated_atom.GetSymbol() != reference_symbol:
                continue

            generated_xyz = generated_conf.GetAtomPosition(
                generated_index
            )

            generated_xyz = (
                generated_xyz.x,
                generated_xyz.y,
                generated_xyz.z,
            )

            candidate_pairs.append((
                euclidean(reference_xyz, generated_xyz),
                fixed_position,
                generated_index,
            ))

    candidate_pairs.sort()

    assigned_fixed = set()
    assigned_generated = set()
    assignments = []

    for distance, fixed_position, generated_index in candidate_pairs:
        if fixed_position in assigned_fixed:
            continue

        if generated_index in assigned_generated:
            continue

        assigned_fixed.add(fixed_position)
        assigned_generated.add(generated_index)

        assignments.append((
            fixed_position,
            generated_index,
            distance,
        ))

        if len(assignments) == len(fixed_indices):
            break

    assignments.sort()

    return assignments


def valence_headroom(atom):
    periodic_table = Chem.GetPeriodicTable()

    try:
        current_valence = float(atom.GetTotalValence())
    except Exception:
        current_valence = float(atom.GetExplicitValence())

    allowed = [
        int(value)
        for value in periodic_table.GetValenceList(
            atom.GetAtomicNum()
        )
        if int(value) >= 0
    ]

    if not allowed:
        return False

    return any(
        allowed_valence >= current_valence + 1
        for allowed_valence in allowed
    )


def interfragment_geometry(mol, fragments):
    heavy_fragment_ids = fragments["heavy_fragment_ids"]

    if len(heavy_fragment_ids) <= 1:
        return {
            "minimum_heavy_interfragment_distance": "",
            "minimum_covalent_radius_ratio": "",
            "closest_atom_1": "",
            "closest_atom_2": "",
            "closest_atom_1_symbol": "",
            "closest_atom_2_symbol": "",
            "closest_pair_both_valence_headroom": "",
            "interfragment_class": "single_heavy_component",
        }

    periodic_table = Chem.GetPeriodicTable()

    minimum_distance = None
    minimum_ratio = None
    ratio_pair = None

    for first_position, first_fragment_id in enumerate(
        heavy_fragment_ids
    ):
        first_atoms = fragments["atom_fragments"][
            first_fragment_id
        ]

        for second_fragment_id in heavy_fragment_ids[
            first_position + 1:
        ]:
            second_atoms = fragments["atom_fragments"][
                second_fragment_id
            ]

            for first_index in first_atoms:
                first_atom = mol.GetAtomWithIdx(first_index)

                if first_atom.GetAtomicNum() <= 1:
                    continue

                first_xyz = atom_xyz(mol, first_index)

                for second_index in second_atoms:
                    second_atom = mol.GetAtomWithIdx(second_index)

                    if second_atom.GetAtomicNum() <= 1:
                        continue

                    second_xyz = atom_xyz(mol, second_index)
                    distance = euclidean(first_xyz, second_xyz)

                    covalent_sum = (
                        periodic_table.GetRcovalent(
                            first_atom.GetAtomicNum()
                        )
                        + periodic_table.GetRcovalent(
                            second_atom.GetAtomicNum()
                        )
                    )

                    ratio = (
                        distance / covalent_sum
                        if covalent_sum > 0
                        else float("inf")
                    )

                    if (
                        minimum_distance is None
                        or distance < minimum_distance
                    ):
                        minimum_distance = distance

                    if (
                        minimum_ratio is None
                        or ratio < minimum_ratio
                    ):
                        minimum_ratio = ratio
                        ratio_pair = (
                            first_index,
                            second_index,
                            first_atom,
                            second_atom,
                        )

    first_index, second_index, first_atom, second_atom = ratio_pair

    both_headroom = (
        valence_headroom(first_atom)
        and valence_headroom(second_atom)
    )

    if minimum_ratio <= 1.25 and both_headroom:
        geometry_class = "potential_missing_bond"
    elif minimum_ratio <= 1.25:
        geometry_class = "bond_distance_valence_limited"
    elif minimum_ratio <= 1.75:
        geometry_class = "close_nonbonded"
    else:
        geometry_class = "geometrically_separated"

    return {
        "minimum_heavy_interfragment_distance":
            minimum_distance,
        "minimum_covalent_radius_ratio":
            minimum_ratio,
        "closest_atom_1": first_index,
        "closest_atom_2": second_index,
        "closest_atom_1_symbol": first_atom.GetSymbol(),
        "closest_atom_2_symbol": second_atom.GetSymbol(),
        "closest_pair_both_valence_headroom":
            both_headroom,
        "interfragment_class": geometry_class,
    }


def shape_values(generated, reference_a, reference_b):
    if generated is None or generated.GetNumConformers() == 0:
        return None

    try:
        tani_a = float(
            rdShapeHelpers.ShapeTanimotoDist(
                generated,
                reference_a,
                ignoreHs=True,
            )
        )

        tani_b = float(
            rdShapeHelpers.ShapeTanimotoDist(
                generated,
                reference_b,
                ignoreHs=True,
            )
        )

        protrude_a = float(
            rdShapeHelpers.ShapeProtrudeDist(
                generated,
                reference_a,
                ignoreHs=True,
            )
        )

        protrude_b = float(
            rdShapeHelpers.ShapeProtrudeDist(
                generated,
                reference_b,
                ignoreHs=True,
            )
        )

        return {
            "tanimoto_A": tani_a,
            "tanimoto_B": tani_b,
            "protrude_A": protrude_a,
            "protrude_B": protrude_b,
            "strict_dual": (
                tani_b < tani_a
                and protrude_b < protrude_a
            ),
        }

    except Exception:
        return None


def add_shape_fields(row, prefix, mol, reference_a, reference_b):
    values = shape_values(
        mol,
        reference_a,
        reference_b,
    )

    if values is None:
        row["{}_shape_ok".format(prefix)] = False

        for field in [
            "tanimoto_A",
            "tanimoto_B",
            "protrude_A",
            "protrude_B",
        ]:
            row[
                "{}_{}".format(prefix, field)
            ] = ""

        row[
            "{}_strict_dual".format(prefix)
        ] = False

        return

    row["{}_shape_ok".format(prefix)] = True

    for field in [
        "tanimoto_A",
        "tanimoto_B",
        "protrude_A",
        "protrude_B",
        "strict_dual",
    ]:
        row[
            "{}_{}".format(prefix, field)
        ] = values[field]


def preflight(design):
    if len(design) != 75:
        raise RuntimeError(
            "Expected 75 manifest rows; found {}".format(
                len(design)
            )
        )

    grouped = defaultdict(set)

    for row in design:
        grouped[
            (row["pair_id"], row["seed"])
        ].add(row["lambda_global"])

        path = Path(row["release_sdf_path"])

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                "Missing or empty SDF: {}".format(path)
            )

    for key, lambdas in grouped.items():
        if lambdas != EXPECTED_LAMBDAS:
            raise RuntimeError(
                "Incomplete design for {}: {}".format(
                    key,
                    sorted(lambdas),
                )
            )

    print("AUDIT_PREFLIGHT_OK")
    print("conditions={}".format(len(design)))


def condition_summary(condition, rows):
    auditable = [
        row for row in rows
        if row["audit_ok"]
    ]

    fragmented = [
        row for row in auditable
        if not row["heavy_connected"]
    ]

    anchor_same = [
        row for row in auditable
        if row["anchor_all_same_fragment"]
    ]

    connected = [
        row for row in auditable
        if row["heavy_connected"]
    ]

    result = {
        "pair_id": condition["pair_id"],
        "A_local": condition["A_local"],
        "B_global": condition["B_global"],
        "seed": condition["seed"],
        "lambda_global": condition["lambda_global"],
        "requested": int(condition["n_samples_requested"]),
        "n_records": len(rows),
        "n_auditable": len(auditable),
        "n_heavy_connected": len(connected),
        "heavy_connected_rate": (
            len(connected) / len(auditable)
            if auditable else 0.0
        ),
        "n_anchor_same_fragment": len(anchor_same),
        "anchor_same_fragment_rate": (
            len(anchor_same) / len(auditable)
            if auditable else 0.0
        ),
        "n_anchor_component_is_parent": sum(
            row["anchor_component_is_parent"]
            for row in anchor_same
        ),
        "anchor_component_is_parent_rate": (
            sum(
                row["anchor_component_is_parent"]
                for row in anchor_same
            ) / len(anchor_same)
            if anchor_same else 0.0
        ),
        "n_fragmented": len(fragmented),
        "n_potential_missing_bond": sum(
            row["interfragment_class"]
            == "potential_missing_bond"
            for row in fragmented
        ),
        "n_bond_distance_valence_limited": sum(
            row["interfragment_class"]
            == "bond_distance_valence_limited"
            for row in fragmented
        ),
        "n_close_nonbonded": sum(
            row["interfragment_class"]
            == "close_nonbonded"
            for row in fragmented
        ),
        "n_geometrically_separated": sum(
            row["interfragment_class"]
            == "geometrically_separated"
            for row in fragmented
        ),
    }

    for metric in [
        "n_heavy_fragments",
        "parent_heavy_fraction",
        "anchor_heavy_fraction",
        "minimum_heavy_interfragment_distance",
        "minimum_covalent_radius_ratio",
    ]:
        stats = describe([
            row.get(metric)
            for row in auditable
        ])

        for suffix, value in stats.items():
            result[
                "{}_{}".format(metric, suffix)
            ] = value

    for unit in UNITS:
        unit_rows = [
            row for row in auditable
            if row.get("{}_shape_ok".format(unit))
        ]

        for metric in [
            "tanimoto_A",
            "tanimoto_B",
            "protrude_A",
            "protrude_B",
        ]:
            stats = describe([
                row.get(
                    "{}_{}".format(unit, metric)
                )
                for row in unit_rows
            ])

            result[
                "{}_{}_mean".format(unit, metric)
            ] = stats["mean"]

            result[
                "{}_{}_sd".format(unit, metric)
            ] = stats["sd"]

        result[
            "{}_strict_dual".format(unit)
        ] = sum(
            row.get(
                "{}_strict_dual".format(unit),
                False,
            )
            for row in unit_rows
        )

    for metric in [
        "tanimoto_A",
        "tanimoto_B",
        "protrude_A",
        "protrude_B",
    ]:
        stats = describe([
            row.get("full_{}".format(metric))
            for row in connected
            if row.get("full_shape_ok")
        ])

        result[
            "connected_{}_mean".format(metric)
        ] = stats["mean"]

        result[
            "connected_{}_sd".format(metric)
        ] = stats["sd"]

    result["connected_strict_dual"] = sum(
        row.get("full_strict_dual", False)
        for row in connected
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open() as handle:
        design = list(
            csv.DictReader(handle, delimiter="\t")
        )

    preflight(design)

    references_a = {
        key: load_molecule(Path(path))
        for key, path in A_SDF.items()
    }

    references_b = {
        key: load_molecule(Path(path))
        for key, path in B_SDF.items()
    }

    molecule_rows = []
    condition_rows = []

    for condition_number, condition in enumerate(
        design,
        start=1,
    ):
        sdf_path = Path(condition["release_sdf_path"])

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=False,
            removeHs=False,
        )

        records = list(supplier)

        reference_a = references_a[
            condition["A_local"]
        ]

        reference_b = references_b[
            condition["B_global"]
        ]

        fixed_indices = FIXED_INDICES[
            condition["A_local"]
        ]

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
            full_mol, sanitize_status = sanitize_record(original)

            row = {
                "pair_id": condition["pair_id"],
                "A_local": condition["A_local"],
                "B_global": condition["B_global"],
                "seed": condition["seed"],
                "lambda_global": condition["lambda_global"],
                "experiment_id": condition["experiment_id"],
                "sample": sample,
                "sdf_path": str(sdf_path),
                "sanitize_status": sanitize_status,
                "audit_ok": False,
            }

            if (
                full_mol is None
                or sanitize_status != "OK"
                or full_mol.GetNumConformers() == 0
            ):
                molecule_rows.append(row)
                current_rows.append(row)
                continue

            fragments = fragment_data(full_mol)

            heavy_fragment_ids = fragments[
                "heavy_fragment_ids"
            ]

            parent_id = fragments["parent_id"]
            parent_mol = fragments[
                "fragment_molecules"
            ][parent_id]

            total_heavy = full_mol.GetNumHeavyAtoms()
            parent_heavy = fragments[
                "heavy_counts"
            ][parent_id]

            assignments = match_fixed_atoms(
                reference_a,
                full_mol,
                fixed_indices,
            )

            matched_generated_indices = [
                generated_index
                for _, generated_index, _ in assignments
            ]

            assignment_distances = [
                distance
                for _, _, distance in assignments
            ]

            matched_fragment_ids = [
                fragments["atom_to_fragment"][
                    generated_index
                ]
                for generated_index
                in matched_generated_indices
            ]

            anchor_all_matched = (
                len(assignments) == len(fixed_indices)
            )

            anchor_all_same_fragment = (
                anchor_all_matched
                and len(set(matched_fragment_ids)) == 1
            )

            anchor_id = (
                matched_fragment_ids[0]
                if anchor_all_same_fragment
                else None
            )

            anchor_mol = (
                fragments["fragment_molecules"][anchor_id]
                if anchor_id is not None
                else None
            )

            anchor_heavy = (
                fragments["heavy_counts"][anchor_id]
                if anchor_id is not None
                else 0
            )

            row.update({
                "audit_ok": True,
                "n_atoms_full": full_mol.GetNumAtoms(),
                "n_heavy_full": total_heavy,
                "n_total_fragments": len(
                    fragments["atom_fragments"]
                ),
                "n_heavy_fragments": len(
                    heavy_fragment_ids
                ),
                "heavy_connected": (
                    len(heavy_fragment_ids) == 1
                ),
                "parent_fragment_id": parent_id,
                "parent_heavy_atoms": parent_heavy,
                "parent_heavy_fraction": (
                    parent_heavy / total_heavy
                    if total_heavy else 0.0
                ),
                "fixed_atoms_expected": len(fixed_indices),
                "fixed_atoms_matched": len(assignments),
                "fixed_atoms_within_0p2A": sum(
                    distance <= 0.2
                    for distance in assignment_distances
                ),
                "fixed_match_max_distance": (
                    max(assignment_distances)
                    if assignment_distances else ""
                ),
                "anchor_all_same_fragment":
                    anchor_all_same_fragment,
                "anchor_fragment_id": (
                    anchor_id
                    if anchor_id is not None
                    else ""
                ),
                "anchor_heavy_atoms": anchor_heavy,
                "anchor_heavy_fraction": (
                    anchor_heavy / total_heavy
                    if total_heavy else 0.0
                ),
                "anchor_component_is_parent": (
                    anchor_id == parent_id
                    if anchor_id is not None
                    else False
                ),
            })

            row.update(
                interfragment_geometry(
                    full_mol,
                    fragments,
                )
            )

            add_shape_fields(
                row,
                "full",
                full_mol,
                reference_a,
                reference_b,
            )

            add_shape_fields(
                row,
                "parent",
                parent_mol,
                reference_a,
                reference_b,
            )

            add_shape_fields(
                row,
                "anchor",
                anchor_mol,
                reference_a,
                reference_b,
            )

            molecule_rows.append(row)
            current_rows.append(row)

        condition_rows.append(
            condition_summary(
                condition,
                current_rows,
            )
        )

    molecule_fields = [
        "pair_id",
        "A_local",
        "B_global",
        "seed",
        "lambda_global",
        "experiment_id",
        "sample",
        "sdf_path",
        "sanitize_status",
        "audit_ok",
        "n_atoms_full",
        "n_heavy_full",
        "n_total_fragments",
        "n_heavy_fragments",
        "heavy_connected",
        "parent_fragment_id",
        "parent_heavy_atoms",
        "parent_heavy_fraction",
        "fixed_atoms_expected",
        "fixed_atoms_matched",
        "fixed_atoms_within_0p2A",
        "fixed_match_max_distance",
        "anchor_all_same_fragment",
        "anchor_fragment_id",
        "anchor_heavy_atoms",
        "anchor_heavy_fraction",
        "anchor_component_is_parent",
        "minimum_heavy_interfragment_distance",
        "minimum_covalent_radius_ratio",
        "closest_atom_1",
        "closest_atom_2",
        "closest_atom_1_symbol",
        "closest_atom_2_symbol",
        "closest_pair_both_valence_headroom",
        "interfragment_class",
        "full_shape_ok",
        "full_tanimoto_A",
        "full_tanimoto_B",
        "full_protrude_A",
        "full_protrude_B",
        "full_strict_dual",
        "parent_shape_ok",
        "parent_tanimoto_A",
        "parent_tanimoto_B",
        "parent_protrude_A",
        "parent_protrude_B",
        "parent_strict_dual",
        "anchor_shape_ok",
        "anchor_tanimoto_A",
        "anchor_tanimoto_B",
        "anchor_protrude_A",
        "anchor_protrude_B",
        "anchor_strict_dual",
    ]

    write_tsv(
        out_dir / "fragment_audit_per_molecule.tsv",
        molecule_rows,
        molecule_fields,
    )

    write_tsv(
        out_dir / "fragment_audit_condition_summary.tsv",
        condition_rows,
        list(condition_rows[0].keys()),
    )

    ###########################################################################
    # Across-seed aggregation
    ###########################################################################

    grouped = defaultdict(list)

    for row in condition_rows:
        grouped[
            (row["pair_id"], row["lambda_global"])
        ].append(row)

    across_rows = []

    rate_fields = [
        "heavy_connected_rate",
        "anchor_same_fragment_rate",
        "anchor_component_is_parent_rate",
    ]

    mean_fields = [
        "n_heavy_fragments_mean",
        "parent_heavy_fraction_mean",
        "anchor_heavy_fraction_mean",
        "minimum_heavy_interfragment_distance_mean",
        "minimum_covalent_radius_ratio_mean",
        "full_tanimoto_A_mean",
        "full_tanimoto_B_mean",
        "full_protrude_A_mean",
        "full_protrude_B_mean",
        "parent_tanimoto_A_mean",
        "parent_tanimoto_B_mean",
        "parent_protrude_A_mean",
        "parent_protrude_B_mean",
        "anchor_tanimoto_A_mean",
        "anchor_tanimoto_B_mean",
        "anchor_protrude_A_mean",
        "anchor_protrude_B_mean",
        "connected_tanimoto_A_mean",
        "connected_tanimoto_B_mean",
        "connected_protrude_A_mean",
        "connected_protrude_B_mean",
    ]

    for key in sorted(
        grouped,
        key=lambda item: (
            item[0],
            float(item[1]),
        ),
    ):
        rows = grouped[key]

        result = {
            "pair_id": key[0],
            "lambda_global": key[1],
            "n_seeds": len(rows),
            "records_total": sum(
                row["n_records"]
                for row in rows
            ),
            "auditable_total": sum(
                row["n_auditable"]
                for row in rows
            ),
            "heavy_connected_total": sum(
                row["n_heavy_connected"]
                for row in rows
            ),
            "fragmented_total": sum(
                row["n_fragmented"]
                for row in rows
            ),
            "potential_missing_bond_total": sum(
                row["n_potential_missing_bond"]
                for row in rows
            ),
            "bond_distance_valence_limited_total": sum(
                row["n_bond_distance_valence_limited"]
                for row in rows
            ),
            "close_nonbonded_total": sum(
                row["n_close_nonbonded"]
                for row in rows
            ),
            "geometrically_separated_total": sum(
                row["n_geometrically_separated"]
                for row in rows
            ),
            "full_strict_dual_total": sum(
                row["full_strict_dual"]
                for row in rows
            ),
            "parent_strict_dual_total": sum(
                row["parent_strict_dual"]
                for row in rows
            ),
            "anchor_strict_dual_total": sum(
                row["anchor_strict_dual"]
                for row in rows
            ),
            "connected_strict_dual_total": sum(
                row["connected_strict_dual"]
                for row in rows
            ),
        }

        for field in rate_fields + mean_fields:
            stats = describe([
                row.get(field)
                for row in rows
            ])

            result[
                "{}_seedmean".format(field)
            ] = stats["mean"]

            result[
                "{}_seedsd".format(field)
            ] = stats["sd"]

        fragmented_total = result["fragmented_total"]

        result["potential_missing_bond_fraction"] = (
            result["potential_missing_bond_total"]
            / fragmented_total
            if fragmented_total else 0.0
        )

        result["geometrically_separated_fraction"] = (
            result["geometrically_separated_total"]
            / fragmented_total
            if fragmented_total else 0.0
        )

        across_rows.append(result)

    write_tsv(
        out_dir / "fragment_audit_across_seed_summary.tsv",
        across_rows,
        list(across_rows[0].keys()),
    )

    ###########################################################################
    # Candidate subsets
    ###########################################################################

    repair_candidates = [
        row for row in molecule_rows
        if (
            row.get("audit_ok")
            and row.get("interfragment_class")
            in {
                "potential_missing_bond",
                "bond_distance_valence_limited",
            }
        )
    ]

    connected_strict = [
        row for row in molecule_rows
        if (
            row.get("audit_ok")
            and row.get("heavy_connected")
            and row.get("full_strict_dual")
        )
    ]

    anchor_strict = [
        row for row in molecule_rows
        if (
            row.get("audit_ok")
            and row.get("anchor_strict_dual")
        )
    ]

    write_tsv(
        out_dir / "bond_distance_compatible_candidates.tsv",
        repair_candidates,
        molecule_fields,
    )

    write_tsv(
        out_dir / "connected_strict_dual_candidates.tsv",
        connected_strict,
        molecule_fields,
    )

    write_tsv(
        out_dir / "anchor_component_strict_dual_candidates.tsv",
        anchor_strict,
        molecule_fields,
    )

    metadata = {
        "schema": "exp06_fragmentation_shape_audit_v1",
        "manifest": str(manifest),
        "connectivity_definition": (
            "A record is connected when all heavy atoms belong "
            "to one RDKit connected component. Hydrogen-only "
            "components are ignored."
        ),
        "shape_convention": (
            "RDKit ShapeTanimotoDist and ShapeProtrudeDist in the "
            "original pocket coordinate frame with ignoreHs=True."
        ),
        "interfragment_classification": {
            "potential_missing_bond": (
                "Minimum covalent-radius ratio <=1.25 and both "
                "closest atoms have estimated valence headroom."
            ),
            "bond_distance_valence_limited": (
                "Minimum covalent-radius ratio <=1.25 but at least "
                "one closest atom appears valence-saturated."
            ),
            "close_nonbonded": (
                "Minimum covalent-radius ratio >1.25 and <=1.75."
            ),
            "geometrically_separated": (
                "Minimum covalent-radius ratio >1.75."
            ),
        },
        "important_limitation": (
            "Bond-distance compatibility is a diagnostic heuristic, "
            "not automatic chemical repair."
        ),
    }

    (
        out_dir / "fragment_audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("FRAGMENT_AUDIT_DONE")

    print()
    print("FRAGMENTATION GEOMETRY")
    print(
        "pair\tlambda\trecords\theavy_connected\t"
        "anchor_same\tanchor_is_parent\t"
        "parent_fraction\tmin_interfrag_A\t"
        "missing_bond_frac\tseparated_frac"
    )

    for row in across_rows:
        print(
            "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t"
            "{:.3f}\t{}\t{:.3f}\t{:.3f}".format(
                row["pair_id"],
                row["lambda_global"],
                row["records_total"],
                float(
                    row[
                        "heavy_connected_rate_seedmean"
                    ]
                ),
                float(
                    row[
                        "anchor_same_fragment_rate_seedmean"
                    ]
                ),
                float(
                    row[
                        "anchor_component_is_parent_rate_seedmean"
                    ]
                ),
                float(
                    row[
                        "parent_heavy_fraction_mean_seedmean"
                    ]
                ),
                (
                    "{:.3f}".format(float(
                        row[
                            "minimum_heavy_interfragment_"
                            "distance_mean_seedmean"
                        ]
                    ))
                    if row[
                        "minimum_heavy_interfragment_"
                        "distance_mean_seedmean"
                    ] != ""
                    else "NA"
                ),
                float(
                    row[
                        "potential_missing_bond_fraction"
                    ]
                ),
                float(
                    row[
                        "geometrically_separated_fraction"
                    ]
                ),
            )
        )

    print()
    print("STEERING BY STRUCTURAL UNIT")
    print(
        "pair\tlambda\tfull_TaniB\tparent_TaniB\t"
        "anchor_TaniB\tconnected_TaniB\t"
        "full_ProtB\tparent_ProtB\tanchor_ProtB\t"
        "connected_ProtB\tstrict_full\tstrict_parent\t"
        "strict_anchor\tstrict_connected"
    )

    for row in across_rows:
        def formatted(field):
            value = row.get(field, "")

            if value == "":
                return "NA"

            return "{:.3f}".format(float(value))

        print(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
            "{}\t{}\t{}\t{}".format(
                row["pair_id"],
                row["lambda_global"],
                formatted(
                    "full_tanimoto_B_mean_seedmean"
                ),
                formatted(
                    "parent_tanimoto_B_mean_seedmean"
                ),
                formatted(
                    "anchor_tanimoto_B_mean_seedmean"
                ),
                formatted(
                    "connected_tanimoto_B_mean_seedmean"
                ),
                formatted(
                    "full_protrude_B_mean_seedmean"
                ),
                formatted(
                    "parent_protrude_B_mean_seedmean"
                ),
                formatted(
                    "anchor_protrude_B_mean_seedmean"
                ),
                formatted(
                    "connected_protrude_B_mean_seedmean"
                ),
                row["full_strict_dual_total"],
                row["parent_strict_dual_total"],
                row["anchor_strict_dual_total"],
                row["connected_strict_dual_total"],
            )
        )

    print()
    print(
        "bond_distance_candidates={}".format(
            len(repair_candidates)
        )
    )

    print(
        "connected_strict_dual_candidates={}".format(
            len(connected_strict)
        )
    )

    print(
        "anchor_strict_dual_candidates={}".format(
            len(anchor_strict)
        )
    )

    print("out_dir={}".format(out_dir))


if __name__ == "__main__":
    main()
