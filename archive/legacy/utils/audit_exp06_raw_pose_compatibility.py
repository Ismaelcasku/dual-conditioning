#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

DIFFSBDD_ROOT = Path("codes/vendor/diffsbdd").resolve()
sys.path.insert(0, str(DIFFSBDD_ROOT))

from analysis.molecule_builder import process_molecule


RECEPTORS = {
    "5R83_x0434": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
        "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_protein.pdb"
    ),
    "5REZ_x0874": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
        "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_protein.pdb"
    ),
}

REFERENCE_LIGANDS = {
    "5R83_x0434": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
        "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_ligand.sdf"
    ),
    "5REZ_x0874": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
        "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_ligand.sdf"
    ),
}

PERIODIC_TABLE = Chem.GetPeriodicTable()


def is_true(value):
    return str(value).strip().upper() == "TRUE"


def safe_name(value):
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
    )


def element_from_pdb_line(line):
    element = line[76:78].strip().title()

    if element:
        return element

    atom_name = line[12:16].strip()
    letters = "".join(
        character
        for character in atom_name
        if character.isalpha()
    )

    if not letters:
        return ""

    if len(letters) >= 2:
        candidate = letters[:2].title()

        try:
            if PERIODIC_TABLE.GetAtomicNumber(candidate) > 0:
                return candidate
        except Exception:
            pass

    return letters[0].upper()


def read_protein_atoms(path):
    atoms = []

    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            residue_name = line[17:20].strip()

            if residue_name in {
                "HOH",
                "WAT",
                "DOD",
            }:
                continue

            element = element_from_pdb_line(line)

            if not element:
                continue

            try:
                atomic_number = PERIODIC_TABLE.GetAtomicNumber(
                    element
                )
            except Exception:
                continue

            if atomic_number <= 1:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append({
                "xyz": (x, y, z),
                "element": element,
                "atomic_number": atomic_number,
                "vdw_radius": float(
                    PERIODIC_TABLE.GetRvdw(atomic_number)
                ),
                "atom_name": line[12:16].strip(),
                "residue_name": residue_name,
                "chain": line[21].strip(),
                "residue_number": line[22:26].strip(),
            })

    if not atoms:
        raise RuntimeError(
            f"No protein heavy atoms read from {path}"
        )

    return atoms


def ligand_arrays(mol):
    conformer = mol.GetConformer()

    coordinates = []
    radii = []
    elements = []
    atom_indices = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue

        position = conformer.GetAtomPosition(
            atom.GetIdx()
        )

        coordinates.append((
            float(position.x),
            float(position.y),
            float(position.z),
        ))

        radii.append(float(
            PERIODIC_TABLE.GetRvdw(
                atom.GetAtomicNum()
            )
        ))

        elements.append(atom.GetSymbol())
        atom_indices.append(atom.GetIdx())

    return (
        np.asarray(coordinates, dtype=float),
        np.asarray(radii, dtype=float),
        elements,
        atom_indices,
    )


def calculate_pose_metrics(mol, protein_atoms):
    (
        ligand_xyz,
        ligand_radii,
        ligand_elements,
        ligand_indices,
    ) = ligand_arrays(mol)

    protein_xyz = np.asarray(
        [atom["xyz"] for atom in protein_atoms],
        dtype=float,
    )

    protein_radii = np.asarray(
        [
            atom["vdw_radius"]
            for atom in protein_atoms
        ],
        dtype=float,
    )

    displacement = (
        ligand_xyz[:, None, :]
        - protein_xyz[None, :, :]
    )

    distances = np.sqrt(
        np.sum(displacement * displacement, axis=2)
    )

    radius_sums = (
        ligand_radii[:, None]
        + protein_radii[None, :]
    )

    overlaps = radius_sums - distances

    moderate_mask = overlaps >= 0.40
    severe_mask = overlaps >= 0.80

    ligand_min_distances = distances.min(axis=1)
    nearest_protein_indices = distances.argmin(axis=1)

    severe_ligand_atoms = np.any(
        severe_mask,
        axis=1,
    )

    moderate_ligand_atoms = np.any(
        moderate_mask,
        axis=1,
    )

    contact_4 = distances <= 4.0
    contact_4p5 = distances <= 4.5
    contact_6 = distances <= 6.0

    residues_4 = set()
    residues_4p5 = set()

    contact_pairs = np.argwhere(contact_4)

    for _, protein_index in contact_pairs:
        atom = protein_atoms[int(protein_index)]

        residues_4.add((
            atom["chain"],
            atom["residue_name"],
            atom["residue_number"],
        ))

    contact_pairs_4p5 = np.argwhere(contact_4p5)

    for _, protein_index in contact_pairs_4p5:
        atom = protein_atoms[int(protein_index)]

        residues_4p5.add((
            atom["chain"],
            atom["residue_name"],
            atom["residue_number"],
        ))

    closest_ligand_index = int(
        np.argmin(ligand_min_distances)
    )

    closest_protein_index = int(
        nearest_protein_indices[
            closest_ligand_index
        ]
    )

    closest_protein_atom = protein_atoms[
        closest_protein_index
    ]

    maximum_overlap_flat = int(
        np.argmax(overlaps)
    )

    maximum_ligand_position, maximum_protein_position = (
        np.unravel_index(
            maximum_overlap_flat,
            overlaps.shape,
        )
    )

    maximum_protein_atom = protein_atoms[
        int(maximum_protein_position)
    ]

    return {
        "n_ligand_heavy_atoms":
            int(len(ligand_xyz)),
        "minimum_protein_ligand_distance":
            float(distances.min()),
        "maximum_vdw_overlap":
            float(overlaps.max()),
        "moderate_overlap_pairs":
            int(moderate_mask.sum()),
        "severe_overlap_pairs":
            int(severe_mask.sum()),
        "moderate_ligand_atoms":
            int(moderate_ligand_atoms.sum()),
        "severe_ligand_atoms":
            int(severe_ligand_atoms.sum()),
        "moderate_ligand_atom_fraction":
            float(moderate_ligand_atoms.mean()),
        "severe_ligand_atom_fraction":
            float(severe_ligand_atoms.mean()),
        "contact_pairs_4A":
            int(contact_4.sum()),
        "contact_pairs_4p5A":
            int(contact_4p5.sum()),
        "contact_pairs_6A":
            int(contact_6.sum()),
        "ligand_atoms_within_4A":
            int(np.any(contact_4, axis=1).sum()),
        "ligand_atoms_within_4p5A":
            int(np.any(contact_4p5, axis=1).sum()),
        "ligand_atoms_within_6A":
            int(np.any(contact_6, axis=1).sum()),
        "ligand_fraction_within_4A":
            float(np.any(contact_4, axis=1).mean()),
        "ligand_fraction_within_4p5A":
            float(np.any(contact_4p5, axis=1).mean()),
        "ligand_fraction_within_6A":
            float(np.any(contact_6, axis=1).mean()),
        "contacted_residues_4A":
            len(residues_4),
        "contacted_residues_4p5A":
            len(residues_4p5),
        "residue_ids_4A": ",".join(
            "{}:{}{}".format(
                chain if chain else "_",
                residue_name,
                residue_number,
            )
            for chain, residue_name, residue_number
            in sorted(residues_4)
        ),
        "residue_ids_4p5A": ",".join(
            "{}:{}{}".format(
                chain if chain else "_",
                residue_name,
                residue_number,
            )
            for chain, residue_name, residue_number
            in sorted(residues_4p5)
        ),
        "closest_ligand_atom_index":
            int(ligand_indices[
                closest_ligand_index
            ]),
        "closest_ligand_element":
            ligand_elements[
                closest_ligand_index
            ],
        "closest_protein_atom":
            closest_protein_atom["atom_name"],
        "closest_protein_residue": (
            "{}:{}{}".format(
                closest_protein_atom["chain"]
                if closest_protein_atom["chain"]
                else "_",
                closest_protein_atom[
                    "residue_name"
                ],
                closest_protein_atom[
                    "residue_number"
                ],
            )
        ),
        "maximum_overlap_ligand_atom_index":
            int(ligand_indices[
                int(maximum_ligand_position)
            ]),
        "maximum_overlap_ligand_element":
            ligand_elements[
                int(maximum_ligand_position)
            ],
        "maximum_overlap_protein_atom":
            maximum_protein_atom["atom_name"],
        "maximum_overlap_protein_residue": (
            "{}:{}{}".format(
                maximum_protein_atom["chain"]
                if maximum_protein_atom["chain"]
                else "_",
                maximum_protein_atom[
                    "residue_name"
                ],
                maximum_protein_atom[
                    "residue_number"
                ],
            )
        ),
    }


def load_reference_ligand(path):
    mol = Chem.MolFromMolFile(
        str(path),
        sanitize=False,
        removeHs=False,
    )

    if mol is None:
        raise RuntimeError(
            f"Could not read reference ligand: {path}"
        )

    Chem.SanitizeMol(mol)

    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            f"Reference ligand has no conformer: {path}"
        )

    return mol


def classify(metrics, reference):
    compatible_max_overlap = max(
        1.20,
        reference["maximum_vdw_overlap"] + 0.40,
    )

    compatible_severe_pairs = max(
        1,
        reference["severe_overlap_pairs"] + 1,
    )

    compatible_severe_fraction = max(
        0.10,
        reference[
            "severe_ligand_atom_fraction"
        ] + 0.05,
    )

    compatible_pocket_fraction = max(
        0.75,
        reference[
            "ligand_fraction_within_6A"
        ] - 0.15,
    )

    if (
        metrics["maximum_vdw_overlap"]
        <= compatible_max_overlap
        and metrics["severe_overlap_pairs"]
        <= compatible_severe_pairs
        and metrics[
            "severe_ligand_atom_fraction"
        ] <= compatible_severe_fraction
        and metrics[
            "ligand_fraction_within_6A"
        ] >= compatible_pocket_fraction
    ):
        return "compatible_raw"

    if (
        metrics["maximum_vdw_overlap"] <= 1.80
        and metrics[
            "severe_ligand_atom_fraction"
        ] <= 0.30
        and metrics[
            "ligand_fraction_within_6A"
        ] >= 0.70
    ):
        return "locally_recoverable"

    return "incompatible_raw"


def write_tsv(path, rows, fields=None):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:
        fields = list(rows[0].keys())

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open() as handle:
        labelled_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    grouped = defaultdict(list)

    for row in labelled_rows:
        grouped[
            row["compatibility_id"]
        ].append(row)

    if len(labelled_rows) != 105:
        raise RuntimeError(
            "Expected 105 pair-labelled rows; "
            f"found {len(labelled_rows)}"
        )

    if len(grouped) != 79:
        raise RuntimeError(
            "Expected 79 physical candidates; "
            f"found {len(grouped)}"
        )

    protein_atoms = {
        key: read_protein_atoms(path)
        for key, path in RECEPTORS.items()
    }

    reference_metrics = {}

    for receptor_key in RECEPTORS:
        reference_mol = load_reference_ligand(
            REFERENCE_LIGANDS[receptor_key]
        )

        metrics = calculate_pose_metrics(
            reference_mol,
            protein_atoms[receptor_key],
        )

        metrics.update({
            "receptor_key": receptor_key,
            "protein_path":
                str(RECEPTORS[receptor_key]),
            "reference_ligand_path":
                str(REFERENCE_LIGANDS[
                    receptor_key
                ]),
        })

        reference_metrics[receptor_key] = metrics

    physical_rows = []
    candidate_molecules = {}

    supplier_cache = {}

    for number, compatibility_id in enumerate(
        sorted(grouped),
        start=1,
    ):
        annotations = grouped[compatibility_id]
        representative = annotations[0]

        raw_sdf = representative["raw_sdf"]
        sample_index = int(
            representative["sample"]
        ) - 1

        if raw_sdf not in supplier_cache:
            supplier_cache[raw_sdf] = list(
                Chem.SDMolSupplier(
                    raw_sdf,
                    sanitize=False,
                    removeHs=False,
                )
            )

        records = supplier_cache[raw_sdf]

        if (
            sample_index < 0
            or sample_index >= len(records)
        ):
            raise RuntimeError(
                f"Invalid sample index for {compatibility_id}"
            )

        raw_mol = records[sample_index]

        if raw_mol is None:
            raise RuntimeError(
                f"Could not read {compatibility_id}"
            )

        parent = process_molecule(
            raw_mol,
            add_hydrogens=False,
            sanitize=True,
            relax_iter=0,
            largest_frag=True,
        )

        if parent is None:
            raise RuntimeError(
                "DiffSBDD native postprocessing failed for "
                f"{compatibility_id}"
            )

        Chem.SanitizeMol(parent)

        receptor_key = representative[
            "receptor_key"
        ]

        metrics = calculate_pose_metrics(
            parent,
            protein_atoms[receptor_key],
        )

        classification = classify(
            metrics,
            reference_metrics[receptor_key],
        )

        candidate_path = (
            out_dir
            / "candidates"
            / receptor_key
            / f"{safe_name(compatibility_id)}.sdf"
        )

        candidate_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_mol = Chem.Mol(parent)
        output_mol.SetProp(
            "_Name",
            compatibility_id,
        )

        output_mol.SetProp(
            "compatibility_id",
            compatibility_id,
        )

        output_mol.SetProp(
            "receptor_key",
            receptor_key,
        )

        output_mol.SetProp(
            "pair_labels",
            ",".join(sorted({
                row["pair_id"]
                for row in annotations
            })),
        )

        output_mol.SetProp(
            "raw_pose_class",
            classification,
        )

        writer = Chem.SDWriter(
            str(candidate_path)
        )
        writer.write(output_mol)
        writer.close()

        candidate_molecules[
            compatibility_id
        ] = output_mol

        physical_row = {
            "compatibility_id":
                compatibility_id,
            "pair_labels": ",".join(sorted({
                row["pair_id"]
                for row in annotations
            })),
            "A_local":
                representative["A_local"],
            "receptor_key": receptor_key,
            "seed": representative["seed"],
            "lambda_global":
                representative["lambda_global"],
            "sample": representative["sample"],
            "candidate_sdf":
                str(candidate_path),
            "raw_sdf": raw_sdf,
            "parent_smiles":
                representative["parent_smiles"],
            "parent_nheavy":
                representative["parent_nheavy"],
            "parent_heavy_fraction":
                representative[
                    "parent_heavy_fraction"
                ],
            "raw_pose_class": classification,
        }

        physical_row.update(metrics)
        physical_rows.append(physical_row)

        print(
            f"[{number}/79] {compatibility_id} "
            f"{classification}",
            flush=True,
        )

    physical_path = (
        out_dir
        / "raw_pose_compatibility_physical.tsv"
    )

    write_tsv(
        physical_path,
        physical_rows,
    )

    expanded_rows = []

    physical_index = {
        row["compatibility_id"]: row
        for row in physical_rows
    }

    for annotation in labelled_rows:
        physical = physical_index[
            annotation["compatibility_id"]
        ]

        expanded = dict(annotation)

        for key, value in physical.items():
            if key not in expanded:
                expanded[key] = value

        expanded_rows.append(expanded)

    expanded_path = (
        out_dir
        / "raw_pose_compatibility_pair_labelled.tsv"
    )

    write_tsv(
        expanded_path,
        expanded_rows,
    )

    reference_rows = [
        reference_metrics[key]
        for key in sorted(reference_metrics)
    ]

    write_tsv(
        out_dir / "reference_pose_calibration.tsv",
        reference_rows,
    )

    summary_rows = []

    summary_groups = defaultdict(list)

    for row in expanded_rows:
        summary_groups[
            (
                row["pair_id"],
                row["lambda_global"],
            )
        ].append(row)

    for key in sorted(
        summary_groups,
        key=lambda item: (
            item[0],
            float(item[1]),
        ),
    ):
        rows = summary_groups[key]
        classes = Counter(
            row["raw_pose_class"]
            for row in rows
        )

        summary_rows.append({
            "pair_id": key[0],
            "lambda_global": key[1],
            "n_candidates": len(rows),
            "compatible_raw":
                classes["compatible_raw"],
            "locally_recoverable":
                classes["locally_recoverable"],
            "incompatible_raw":
                classes["incompatible_raw"],
            "compatible_or_recoverable": (
                classes["compatible_raw"]
                + classes["locally_recoverable"]
            ),
            "conditional_compatibility_yield": (
                (
                    classes["compatible_raw"]
                    + classes["locally_recoverable"]
                ) / len(rows)
                if rows else 0.0
            ),
            "global_compatibility_yield": (
                (
                    classes["compatible_raw"]
                    + classes["locally_recoverable"]
                ) / 50.0
            ),
            "mean_maximum_vdw_overlap":
                float(np.mean([
                    float(row[
                        "maximum_vdw_overlap"
                    ])
                    for row in rows
                ])),
            "mean_severe_overlap_pairs":
                float(np.mean([
                    int(row[
                        "severe_overlap_pairs"
                    ])
                    for row in rows
                ])),
            "mean_severe_ligand_atom_fraction":
                float(np.mean([
                    float(row[
                        "severe_ligand_atom_fraction"
                    ])
                    for row in rows
                ])),
            "mean_ligand_fraction_within_6A":
                float(np.mean([
                    float(row[
                        "ligand_fraction_within_6A"
                    ])
                    for row in rows
                ])),
            "mean_contacted_residues_4p5A":
                float(np.mean([
                    int(row[
                        "contacted_residues_4p5A"
                    ])
                    for row in rows
                ])),
        })

    write_tsv(
        out_dir / "raw_pose_compatibility_summary.tsv",
        summary_rows,
    )

    combined_dir = out_dir / "combined_sdf"
    combined_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for receptor_key in sorted(RECEPTORS):
        path = (
            combined_dir
            / f"{receptor_key}_all_candidates.sdf"
        )

        writer = Chem.SDWriter(str(path))

        for row in physical_rows:
            if row["receptor_key"] != receptor_key:
                continue

            writer.write(
                candidate_molecules[
                    row["compatibility_id"]
                ]
            )

        writer.close()

    metadata = {
        "schema":
            "exp06_raw_pose_compatibility_v1",
        "n_pair_labelled_entries":
            len(labelled_rows),
        "n_physical_candidates":
            len(physical_rows),
        "protein_atoms": (
            "Protein heavy atoms from PDB; waters and "
            "hydrogens excluded."
        ),
        "ligand_atoms":
            "Parent heavy atoms only.",
        "moderate_overlap_threshold_angstrom":
            0.40,
        "severe_overlap_threshold_angstrom":
            0.80,
        "pose_modification": False,
        "uff_relaxation": False,
        "docking": False,
        "classification": (
            "Heuristic classification calibrated against "
            "the corresponding crystallographic ligand. "
            "It is intended for triage, not as a final "
            "energetic validation."
        ),
    }

    (
        out_dir / "raw_pose_compatibility_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("REFERENCE POSE CALIBRATION")
    print(
        "receptor\tnHeavy\tminDist\tmaxOverlap\t"
        "moderatePairs\tseverePairs\t"
        "severeAtomFraction\tfractionWithin6A\t"
        "residues4p5A"
    )

    for row in reference_rows:
        print(
            "{}\t{}\t{:.3f}\t{:.3f}\t{}\t{}\t"
            "{:.3f}\t{:.3f}\t{}".format(
                row["receptor_key"],
                row["n_ligand_heavy_atoms"],
                row[
                    "minimum_protein_ligand_distance"
                ],
                row["maximum_vdw_overlap"],
                row["moderate_overlap_pairs"],
                row["severe_overlap_pairs"],
                row[
                    "severe_ligand_atom_fraction"
                ],
                row[
                    "ligand_fraction_within_6A"
                ],
                row[
                    "contacted_residues_4p5A"
                ],
            )
        )

    print()
    print("RAW POSE COMPATIBILITY SUMMARY")
    print(
        "pair\tlambda\tn\tcompatible\trecoverable\t"
        "incompatible\tconditional_yield\tglobal_yield\t"
        "meanMaxOverlap\tmeanSevereFraction\t"
        "meanFractionWithin6A"
    )

    for row in summary_rows:
        print(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.3f}\t"
            "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                row["pair_id"],
                row["lambda_global"],
                row["n_candidates"],
                row["compatible_raw"],
                row["locally_recoverable"],
                row["incompatible_raw"],
                row[
                    "conditional_compatibility_yield"
                ],
                row[
                    "global_compatibility_yield"
                ],
                row[
                    "mean_maximum_vdw_overlap"
                ],
                row[
                    "mean_severe_ligand_atom_fraction"
                ],
                row[
                    "mean_ligand_fraction_within_6A"
                ],
            )
        )

    print()
    print(f"physical_candidates={len(physical_rows)}")
    print(f"pair_labelled_entries={len(expanded_rows)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
