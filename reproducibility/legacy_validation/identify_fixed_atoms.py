#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


def read_first_molecule(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=False,
    )

    for molecule in supplier:
        if molecule is not None:
            return molecule

    raise RuntimeError(
        f"No readable molecule found in {path}"
    )


def heavy_atoms_and_coordinates(molecule):
    conformer = molecule.GetConformer()

    atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    coordinates = np.asarray(
        [
            list(
                conformer.GetAtomPosition(
                    atom.GetIdx()
                )
            )
            for atom in atoms
        ],
        dtype=float,
    )

    return atoms, coordinates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True,
    )
    parser.add_argument(
        "--candidate",
        required=True,
    )
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    project = Path(args.project).resolve()
    candidate_path = Path(args.candidate).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    config = json.loads(
        config_path.read_text()
    )

    candidate = read_first_molecule(
        candidate_path
    )

    if not candidate.HasProp("A_local"):
        raise RuntimeError(
            "Candidate SDF does not contain the A_local property."
        )

    a_local = candidate.GetProp(
        "A_local"
    ).strip()

    local_conditions = config.get(
        "local_conditions",
        {},
    )

    if a_local not in local_conditions:
        raise RuntimeError(
            f"A_local={a_local!r} is not defined in {config_path}"
        )

    condition = local_conditions[a_local]

    expected_fixed_atoms = int(
        condition["expected_fixed_heavy_atoms"]
    )

    threshold = float(
        config["distance_threshold_angstrom"]
    )

    reference_path = (
        project
        / condition["reference_ligand"]
    ).resolve()

    protein_path = (
        project
        / condition["protein"]
    ).resolve()

    if not reference_path.is_file():
        raise FileNotFoundError(
            reference_path
        )

    if not protein_path.is_file():
        raise FileNotFoundError(
            protein_path
        )

    reference = read_first_molecule(
        reference_path
    )

    candidate_atoms, candidate_xyz = (
        heavy_atoms_and_coordinates(candidate)
    )

    reference_atoms, reference_xyz = (
        heavy_atoms_and_coordinates(reference)
    )

    cost = np.full(
        (
            len(candidate_atoms),
            len(reference_atoms),
        ),
        1.0e6,
        dtype=float,
    )

    for candidate_row, candidate_atom in enumerate(
        candidate_atoms
    ):
        for reference_col, reference_atom in enumerate(
            reference_atoms
        ):
            if (
                candidate_atom.GetAtomicNum()
                != reference_atom.GetAtomicNum()
            ):
                continue

            cost[
                candidate_row,
                reference_col,
            ] = np.linalg.norm(
                candidate_xyz[candidate_row]
                - reference_xyz[reference_col]
            )

    candidate_rows, reference_cols = (
        linear_sum_assignment(cost)
    )

    matches = []

    for candidate_row, reference_col in zip(
        candidate_rows,
        reference_cols,
    ):
        distance = float(
            cost[candidate_row, reference_col]
        )

        if distance >= 1.0e5:
            continue

        candidate_atom = candidate_atoms[
            candidate_row
        ]
        reference_atom = reference_atoms[
            reference_col
        ]

        matches.append(
            {
                "candidate_index":
                    candidate_atom.GetIdx(),
                "candidate_element":
                    candidate_atom.GetSymbol(),
                "reference_index":
                    reference_atom.GetIdx(),
                "reference_element":
                    reference_atom.GetSymbol(),
                "distance_angstrom":
                    distance,
            }
        )

    matches.sort(
        key=lambda record:
        record["distance_angstrom"]
    )

    fixed_matches = [
        record
        for record in matches
        if record["distance_angstrom"]
        <= threshold
    ]

    print(
        "candidate_index\telement\t"
        "reference_index\tdistance_A\tclassification"
    )

    for record in matches:
        classification = (
            "FIXED"
            if record in fixed_matches
            else "FREE"
        )

        print(
            f'{record["candidate_index"]}\t'
            f'{record["candidate_element"]}\t'
            f'{record["reference_index"]}\t'
            f'{record["distance_angstrom"]:.6f}\t'
            f'{classification}'
        )

    if len(fixed_matches) != expected_fixed_atoms:
        raise RuntimeError(
            "Unexpected number of fixed atoms: "
            f"detected={len(fixed_matches)}, "
            f"expected={expected_fixed_atoms}, "
            f"A_local={a_local}, "
            f"threshold={threshold} Å"
        )

    fixed_candidate_indices = sorted(
        record["candidate_index"]
        for record in fixed_matches
    )

    fixed_reference_indices = [
        record["reference_index"]
        for record in sorted(
            fixed_matches,
            key=lambda record:
            record["candidate_index"],
        )
    ]

    fixed_rmsd = float(
        np.sqrt(
            np.mean(
                [
                    record["distance_angstrom"] ** 2
                    for record in fixed_matches
                ]
            )
        )
    )

    result = {
        "experiment": config.get(
            "experiment",
            "",
        ),
        "candidate": str(
            candidate_path
        ),
        "A_local": a_local,
        "reference_ligand": str(
            reference_path
        ),
        "protein": str(
            protein_path
        ),
        "expected_fixed_atoms":
            expected_fixed_atoms,
        "detected_fixed_atoms":
            len(fixed_matches),
        "threshold_angstrom":
            threshold,
        "fixed_candidate_indices":
            fixed_candidate_indices,
        "fixed_reference_indices":
            fixed_reference_indices,
        "fixed_atom_rmsd_angstrom":
            fixed_rmsd,
        "matches": matches,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(f"A_local={a_local}")
    print(
        "expected_fixed_atoms="
        f"{expected_fixed_atoms}"
    )
    print(
        "detected_fixed_atoms="
        f"{len(fixed_matches)}"
    )
    print(
        "fixed_candidate_indices="
        f"{fixed_candidate_indices}"
    )
    print(
        f"fixed_atom_rmsd_A={fixed_rmsd:.6f}"
    )
    print(f"protein={protein_path}")
    print(f"output={output_path}")
    print(
        "FIXED_ATOM_IDENTIFICATION_STATUS=OK"
    )


if __name__ == "__main__":
    main()
