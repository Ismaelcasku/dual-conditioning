#!/usr/bin/env python

import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


def read_first(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=False,
    )

    for molecule in supplier:
        if molecule is not None:
            return molecule

    raise RuntimeError(f"No readable molecule in {path}")


reference_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

reference = read_first(reference_path)
candidate = read_first(candidate_path)

reference_conf = reference.GetConformer()
candidate_conf = candidate.GetConformer()

reference_atoms = [
    atom for atom in reference.GetAtoms()
    if atom.GetAtomicNum() > 1
]

candidate_atoms = [
    atom for atom in candidate.GetAtoms()
    if atom.GetAtomicNum() > 1
]

reference_xyz = np.array([
    list(reference_conf.GetAtomPosition(atom.GetIdx()))
    for atom in reference_atoms
])

candidate_xyz = np.array([
    list(candidate_conf.GetAtomPosition(atom.GetIdx()))
    for atom in candidate_atoms
])

cost = np.full(
    (len(candidate_atoms), len(reference_atoms)),
    1.0e6,
    dtype=float,
)

for i, candidate_atom in enumerate(candidate_atoms):
    for j, reference_atom in enumerate(reference_atoms):
        if (
            candidate_atom.GetAtomicNum()
            == reference_atom.GetAtomicNum()
        ):
            cost[i, j] = np.linalg.norm(
                candidate_xyz[i] - reference_xyz[j]
            )

candidate_rows, reference_cols = linear_sum_assignment(cost)

matches = []

for row, col in zip(candidate_rows, reference_cols):
    candidate_atom = candidate_atoms[row]
    reference_atom = reference_atoms[col]

    matches.append({
        "candidate_index": candidate_atom.GetIdx(),
        "candidate_element": candidate_atom.GetSymbol(),
        "reference_index": reference_atom.GetIdx(),
        "reference_element": reference_atom.GetSymbol(),
        "distance_angstrom": float(cost[row, col]),
    })

matches.sort(key=lambda item: item["distance_angstrom"])

print(
    "candidate_index\telement\treference_index\t"
    "distance_A\tclassification"
)

for match in matches:
    classification = (
        "FIXED"
        if match["distance_angstrom"] <= 0.2
        else "FREE"
    )

    print(
        f'{match["candidate_index"]}\t'
        f'{match["candidate_element"]}\t'
        f'{match["reference_index"]}\t'
        f'{match["distance_angstrom"]:.6f}\t'
        f'{classification}'
    )

fixed = [
    match for match in matches
    if match["distance_angstrom"] <= 0.2
]

if len(fixed) != 7:
    raise SystemExit(
        "\nERROR: se esperaban exactamente 7 átomos "
        f"fijos dentro de 0.2 Å, pero se detectaron {len(fixed)}."
    )

fixed_candidate_indices = sorted(
    match["candidate_index"]
    for match in fixed
)

fixed_reference_indices = [
    match["reference_index"]
    for match in sorted(
        fixed,
        key=lambda item: item["candidate_index"],
    )
]

rmsd = float(np.sqrt(np.mean([
    match["distance_angstrom"] ** 2
    for match in fixed
])))

result = {
    "candidate": str(candidate_path),
    "reference": str(reference_path),
    "threshold_angstrom": 0.2,
    "fixed_candidate_indices": fixed_candidate_indices,
    "fixed_reference_indices": fixed_reference_indices,
    "fixed_atom_rmsd_angstrom": rmsd,
    "matches": matches,
}

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(result, indent=2)
)

print()
print(f"fixed_candidate_indices={fixed_candidate_indices}")
print(f"fixed_reference_indices={fixed_reference_indices}")
print(f"fixed_atom_rmsd_A={rmsd:.6f}")
print(f"output={output_path}")
print("FIXED_ATOM_IDENTIFICATION_STATUS=OK")
