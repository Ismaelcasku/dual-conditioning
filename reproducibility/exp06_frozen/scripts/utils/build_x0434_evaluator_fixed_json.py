#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


def load_first(path, remove_hs):
    supplier = Chem.SDMolSupplier(
        str(path),
        sanitize=False,
        removeHs=remove_hs,
    )

    for mol in supplier:
        if mol is not None:
            return mol

    raise RuntimeError("Could not read molecule: {}".format(path))


def coords_and_symbols(mol, indices):
    conf = mol.GetConformer()
    coords = []
    symbols = []

    for idx in indices:
        atom = mol.GetAtomWithIdx(int(idx))
        pos = conf.GetAtomPosition(int(idx))

        coords.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())

    return np.asarray(coords, dtype=float), symbols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A_sdf", required=True)
    parser.add_argument("--blind_json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.blind_json) as handle:
        blind = json.load(handle)

    # Same representation used to define the original mask.
    mol_no_h = load_first(args.A_sdf, remove_hs=True)

    # Representation used by phase0_official_evaluator.py.
    mol_full = load_first(args.A_sdf, remove_hs=False)

    source_indices = [
        int(x) for x in blind["rdkit_indices_0based"]
    ]

    source_coords, source_symbols = coords_and_symbols(
        mol_no_h,
        source_indices,
    )

    full_heavy_indices = [
        atom.GetIdx()
        for atom in mol_full.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    full_coords, full_symbols = coords_and_symbols(
        mol_full,
        full_heavy_indices,
    )

    distances = np.linalg.norm(
        source_coords[:, None, :] - full_coords[None, :, :],
        axis=2,
    )

    cost = distances.copy()

    for i, source_symbol in enumerate(source_symbols):
        for j, full_symbol in enumerate(full_symbols):
            if source_symbol != full_symbol:
                cost[i, j] = 1.0e9

    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = {}

    for row, col in zip(row_ind, col_ind):
        assignment[int(row)] = int(col)

    if len(assignment) != len(source_indices):
        raise RuntimeError("Incomplete source-to-full atom mapping")

    evaluator_indices = []
    mapping_distances = []

    for source_position in range(len(source_indices)):
        full_position = assignment[source_position]
        distance = float(distances[source_position, full_position])

        if distance > 0.05:
            raise RuntimeError(
                "Unexpected coordinate mismatch for anchor atom {}: {:.6f} A".format(
                    source_position,
                    distance,
                )
            )

        evaluator_indices.append(
            int(full_heavy_indices[full_position])
        )

        mapping_distances.append(distance)

    evaluator_symbols = [
        mol_full.GetAtomWithIdx(idx).GetSymbol()
        for idx in evaluator_indices
    ]

    expected_symbols = list(blind["elements"])

    if evaluator_symbols != expected_symbols:
        raise RuntimeError(
            "Element mismatch after mapping: {} versus {}".format(
                evaluator_symbols,
                expected_symbols,
            )
        )

    out = {
        "schema": "phase0_official_evaluator_fixed_atoms_v2",
        "anchor_tag": blind["anchor_tag"],
        "selection_blind_to_targets":
            blind["selection_blind_to_targets"],
        "selection_rule": blind["selection_rule"],

        # Keys required by phase0_official_evaluator.py
        "fixed_atom_indices_0based": evaluator_indices,
        "fixed_pdb_atom_names": blind["pdb_atom_names"],

        # Audit information
        "fixed_atom_symbols": evaluator_symbols,
        "source_removeHs_true_indices_0based": source_indices,
        "mapping_distances_A": mapping_distances,
        "maximum_mapping_distance_A": max(mapping_distances),
        "A_sdf": args.A_sdf,
        "n_fixed": len(evaluator_indices),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))

    print("EVALUATOR_FIXED_JSON_DONE")
    print(
        "source_indices_removeHs_true={}".format(
            " ".join(str(x) for x in source_indices)
        )
    )
    print(
        "evaluator_indices_removeHs_false={}".format(
            " ".join(str(x) for x in evaluator_indices)
        )
    )
    print(
        "pdb_atom_names={}".format(
            " ".join(blind["pdb_atom_names"])
        )
    )
    print(
        "maximum_mapping_distance_A={:.8f}".format(
            max(mapping_distances)
        )
    )
    print("out={}".format(output))


if __name__ == "__main__":
    main()
