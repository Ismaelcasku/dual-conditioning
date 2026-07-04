#!/usr/bin/env python3

import argparse
import csv
import glob
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers


GRID_SPACING = 0.5
VDW_SCALE = 0.8
IGNORE_HS = True


def find_ligand_sdf(fragment_dir):
    candidates = sorted(
        glob.glob(os.path.join(fragment_dir, "*_ligand.sdf"))
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one *_ligand.sdf in {fragment_dir}; "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def load_molecule(sdf_path):
    supplier = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=IGNORE_HS,
        sanitize=True,
    )

    valid = [mol for mol in supplier if mol is not None]

    if not valid:
        raise RuntimeError(f"Could not sanitize/read ligand: {sdf_path}")

    mol = valid[0]

    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"Ligand has no 3D conformer: {sdf_path}")

    return mol


def heavy_atom_coordinates(mol):
    conformer = mol.GetConformer()

    coordinates = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue

        position = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append([position.x, position.y, position.z])

    if not coordinates:
        raise RuntimeError("Molecule contains no heavy atoms")

    return np.asarray(coordinates, dtype=float)


def heavy_atom_centroid(mol):
    return heavy_atom_coordinates(mol).mean(axis=0)


def heavy_atom_count(mol):
    return sum(
        atom.GetAtomicNum() > 1
        for atom in mol.GetAtoms()
    )


def shape_tanimoto(probe, reference):
    return float(
        rdShapeHelpers.ShapeTanimotoDist(
            probe,
            reference,
            gridSpacing=GRID_SPACING,
            vdwScale=VDW_SCALE,
            ignoreHs=IGNORE_HS,
        )
    )


def shape_protrude(probe, reference):
    return float(
        rdShapeHelpers.ShapeProtrudeDist(
            probe,
            reference,
            gridSpacing=GRID_SPACING,
            vdwScale=VDW_SCALE,
            ignoreHs=IGNORE_HS,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    ligands = {}

    for fragment_dir in sorted(root.iterdir()):
        if not fragment_dir.is_dir():
            continue

        tag = fragment_dir.name.split("__", 1)[0]

        try:
            sdf = find_ligand_sdf(fragment_dir)
            molecule = load_molecule(sdf)
        except Exception as exc:
            print(f"[ERROR] {tag}: {exc}", file=sys.stderr)
            continue

        ligands[tag] = {
            "mol": molecule,
            "sdf": str(Path(sdf).resolve()),
            "n_heavy": heavy_atom_count(molecule),
            "centroid": heavy_atom_centroid(molecule),
        }

        print(
            f"loaded {tag:8s} "
            f"heavy={ligands[tag]['n_heavy']:2d} "
            f"{Path(sdf).name}"
        )

    tags = sorted(ligands)

    if len(tags) < 2:
        raise SystemExit("Need at least two valid ligands")

    rows = []

    # Directed pairs: A is the local anchor, B is the global target.
    for tag_a, tag_b in itertools.permutations(tags, 2):
        data_a = ligands[tag_a]
        data_b = ligands[tag_b]

        mol_a = data_a["mol"]
        mol_b = data_b["mol"]

        tani_ab = shape_tanimoto(mol_a, mol_b)
        tani_ba = shape_tanimoto(mol_b, mol_a)

        protrude_ab = shape_protrude(mol_a, mol_b)
        protrude_ba = shape_protrude(mol_b, mol_a)

        centroid_distance = float(
            np.linalg.norm(
                data_a["centroid"] - data_b["centroid"]
            )
        )

        rows.append({
            "A_local": tag_a,
            "B_global": tag_b,
            "shapeTani_A_to_B": tani_ab,
            "shapeTani_B_to_A": tani_ba,
            "shapeTani_symmetric_mean": 0.5 * (tani_ab + tani_ba),
            "shapeTani_asymmetry": abs(tani_ab - tani_ba),
            "shapeProtrude_A_from_B": protrude_ab,
            "shapeProtrude_B_from_A": protrude_ba,
            "centroid_dist_AB_Angstrom": centroid_distance,
            "nHeavy_A": data_a["n_heavy"],
            "nHeavy_B": data_b["n_heavy"],
            "delta_nHeavy_B_minus_A":
                data_b["n_heavy"] - data_a["n_heavy"],
            "abs_delta_nHeavy":
                abs(data_b["n_heavy"] - data_a["n_heavy"]),
            "A_sdf": data_a["sdf"],
            "B_sdf": data_b["sdf"],
            "rdkit_version": rdkit.__version__,
        })

    # Primary operational difficulty:
    # A-like starting geometry compared in-frame against B.
    rows.sort(
        key=lambda row: (
            row["shapeTani_A_to_B"],
            row["shapeProtrude_A_from_B"],
        )
    )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"{'A local':8s} {'B global':8s} "
        f"{'TaniAB':>7s} {'ProtAB':>7s} {'ProtBA':>7s} "
        f"{'Centroid':>9s} {'hA':>3s} {'hB':>3s}"
    )
    print("-" * 68)

    for row in rows:
        print(
            f"{row['A_local']:8s} "
            f"{row['B_global']:8s} "
            f"{row['shapeTani_A_to_B']:7.3f} "
            f"{row['shapeProtrude_A_from_B']:7.3f} "
            f"{row['shapeProtrude_B_from_A']:7.3f} "
            f"{row['centroid_dist_AB_Angstrom']:9.2f} "
            f"{row['nHeavy_A']:3d} "
            f"{row['nHeavy_B']:3d}"
        )

    current = [
        row for row in rows
        if row["A_local"] == "x0874"
        and row["B_global"] == "x1093"
    ]

    if current:
        row = current[0]
        rank = rows.index(row) + 1

        print()
        print(
            "CURRENT DIRECTED PAIR "
            f"x0874 -> x1093: "
            f"TaniAB={row['shapeTani_A_to_B']:.3f}, "
            f"ProtAB={row['shapeProtrude_A_from_B']:.3f}, "
            f"ProtBA={row['shapeProtrude_B_from_A']:.3f}, "
            f"centroid={row['centroid_dist_AB_Angstrom']:.2f} Å, "
            f"rank={rank}/{len(rows)}"
        )

    print()
    print(f"Wrote {output}")
    print(f"Directed pairs: {len(rows)}")


if __name__ == "__main__":
    main()
