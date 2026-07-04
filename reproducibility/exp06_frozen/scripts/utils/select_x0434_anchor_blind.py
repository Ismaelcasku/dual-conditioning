#!/usr/bin/env python3

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


ROOT = Path("data/mpro/prepared/silvr_xchem_hits")
OUTDIR = Path("artifacts/fixed_atom_screen/x0434_blind")
OUTDIR.mkdir(parents=True, exist_ok=True)

TAGS = {
    "A": "x0434",
    "B_easy": "x1093",
    "B_hard": "x2193",
}


def find_fragment_dir(tag):
    matches = sorted(ROOT.glob(tag + "__*"))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one directory for {}; found {}".format(
                tag, matches
            )
        )
    return matches[0]


def find_one(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one {} in {}; found {}".format(
                pattern, directory, matches
            )
        )
    return matches[0]


def load_sdf(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=True,
        sanitize=True,
    )

    molecules = [mol for mol in supplier if mol is not None]

    if not molecules:
        raise RuntimeError("Could not load {}".format(path))

    mol = molecules[0]

    if mol.GetNumConformers() == 0:
        raise RuntimeError("No conformer in {}".format(path))

    return mol


def heavy_indices(mol):
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]


def atom_coordinates(mol, indices):
    conf = mol.GetConformer()
    coords = []

    for idx in indices:
        p = conf.GetAtomPosition(idx)
        coords.append([p.x, p.y, p.z])

    return np.asarray(coords, dtype=float)


def element_from_pdb_line(line):
    element = line[76:78].strip()

    if element:
        return element.capitalize()

    atom_name = line[12:16].strip()
    letters = "".join(c for c in atom_name if c.isalpha())

    if letters[:2].upper() in {"CL", "BR"}:
        return letters[:2].capitalize()

    return letters[:1].upper()


def parse_pdb_ligands(path):
    residues = defaultdict(list)

    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue

            resname = line[17:20].strip()

            if resname in {"HOH", "WAT", "DOD"}:
                continue

            element = element_from_pdb_line(line)

            if element.upper() == "H":
                continue

            try:
                coord = np.asarray([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue

            key = (
                line[21].strip() or "_",
                line[22:26].strip(),
                line[26].strip(),
                resname,
            )

            residues[key].append({
                "name": line[12:16].strip(),
                "element": element,
                "coord": coord,
            })

    return residues


def select_pdb_residue(mol, complex_pdb):
    heavy = heavy_indices(mol)
    sdf_coords = atom_coordinates(mol, heavy)
    sdf_centroid = sdf_coords.mean(axis=0)

    candidates = []

    for key, atoms in parse_pdb_ligands(complex_pdb).items():
        pdb_coords = np.asarray([a["coord"] for a in atoms])

        score = (
            100.0 * abs(len(atoms) - len(heavy))
            + float(
                np.linalg.norm(
                    pdb_coords.mean(axis=0) - sdf_centroid
                )
            )
        )

        candidates.append((score, key, atoms))

    if not candidates:
        raise RuntimeError("No ligand-like PDB residue found")

    candidates.sort(key=lambda x: x[0])
    _, key, atoms = candidates[0]

    if len(atoms) != len(heavy):
        raise RuntimeError(
            "PDB/SDF heavy-atom count mismatch: {} versus {}".format(
                len(atoms), len(heavy)
            )
        )

    return key, atoms


def map_sdf_to_pdb(mol, pdb_atoms):
    conf = mol.GetConformer()
    pairs = []

    for idx in heavy_indices(mol):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol().capitalize()
        p = conf.GetAtomPosition(idx)
        coord = np.asarray([p.x, p.y, p.z])

        for j, pdb_atom in enumerate(pdb_atoms):
            if pdb_atom["element"].capitalize() != symbol:
                continue

            distance = float(
                np.linalg.norm(coord - pdb_atom["coord"])
            )

            pairs.append((distance, idx, j))

    pairs.sort()

    used_sdf = set()
    used_pdb = set()
    mapping = {}

    for distance, idx, j in pairs:
        if idx in used_sdf or j in used_pdb:
            continue

        used_sdf.add(idx)
        used_pdb.add(j)

        mapping[idx] = {
            "pdb_name": pdb_atoms[j]["name"],
            "element": pdb_atoms[j]["element"],
            "mapping_distance": distance,
        }

    expected = len(heavy_indices(mol))

    if len(mapping) != expected:
        raise RuntimeError(
            "Mapped only {}/{} heavy atoms".format(
                len(mapping), expected
            )
        )

    return mapping


def ring_systems(mol):
    """
    Merge rings sharing at least one atom into ring systems.
    Selection uses A only.
    """
    rings = [
        set(int(idx) for idx in ring)
        for ring in Chem.GetSymmSSSR(mol)
    ]

    systems = []

    for ring in rings:
        merged = set(ring)
        remaining = []

        changed = True
        while changed:
            changed = False
            remaining = []

            for system in systems:
                if merged.intersection(system):
                    merged.update(system)
                    changed = True
                else:
                    remaining.append(system)

            systems = remaining

        systems.append(merged)

    # Defensive second merge.
    changed = True
    while changed:
        changed = False
        new_systems = []

        while systems:
            current = systems.pop()
            merged_any = False

            for i, other in enumerate(systems):
                if current.intersection(other):
                    systems[i] = current.union(other)
                    merged_any = True
                    changed = True
                    break

            if not merged_any:
                new_systems.append(current)

        systems = new_systems

    return systems


def boundary_bonds(mol, subset):
    subset = set(subset)
    count = 0

    for idx in subset:
        atom = mol.GetAtomWithIdx(idx)

        for neighbour in atom.GetNeighbors():
            j = neighbour.GetIdx()

            if neighbour.GetAtomicNum() > 1 and j not in subset:
                count += 1

    return count


def aromatic_fraction(mol, subset):
    return float(
        np.mean([
            mol.GetAtomWithIdx(idx).GetIsAromatic()
            for idx in subset
        ])
    )


def choose_primary_ring_system(mol):
    systems = ring_systems(mol)

    if not systems:
        raise RuntimeError(
            "x0434 has no ring system; define another A-only rule"
        )

    ranked = []

    for system in systems:
        ranked.append({
            "atoms": set(system),
            "size": len(system),
            "aromatic_fraction": aromatic_fraction(mol, system),
            "boundary_bonds": boundary_bonds(mol, system),
            "minimum_index": min(system),
        })

    # All ranking variables depend exclusively on A.
    ranked.sort(
        key=lambda r: (
            -r["size"],
            -r["aromatic_fraction"],
            r["boundary_bonds"],
            r["minimum_index"],
        )
    )

    return ranked[0], ranked


def one_shell_heavy_neighbours(mol, core):
    anchor = set(core)

    for idx in list(core):
        atom = mol.GetAtomWithIdx(idx)

        for neighbour in atom.GetNeighbors():
            if neighbour.GetAtomicNum() > 1:
                anchor.add(neighbour.GetIdx())

    return anchor


def nearest_metrics(source_coords, target_coords):
    delta = (
        source_coords[:, None, :]
        - target_coords[None, :, :]
    )

    distances = np.sqrt(np.sum(delta * delta, axis=2))
    nearest = distances.min(axis=1)

    return {
        "mean_nearest_A": float(nearest.mean()),
        "median_nearest_A": float(np.median(nearest)),
        "max_nearest_A": float(nearest.max()),
        "fraction_within_1p7A": float(np.mean(nearest <= 1.7)),
        "fraction_within_2p5A": float(np.mean(nearest <= 2.5)),
    }


def make_2d_figure(mol, core, anchor, output):
    mol_core = Chem.Mol(mol)
    mol_anchor = Chem.Mol(mol)

    rdDepictor.Compute2DCoords(mol_core)
    rdDepictor.Compute2DCoords(mol_anchor)

    image = Draw.MolsToGridImage(
        [mol_core, mol_anchor],
        molsPerRow=2,
        subImgSize=(600, 450),
        legends=[
            "Primary ring system",
            "Frozen anchor: ring system + one heavy-atom shell",
        ],
        highlightAtomLists=[
            sorted(core),
            sorted(anchor),
        ],
    )

    image.save(str(output))


def main():
    fragment_dirs = {
        key: find_fragment_dir(tag)
        for key, tag in TAGS.items()
    }

    paths = {
        key: {
            "sdf": find_one(directory, "*_ligand.sdf"),
            "complex": find_one(directory, "*_complex.pdb"),
        }
        for key, directory in fragment_dirs.items()
    }

    mol_a = load_sdf(paths["A"]["sdf"])
    mol_easy = load_sdf(paths["B_easy"]["sdf"])
    mol_hard = load_sdf(paths["B_hard"]["sdf"])

    residue_key, pdb_atoms = select_pdb_residue(
        mol_a,
        paths["A"]["complex"],
    )

    mapping = map_sdf_to_pdb(mol_a, pdb_atoms)

    selected_system, all_systems = choose_primary_ring_system(mol_a)

    core = set(selected_system["atoms"])
    anchor_plus_shell = one_shell_heavy_neighbours(mol_a, core)

    n_heavy_total = len(heavy_indices(mol_a))
    n_free_plus_shell = n_heavy_total - len(anchor_plus_shell)

    # Predetermined A-only fallback:
    # retain at least three original heavy atoms outside the fixed mask.
    if n_free_plus_shell >= 3:
        anchor = anchor_plus_shell
        rule_used = "largest_ring_system_plus_one_heavy_atom_shell"
    else:
        anchor = core
        rule_used = (
            "largest_ring_system_only_fallback_"
            "because_plus_shell_left_fewer_than_3_free_atoms"
        )

    anchor = sorted(anchor)
    core = sorted(core)

    pdb_names = [mapping[idx]["pdb_name"] for idx in anchor]
    elements = [
        mol_a.GetAtomWithIdx(idx).GetSymbol()
        for idx in anchor
    ]

    # Freeze the mask before target characterization.
    frozen = {
        "anchor_tag": "x0434",
        "selection_blind_to_targets": True,
        "selection_rule": rule_used,
        "selection_variables": [
            "ring_system_size",
            "aromatic_fraction",
            "boundary_bonds",
            "minimum_rdkit_index_tiebreak",
        ],
        "n_heavy_A": n_heavy_total,
        "n_fixed": len(anchor),
        "n_unfixed_original_A": n_heavy_total - len(anchor),
        "fixed_fraction_of_A": float(len(anchor)) / n_heavy_total,
        "rdkit_indices_0based": anchor,
        "rdkit_indices_1based": [idx + 1 for idx in anchor],
        "pdb_atom_names": pdb_names,
        "elements": elements,
        "complex_pdb": str(paths["A"]["complex"]),
        "ligand_sdf": str(paths["A"]["sdf"]),
        "pdb_residue": {
            "chain": residue_key[0],
            "resid": residue_key[1],
            "icode": residue_key[2],
            "resname": residue_key[3],
        },
    }

    frozen_path = OUTDIR / "x0434_fixed_anchor_BLIND.json"

    with frozen_path.open("w") as handle:
        json.dump(frozen, handle, indent=2)

    # Descriptive target measurements are computed only after freezing.
    anchor_coords = atom_coordinates(mol_a, anchor)

    easy_coords = atom_coordinates(
        mol_easy,
        heavy_indices(mol_easy),
    )

    hard_coords = atom_coordinates(
        mol_hard,
        heavy_indices(mol_hard),
    )

    characterization = {
        "anchor_file": str(frozen_path),
        "anchor_selection_blind_to_targets": True,
        "x1093": nearest_metrics(anchor_coords, easy_coords),
        "x2193": nearest_metrics(anchor_coords, hard_coords),
    }

    characterization_path = (
        OUTDIR / "x0434_anchor_target_characterization.json"
    )

    with characterization_path.open("w") as handle:
        json.dump(characterization, handle, indent=2)

    mapping_path = OUTDIR / "x0434_atom_mapping.tsv"

    with mapping_path.open("w", newline="") as handle:
        fields = [
            "rdkit_index_0based",
            "rdkit_index_1based",
            "element",
            "pdb_atom_name",
            "mapping_distance_A",
            "is_primary_ring_core",
            "is_fixed_anchor",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
        )

        writer.writeheader()

        for idx in sorted(mapping):
            writer.writerow({
                "rdkit_index_0based": idx,
                "rdkit_index_1based": idx + 1,
                "element": mol_a.GetAtomWithIdx(idx).GetSymbol(),
                "pdb_atom_name": mapping[idx]["pdb_name"],
                "mapping_distance_A":
                    mapping[idx]["mapping_distance"],
                "is_primary_ring_core": idx in core,
                "is_fixed_anchor": idx in anchor,
            })

    figure_path = OUTDIR / "x0434_blind_anchor.png"

    make_2d_figure(
        mol_a,
        core,
        anchor,
        figure_path,
    )

    cli_path = OUTDIR / "x0434_fix_atoms_cli.txt"
    cli_path.write_text(
        "--fix_atoms {}\n".format(" ".join(pdb_names))
    )

    print("X0434_BLIND_ANCHOR_FROZEN")
    print("selection_rule={}".format(rule_used))
    print("n_heavy_A={}".format(n_heavy_total))
    print("n_fixed={}".format(len(anchor)))
    print(
        "fixed_fraction={:.3f}".format(
            float(len(anchor)) / n_heavy_total
        )
    )
    print(
        "rdkit_indices_0based={}".format(
            " ".join(str(idx) for idx in anchor)
        )
    )
    print(
        "pdb_atom_names={}".format(
            " ".join(pdb_names)
        )
    )

    print()
    print("CLI:")
    print("--fix_atoms {}".format(" ".join(pdb_names)))

    print()
    print("POST-FREEZE TARGET CHARACTERIZATION")

    for target in ["x1093", "x2193"]:
        metrics = characterization[target]

        print(
            "{} mean={:.3f} median={:.3f} max={:.3f} "
            "frac<=1.7={:.3f} frac<=2.5={:.3f}".format(
                target,
                metrics["mean_nearest_A"],
                metrics["median_nearest_A"],
                metrics["max_nearest_A"],
                metrics["fraction_within_1p7A"],
                metrics["fraction_within_2p5A"],
            )
        )

    print()
    print("Outputs:")
    print(frozen_path)
    print(characterization_path)
    print(mapping_path)
    print(figure_path)
    print(cli_path)


if __name__ == "__main__":
    main()
