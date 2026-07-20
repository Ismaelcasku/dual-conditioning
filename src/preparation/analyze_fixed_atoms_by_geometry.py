import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


def get_coords(mol):
    conf = mol.GetConformer()
    coords = []
    symbols = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
        symbols.append(atom.GetSymbol())
    return np.asarray(coords, dtype=float), symbols


def assignment_bruteforce(ref, gen):
    """
    Exact assignment for 7 reference atoms against n generated atoms.
    Brute force over permutations of generated indices.
    n is small enough for n<=16, but can be large for n=29/32.
    We therefore first keep the 14 closest generated candidates to the fixed refs.
    """
    n_ref = len(ref)
    dmat_full = np.linalg.norm(ref[:, None, :] - gen[None, :, :], axis=2)

    candidate_scores = dmat_full.min(axis=0)
    keep_n = min(len(gen), max(14, n_ref))
    keep = np.argsort(candidate_scores)[:keep_n]

    dmat = dmat_full[:, keep]

    best = None
    best_perm = None

    for perm in itertools.permutations(range(len(keep)), n_ref):
        vals = [dmat[i, perm[i]] for i in range(n_ref)]
        rmsd = float(np.sqrt(np.mean(np.square(vals))))
        if best is None or rmsd < best:
            best = rmsd
            best_perm = perm

    matched_gen_indices = [int(keep[j]) for j in best_perm]
    matched_distances = [
        float(dmat_full[i, matched_gen_indices[i]])
        for i in range(n_ref)
    ]

    return best, matched_gen_indices, matched_distances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_sdf", required=True)
    ap.add_argument("--gen_sdf", required=True)
    ap.add_argument("--fixed_json", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    with open(args.fixed_json) as f:
        fixed = json.load(f)

    ref_mol = Chem.MolFromMolFile(args.ref_sdf, sanitize=False, removeHs=False)
    if ref_mol is None:
        raise SystemExit("Could not read reference SDF")

    ref_conf = ref_mol.GetConformer()
    fixed_idx = fixed["fixed_atom_indices_0based"]
    fixed_names = fixed["fixed_pdb_atom_names"]

    ref_coords = []
    for idx in fixed_idx:
        p = ref_conf.GetAtomPosition(idx)
        ref_coords.append([p.x, p.y, p.z])
    ref_coords = np.asarray(ref_coords, dtype=float)

    mols = [m for m in Chem.SDMolSupplier(args.gen_sdf, sanitize=False, removeHs=False) if m is not None]

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample",
        "n_heavy",
        "fixed_atom_names",
        "best_geom_rmsd_A",
        "max_matched_distance_A",
        "mean_matched_distance_A",
        "matched_generated_indices_0based",
        "matched_generated_symbols",
        "matched_distances_A",
        "n_fixed_atoms_matched_under_0p2A",
        "n_fixed_atoms_matched_under_0p5A",
        "n_fixed_atoms_matched_under_1p0A",
    ]

    with out.open("w") as f:
        f.write("\t".join(fields) + "\n")

        for i, mol in enumerate(mols, 1):
            gen_coords, gen_symbols = get_coords(mol)

            best, matched, dists = assignment_bruteforce(ref_coords, gen_coords)

            matched_symbols = [gen_symbols[j] for j in matched]

            row = {
                "sample": i,
                "n_heavy": len(gen_coords),
                "fixed_atom_names": ",".join(fixed_names),
                "best_geom_rmsd_A": best,
                "max_matched_distance_A": max(dists),
                "mean_matched_distance_A": float(np.mean(dists)),
                "matched_generated_indices_0based": ",".join(map(str, matched)),
                "matched_generated_symbols": ",".join(matched_symbols),
                "matched_distances_A": ",".join("{:.4f}".format(x) for x in dists),
                "n_fixed_atoms_matched_under_0p2A": sum(d <= 0.2 for d in dists),
                "n_fixed_atoms_matched_under_0p5A": sum(d <= 0.5 for d in dists),
                "n_fixed_atoms_matched_under_1p0A": sum(d <= 1.0 for d in dists),
            }

            f.write("\t".join(str(row[k]) for k in fields) + "\n")

    print(f"mols={len(mols)}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
