import csv
import math
import os
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors, Descriptors
    RDKIT_OK = True
except Exception as e:
    RDKIT_OK = False
    RDKIT_ERROR = repr(e)


def read_manifest(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def load_mol(sdf_path):
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {sdf_path}")

    sanitize_status = "OK"
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        sanitize_status = "SANITIZE_FAILED:" + repr(e)

    return mol, sanitize_status


def heavy_atom_positions(mol):
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformer in molecule")

    conf = mol.GetConformer()
    coords = []
    elems = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
        elems.append(atom.GetSymbol())

    if not coords:
        raise ValueError("No heavy atoms")

    return np.array(coords, dtype=float), elems


def geom_features(coords):
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    rgyr = float(np.sqrt((centered * centered).sum(axis=1).mean()))

    if len(coords) >= 2:
        dmax = 0.0
        for i in range(len(coords)):
            diff = coords[i + 1:] - coords[i]
            if len(diff):
                d = np.sqrt((diff * diff).sum(axis=1)).max()
                dmax = max(dmax, float(d))
    else:
        dmax = 0.0

    cov = np.cov(centered.T) if len(coords) >= 3 else np.zeros((3, 3))
    try:
        vals = np.linalg.eigvalsh(cov)
        vals = np.sort(np.maximum(vals, 0.0))[::-1]
        pca_extent = np.sqrt(vals)
    except Exception:
        pca_extent = np.zeros(3)

    bbox = coords.max(axis=0) - coords.min(axis=0)
    bbox_sorted = np.sort(bbox)[::-1]

    return {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "rgyr": rgyr,
        "max_pairwise_dist": dmax,
        "pca_extent_1": float(pca_extent[0]),
        "pca_extent_2": float(pca_extent[1]),
        "pca_extent_3": float(pca_extent[2]),
        "bbox_1": float(bbox_sorted[0]),
        "bbox_2": float(bbox_sorted[1]),
        "bbox_3": float(bbox_sorted[2]),
    }


def fingerprint(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return Chem.RDKFingerprint(mol)


def tanimoto(fp1, fp2):
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return float("nan")


def vec(row, keys):
    return np.array([safe_float(row[k]) for k in keys], dtype=float)


def main():
    if not RDKIT_OK:
        raise SystemExit("RDKit import failed: " + RDKIT_ERROR)

    manifest = Path("data/mpro/manifests/silvr_xchem_ready_manifest.tsv")
    out_ligands = Path("data/mpro/manifests/b0_ligand_geometry_summary.tsv")
    out_pairs = Path("data/mpro/manifests/b0_candidate_pairs.tsv")
    out_reco = Path("artifacts/reports/pair_selection/B0_RECOMMENDED_PAIRS.txt")

    rows = read_manifest(manifest)

    ligands = []
    fps = {}

    for r in rows:
        if r["status"] != "READY":
            continue

        sdf = Path(r["ligand_sdf"])
        if not sdf.exists() or sdf.stat().st_size == 0:
            raise FileNotFoundError(f"Missing SDF: {sdf}")

        mol, sanitize_status = load_mol(sdf)
        coords, elems = heavy_atom_positions(mol)
        gf = geom_features(coords)

        fp = fingerprint(mol)
        fps[r["xchem_id"]] = fp

        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            smiles = ""

        try:
            formula = rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            formula = ""

        entry = dict(r)
        entry.update(gf)
        entry.update({
            "sanitize_status": sanitize_status,
            "rdkit_atoms": mol.GetNumAtoms(),
            "rdkit_bonds": mol.GetNumBonds(),
            "heavy_atoms_rdkit": mol.GetNumHeavyAtoms(),
            "formula": formula,
            "mol_wt": Descriptors.MolWt(mol),
            "smiles": smiles,
            "elements": ",".join(sorted(set(elems))),
        })

        ligands.append(entry)

    ligand_fields = [
        "xchem_id", "base", "ligand_comp_id", "status",
        "ligand_atoms", "ligand_heavy_atoms",
        "rdkit_atoms", "rdkit_bonds", "heavy_atoms_rdkit",
        "sanitize_status", "formula", "mol_wt",
        "centroid_x", "centroid_y", "centroid_z",
        "rgyr", "max_pairwise_dist",
        "pca_extent_1", "pca_extent_2", "pca_extent_3",
        "bbox_1", "bbox_2", "bbox_3",
        "elements", "smiles", "protein_pdb", "ligand_sdf"
    ]

    with out_ligands.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ligand_fields, delimiter="\t")
        w.writeheader()
        for r in ligands:
            w.writerow({k: r.get(k, "") for k in ligand_fields})

    pair_rows = []

    centroid_keys = ["centroid_x", "centroid_y", "centroid_z"]
    shape_keys = ["pca_extent_1", "pca_extent_2", "pca_extent_3", "rgyr", "max_pairwise_dist"]

    for i in range(len(ligands)):
        for j in range(i + 1, len(ligands)):
            a = ligands[i]
            b = ligands[j]

            ca = vec(a, centroid_keys)
            cb = vec(b, centroid_keys)
            centroid_dist = float(np.linalg.norm(ca - cb))

            sa = vec(a, shape_keys)
            sb = vec(b, shape_keys)
            shape_delta = float(np.linalg.norm(sa - sb))

            tani = tanimoto(fps[a["xchem_id"]], fps[b["xchem_id"]])
            chem_dissim = 1.0 - tani if not math.isnan(tani) else float("nan")

            heavy_delta = abs(int(a["heavy_atoms_rdkit"]) - int(b["heavy_atoms_rdkit"]))
            rgyr_delta = abs(float(a["rgyr"]) - float(b["rgyr"]))

            # Selection logic:
            # - Prefer chemically different molecules.
            # - Prefer shape/size difference.
            # - Penalize very displaced centroids because those may be different subpocket regimes.
            # - Penalize huge heavy atom mismatch because global guidance would become confounded by size.
            centroid_penalty = max(0.0, centroid_dist - 6.0) * 0.35
            heavy_penalty = max(0.0, heavy_delta - 6.0) * 0.10

            score = (
                chem_dissim * 2.0
                + min(shape_delta / 3.0, 1.5)
                + min(rgyr_delta / 1.5, 1.0)
                - centroid_penalty
                - heavy_penalty
            )

            same_pocket_flag = "YES" if centroid_dist <= 6.0 else "CHECK"

            pair_rows.append({
                "A_local": a["xchem_id"],
                "B_global": b["xchem_id"],
                "A_comp": a["ligand_comp_id"],
                "B_comp": b["ligand_comp_id"],
                "A_heavy": a["heavy_atoms_rdkit"],
                "B_heavy": b["heavy_atoms_rdkit"],
                "heavy_delta": heavy_delta,
                "centroid_dist_A": centroid_dist,
                "tanimoto_morgan": tani,
                "chem_dissim": chem_dissim,
                "shape_delta": shape_delta,
                "rgyr_A": a["rgyr"],
                "rgyr_B": b["rgyr"],
                "rgyr_delta": rgyr_delta,
                "same_pocket_flag": same_pocket_flag,
                "selection_score": score,
                "A_protein_pdb": a["protein_pdb"],
                "A_ligand_sdf": a["ligand_sdf"],
                "B_protein_pdb": b["protein_pdb"],
                "B_ligand_sdf": b["ligand_sdf"],
            })

    pair_rows.sort(key=lambda r: r["selection_score"], reverse=True)

    pair_fields = [
        "A_local", "B_global", "A_comp", "B_comp",
        "A_heavy", "B_heavy", "heavy_delta",
        "centroid_dist_A", "tanimoto_morgan", "chem_dissim",
        "shape_delta", "rgyr_A", "rgyr_B", "rgyr_delta",
        "same_pocket_flag", "selection_score",
        "A_protein_pdb", "A_ligand_sdf", "B_protein_pdb", "B_ligand_sdf"
    ]

    with out_pairs.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_fields, delimiter="\t")
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    strict = [
        r for r in pair_rows
        if r["same_pocket_flag"] == "YES"
        and r["tanimoto_morgan"] <= 0.50
        and r["heavy_delta"] <= 6
    ]

    relaxed = [
        r for r in pair_rows
        if r["centroid_dist_A"] <= 8.0
        and r["tanimoto_morgan"] <= 0.65
        and r["heavy_delta"] <= 8
    ]

    selected = strict[:10] if strict else relaxed[:10]

    with out_reco.open("w") as f:
        f.write("B0 pair recommendation report\n")
        f.write("=============================\n\n")
        f.write(f"RDKit ligands loaded: {len(ligands)}\n")
        f.write(f"All pair count: {len(pair_rows)}\n")
        f.write(f"Strict candidate count: {len(strict)}\n")
        f.write(f"Relaxed candidate count: {len(relaxed)}\n\n")

        f.write("Recommended pairs, interpreted as A_local scaffold source and B_global shape source:\n\n")
        for idx, r in enumerate(selected, 1):
            f.write(
                f"{idx}. A={r['A_local']} ({r['A_comp']}) -> "
                f"B={r['B_global']} ({r['B_comp']}) | "
                f"score={r['selection_score']:.3f}, "
                f"centroid={r['centroid_dist_A']:.2f} Å, "
                f"Tanimoto={r['tanimoto_morgan']:.3f}, "
                f"shape_delta={r['shape_delta']:.3f}, "
                f"heavy_delta={r['heavy_delta']}\n"
            )
            f.write(f"   A_ligand_sdf={r['A_ligand_sdf']}\n")
            f.write(f"   B_ligand_sdf={r['B_ligand_sdf']}\n\n")

    print("Wrote:", out_ligands)
    print("Wrote:", out_pairs)
    print("Wrote:", out_reco)
    print()
    print("=== Ligands ===")
    for r in ligands:
        print(
            r["xchem_id"],
            r["ligand_comp_id"],
            "heavy=" + str(r["heavy_atoms_rdkit"]),
            "bonds=" + str(r["rdkit_bonds"]),
            "rgyr={:.2f}".format(float(r["rgyr"])),
            r["sanitize_status"]
        )

    print()
    print("=== Top 10 candidate pairs ===")
    for r in pair_rows[:10]:
        print(
            "{} -> {} | score={:.3f} centroid={:.2f}A tanimoto={:.3f} shape_delta={:.3f} heavy_delta={} {}".format(
                r["A_local"],
                r["B_global"],
                r["selection_score"],
                r["centroid_dist_A"],
                r["tanimoto_morgan"],
                r["shape_delta"],
                r["heavy_delta"],
                r["same_pocket_flag"],
            )
        )


if __name__ == "__main__":
    main()
