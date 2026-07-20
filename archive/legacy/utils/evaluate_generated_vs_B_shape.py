import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def get_heavy_coords(mol):
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


def geom_features(coords):
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    rgyr = float(np.sqrt((centered * centered).sum(axis=1).mean()))

    dmax = 0.0
    for i in range(len(coords)):
        diff = coords[i + 1:] - coords[i]
        if len(diff):
            dmax = max(dmax, float(np.sqrt((diff * diff).sum(axis=1)).max()))

    cov = np.cov(centered.T) if len(coords) >= 3 else np.zeros((3, 3))
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))[::-1]
    pca_extent = np.sqrt(vals)

    return {
        "centroid": centroid,
        "rgyr": rgyr,
        "dmax": dmax,
        "pca": pca_extent,
    }


def centroid_distance(a, b):
    return float(np.linalg.norm(a["centroid"] - b["centroid"]))


def shape_delta(a, b):
    va = np.asarray([a["pca"][0], a["pca"][1], a["pca"][2], a["rgyr"], a["dmax"]])
    vb = np.asarray([b["pca"][0], b["pca"][1], b["pca"][2], b["rgyr"], b["dmax"]])
    return float(np.linalg.norm(va - vb))


def fp(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return Chem.RDKFingerprint(mol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_sdf", required=True)
    ap.add_argument("--B_sdf", required=True)
    ap.add_argument("--gen_sdf", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    A = Chem.MolFromMolFile(args.A_sdf, sanitize=False, removeHs=False)
    B = Chem.MolFromMolFile(args.B_sdf, sanitize=False, removeHs=False)

    if A is None:
        raise SystemExit("Could not read A_sdf")
    if B is None:
        raise SystemExit("Could not read B_sdf")

    try:
        Chem.SanitizeMol(A)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(B)
    except Exception:
        pass

    A_coords, _ = get_heavy_coords(A)
    B_coords, _ = get_heavy_coords(B)

    A_feat = geom_features(A_coords)
    B_feat = geom_features(B_coords)

    A_fp = fp(A)
    B_fp = fp(B)

    mols = [m for m in Chem.SDMolSupplier(args.gen_sdf, sanitize=False, removeHs=False) if m is not None]

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample",
        "n_heavy",
        "centroid_dist_to_A",
        "centroid_dist_to_B",
        "shape_delta_to_A",
        "shape_delta_to_B",
        "rgyr",
        "rgyr_A",
        "rgyr_B",
        "dmax",
        "dmax_A",
        "dmax_B",
        "tanimoto_to_A",
        "tanimoto_to_B",
        "closer_centroid",
        "closer_shape",
        "closer_chem",
    ]

    with out.open("w") as f:
        f.write("\t".join(fields) + "\n")

        for i, mol in enumerate(mols, 1):
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass

            coords, _ = get_heavy_coords(mol)
            feat = geom_features(coords)
            mfp = fp(mol)

            cd_A = centroid_distance(feat, A_feat)
            cd_B = centroid_distance(feat, B_feat)
            sd_A = shape_delta(feat, A_feat)
            sd_B = shape_delta(feat, B_feat)
            tani_A = float(DataStructs.TanimotoSimilarity(mfp, A_fp))
            tani_B = float(DataStructs.TanimotoSimilarity(mfp, B_fp))

            row = {
                "sample": i,
                "n_heavy": len(coords),
                "centroid_dist_to_A": cd_A,
                "centroid_dist_to_B": cd_B,
                "shape_delta_to_A": sd_A,
                "shape_delta_to_B": sd_B,
                "rgyr": feat["rgyr"],
                "rgyr_A": A_feat["rgyr"],
                "rgyr_B": B_feat["rgyr"],
                "dmax": feat["dmax"],
                "dmax_A": A_feat["dmax"],
                "dmax_B": B_feat["dmax"],
                "tanimoto_to_A": tani_A,
                "tanimoto_to_B": tani_B,
                "closer_centroid": "A" if cd_A < cd_B else "B",
                "closer_shape": "A" if sd_A < sd_B else "B",
                "closer_chem": "A" if tani_A > tani_B else "B",
            }

            f.write("\t".join(str(row[k]) for k in fields) + "\n")

    print(f"mols={len(mols)}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
