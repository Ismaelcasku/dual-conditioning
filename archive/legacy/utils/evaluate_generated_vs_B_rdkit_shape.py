import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdShapeHelpers


def read_one(path):
    mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    if mol is None:
        raise SystemExit(f"Could not read {path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def read_many(path):
    return [m for m in Chem.SDMolSupplier(path, sanitize=False, removeHs=False) if m is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_sdf", required=True)
    ap.add_argument("--B_sdf", required=True)
    ap.add_argument("--gen_sdf", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    A = read_one(args.A_sdf)
    B = read_one(args.B_sdf)
    mols = read_many(args.gen_sdf)

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample",
        "n_atoms",
        "shape_tanimoto_dist_to_A",
        "shape_tanimoto_dist_to_B",
        "shape_protrude_dist_to_A",
        "shape_protrude_dist_to_B",
        "closer_shape_tanimoto",
        "closer_shape_protrude",
    ]

    with out.open("w") as f:
        f.write("\t".join(fields) + "\n")

        for i, mol in enumerate(mols, 1):
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass

            st_A = rdShapeHelpers.ShapeTanimotoDist(mol, A)
            st_B = rdShapeHelpers.ShapeTanimotoDist(mol, B)

            sp_A = rdShapeHelpers.ShapeProtrudeDist(mol, A)
            sp_B = rdShapeHelpers.ShapeProtrudeDist(mol, B)

            row = {
                "sample": i,
                "n_atoms": mol.GetNumAtoms(),
                "shape_tanimoto_dist_to_A": st_A,
                "shape_tanimoto_dist_to_B": st_B,
                "shape_protrude_dist_to_A": sp_A,
                "shape_protrude_dist_to_B": sp_B,
                "closer_shape_tanimoto": "A" if st_A < st_B else "B",
                "closer_shape_protrude": "A" if sp_A < sp_B else "B",
            }

            f.write("\t".join(str(row[k]) for k in fields) + "\n")

    print(f"mols={len(mols)}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
