import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


def kabsch_rmsd(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)

    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])

    U = V @ D @ Wt
    P_rot = Pc @ U

    return float(np.sqrt(((P_rot - Qc) ** 2).sum(axis=1).mean()))


def direct_rmsd(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))


def read_mols(path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    return [m for m in supplier if m is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_sdf", required=True)
    ap.add_argument("--out_sdf", required=True)
    ap.add_argument("--fixed_json", required=True)
    ap.add_argument("--report_tsv", required=True)
    args = ap.parse_args()

    ref = Chem.MolFromMolFile(args.ref_sdf, sanitize=False, removeHs=False)
    if ref is None:
        raise SystemExit(f"Could not read ref_sdf: {args.ref_sdf}")

    with open(args.fixed_json) as f:
        fixed = json.load(f)

    # Original atom indices inside A=x0874/T54.
    ref_fixed_idx = fixed["fixed_atom_indices_0based"]
    fixed_names = fixed["fixed_pdb_atom_names"]

    ref_conf = ref.GetConformer()
    ref_fixed = []
    for idx in ref_fixed_idx:
        p = ref_conf.GetAtomPosition(idx)
        ref_fixed.append([p.x, p.y, p.z])
    ref_fixed = np.asarray(ref_fixed, dtype=float)

    # DiffSBDD fills fixed atoms into the FIRST n_fixed generated ligand nodes.
    n_fixed = len(ref_fixed_idx)
    gen_fixed_idx = list(range(n_fixed))

    mols = read_mols(args.out_sdf)

    rows = []
    for i, mol in enumerate(mols, 1):
        status = "OK"
        msg = ""

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            status = "SANITIZE_WARN"
            msg = repr(e)

        n_atoms = mol.GetNumAtoms()
        n_heavy = mol.GetNumHeavyAtoms()
        n_bonds = mol.GetNumBonds()

        rmsd_direct = ""
        rmsd_kabsch = ""

        try:
            if mol.GetNumConformers() == 0:
                raise ValueError("no conformer")

            if mol.GetNumAtoms() < n_fixed:
                status = "TOO_FEW_ATOMS_FOR_FIXED_BLOCK"
            else:
                conf = mol.GetConformer()
                gen_fixed = []
                for idx in gen_fixed_idx:
                    p = conf.GetAtomPosition(idx)
                    gen_fixed.append([p.x, p.y, p.z])
                gen_fixed = np.asarray(gen_fixed, dtype=float)

                rmsd_direct = direct_rmsd(ref_fixed, gen_fixed)
                rmsd_kabsch = kabsch_rmsd(ref_fixed, gen_fixed)

        except Exception as e:
            status = "RMSD_FAILED"
            msg = repr(e)

        rows.append({
            "sample": i,
            "status": status,
            "n_atoms": n_atoms,
            "n_heavy": n_heavy,
            "n_bonds": n_bonds,
            "fixed_atom_names": ",".join(fixed_names),
            "ref_fixed_indices_0based": ",".join(map(str, ref_fixed_idx)),
            "generated_fixed_block_indices_0based": ",".join(map(str, gen_fixed_idx)),
            "fixed_rmsd_direct_A": rmsd_direct,
            "fixed_rmsd_kabsch_A": rmsd_kabsch,
            "message": msg,
        })

    out = Path(args.report_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample", "status", "n_atoms", "n_heavy", "n_bonds",
        "fixed_atom_names", "ref_fixed_indices_0based",
        "generated_fixed_block_indices_0based",
        "fixed_rmsd_direct_A", "fixed_rmsd_kabsch_A",
        "message"
    ]

    with out.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in fields) + "\n")

    print(f"mols_read={len(mols)}")
    print(f"wrote={out}")
    for r in rows:
        print(
            f"sample={r['sample']} status={r['status']} "
            f"atoms={r['n_atoms']} heavy={r['n_heavy']} "
            f"rmsd_direct={r['fixed_rmsd_direct_A']} "
            f"rmsd_kabsch={r['fixed_rmsd_kabsch_A']}"
        )


if __name__ == "__main__":
    main()
