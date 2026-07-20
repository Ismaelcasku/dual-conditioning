#!/usr/bin/env python
"""
chem_novelty_peek.py — CHEAP chemical novelty/diversity peek (ECFP4) over a
baseline SDF and one or more guided SDFs.

PURPOSE: decide whether guidance produced chemically DIFFERENT molecules, or just
geometrically distorted versions of the same chemistry. Our guide only moved
COORDINATES (atom types were left to the model), so chemical novelty is NOT
guaranteed — this measures it.

NOT A CLAIM: with n~5 per condition this is a peek to decide whether the proper
physical-validation run (PoseCheck + docking, larger n) is worth it.

Metrics per condition (ECFP4, radius 2, 2048 bits, Tanimoto):
  n_valid           : molecules read+sanitized OK
  sim_to_A          : mean ECFP Tanimoto to reference A (the warhead source)
  sim_to_B          : mean ECFP Tanimoto to reference B
  internal_div      : 1 - mean pairwise Tanimoto within the set (higher = more diverse)
  NN_sim_to_base    : mean nearest-neighbor Tanimoto to the BASELINE set
  novelty_vs_base   : 1 - NN_sim_to_base (higher = chemically farther from baseline)

Usage:
  python chem_novelty_peek.py \
     --A_sdf A.sdf --B_sdf B.sdf \
     --baseline base=path/to/lambda0.sdf \
     --gen l1=path/to/l1.sdf --gen l5=path/to/l5.sdf --gen l20=path/to/l20.sdf
"""
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def read_fps(path):
    """Read all records; sanitize; return list of fingerprints (skip failures)."""
    supp = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    fps = []
    n_records = 0
    for mol in supp:
        n_records += 1
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            fps.append(fp(mol))
        except Exception:
            continue
    return fps, n_records


def read_one(path):
    mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    Chem.SanitizeMol(mol)
    return fp(mol)


def mean_sim_to_ref(fps, ref):
    if not fps:
        return float("nan")
    return float(np.mean([DataStructs.TanimotoSimilarity(f, ref) for f in fps]))


def internal_diversity(fps):
    if len(fps) < 2:
        return float("nan")
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims))


def nn_sim_to_set(fps, ref_fps, exclude_self=False):
    """Mean over fps of the max Tanimoto to any fingerprint in ref_fps."""
    if not fps or not ref_fps:
        return float("nan")
    out = []
    for i, f in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(f, ref_fps)
        if exclude_self:
            sims[i] = -1.0  # drop self when comparing a set to itself
        out.append(max(sims))
    return float(np.mean(out))


def parse_kv(s):
    label, path = s.split("=", 1)
    return label, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_sdf", required=True)
    ap.add_argument("--B_sdf", required=True)
    ap.add_argument("--baseline", required=True, type=parse_kv,
                    help="label=path for the lambda=0 baseline SDF")
    ap.add_argument("--gen", action="append", default=[], type=parse_kv,
                    help="label=path for a guided SDF (repeatable)")
    args = ap.parse_args()

    A_fp = read_one(args.A_sdf)
    B_fp = read_one(args.B_sdf)

    base_label, base_path = args.baseline
    base_fps, base_n = read_fps(base_path)

    conditions = [(base_label, base_path, base_fps, base_n, True)]
    for label, path in args.gen:
        fps, n = read_fps(path)
        conditions.append((label, path, fps, n, False))

    header = ["condition", "n_valid/rec", "sim_to_A", "sim_to_B",
              "internal_div", "NN_sim_to_base", "novelty_vs_base"]
    rows = []
    for label, path, fps, n_rec, is_base in conditions:
        nn = nn_sim_to_set(fps, base_fps, exclude_self=is_base)
        nov = (1.0 - nn) if not np.isnan(nn) else float("nan")
        rows.append([
            label,
            f"{len(fps)}/{n_rec}",
            f"{mean_sim_to_ref(fps, A_fp):.3f}",
            f"{mean_sim_to_ref(fps, B_fp):.3f}",
            f"{internal_diversity(fps):.3f}",
            f"{nn:.3f}",
            ("(baseline)" if is_base else f"{nov:.3f}"),
        ])

    w = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    line = lambda r: "  ".join(str(r[i]).ljust(w[i]) for i in range(len(r)))
    print("\n=== CHEAP ECFP novelty peek (n~5 per condition — NOT a claim) ===\n")
    print(line(header))
    print("  ".join("-" * w[i] for i in range(len(header))))
    for r in rows:
        print(line(r))
    print("\nReading guide:")
    print("  novelty_vs_base near 0  -> chemically ~identical to baseline:")
    print("     guidance only distorted geometry, not chemistry -> likely noise, close it.")
    print("  novelty_vs_base clearly >0 AND molecules valid -> chemically distinct:")
    print("     justifies the proper physical-validation run (PoseCheck + docking, larger n).")
    print("  (Novelty alone is not value: only useful if poses/docking survive — that is level 2.)")


if __name__ == "__main__":
    main()

