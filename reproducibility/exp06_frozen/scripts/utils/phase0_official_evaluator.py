#!/usr/bin/env python
"""
Official Phase 0 evaluator for dual-conditioning experiments.

Purpose
-------
Evaluate any generated SDF with one frozen schema across:
- baseline inpainting-only runs,
- future lambda_global > 0 runs,
- future A/B pairs.

This evaluator intentionally emits raw per-molecule metrics.
It does NOT decide whether a lambda_global condition improved over baseline.
That comparison belongs to a separate aggregation script across conditions.

Core frozen metrics
-------------------
Per molecule:
- RDKit read validity, including failed/None SDF records in the denominator.
- Sanitization status.
- Local A-substructure conservation by geometry matching.
- Hard element identity check for fixed atoms.
- RDKit ShapeTanimotoDist and ShapeProtrudeDist to A and B.
- Coarse descriptor distances to A and B.
- Size, formula, centroid, Rgyr, dmax.
- Informative flags:
    local_pass_all_atoms_0p2A
    b_closer_than_A_shape_tanimoto
    b_closer_than_A_shape_protrude
    dual_candidate_vs_A_strict

Important
---------
dual_candidate_vs_A_strict is only an informative per-molecule flag.
The actual global-guidance success criterion must be computed against
the lambda_global = 0 baseline in an aggregation step.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors, rdShapeHelpers

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise SystemExit(
        "scipy is required for the official evaluator. "
        "Install/use scipy in the sandbox rather than falling back to a fragile combinatorial solver. "
        f"Original error: {repr(exc)}"
    )


LOCAL_RMSD_THRESHOLD_A = 0.20
LOCAL_ATOM_DISTANCE_THRESHOLD_A = 0.20


def bool_str(x):
    return "TRUE" if bool(x) else "FALSE"


def nan():
    return float("nan")


def read_single_mol(path, name):
    mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        raise SystemExit(f"Could not read {name}: {path}")
    return mol


def read_generated_records(path):
    """
    Return all SDF records, preserving failed RDKit reads as None.
    This avoids inflating validity rates by silently dropping failed records.
    """
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    records = []
    for i, mol in enumerate(supplier, 1):
        records.append((i, mol))
    return records


def sanitize_status(mol):
    if mol is None:
        return "READ_FAIL", "RDKit supplier returned None"

    try:
        Chem.SanitizeMol(mol)
        return "OK", ""
    except Exception as e:
        return "WARN", repr(e)


def get_heavy_coords_symbols(mol):
    if mol is None:
        raise ValueError("Molecule is None")

    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    conf = mol.GetConformer()
    coords = []
    symbols = []
    atom_indices = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
        symbols.append(atom.GetSymbol())
        atom_indices.append(atom.GetIdx())

    if not coords:
        raise ValueError("Molecule has no heavy atoms")

    return np.asarray(coords, dtype=float), symbols, atom_indices


def get_fixed_reference(A_mol, fixed_json):
    conf = A_mol.GetConformer()
    fixed_idx = [int(x) for x in fixed_json["fixed_atom_indices_0based"]]
    fixed_names = list(fixed_json["fixed_pdb_atom_names"])

    coords = []
    symbols = []

    for idx in fixed_idx:
        atom = A_mol.GetAtomWithIdx(idx)
        p = conf.GetAtomPosition(idx)
        coords.append([p.x, p.y, p.z])
        symbols.append(atom.GetSymbol())

    return np.asarray(coords, dtype=float), symbols, fixed_idx, fixed_names


def assign_fixed_atoms_by_geometry(ref_coords, ref_symbols, gen_coords, gen_symbols):
    """
    Assign each fixed reference atom to a unique generated heavy atom.

    Element identity is hard:
    - Mismatched elements are not valid matches.
    - If no assignment with matching elements exists, conservation fails explicitly.
    """
    n_ref = len(ref_coords)
    n_gen = len(gen_coords)

    if n_gen < n_ref:
        return {
            "assignment_ok": False,
            "local_element_match_ok": False,
            "assignment_message": f"Generated molecule has fewer heavy atoms ({n_gen}) than fixed atoms ({n_ref})",
            "best_fixed_geom_rmsd_A": nan(),
            "mean_fixed_atom_dist_A": nan(),
            "max_fixed_atom_dist_A": nan(),
            "matched_generated_heavy_indices_0based": [],
            "matched_generated_symbols": [],
            "matched_distances_A": [],
            "n_fixed_atoms_under_0p2A": 0,
            "n_fixed_atoms_under_0p5A": 0,
            "n_fixed_atoms_under_1p0A": 0,
        }

    dmat = np.linalg.norm(ref_coords[:, None, :] - gen_coords[None, :, :], axis=2)

    # Hard element constraint: impossible assignments receive infinite cost.
    cost = dmat.copy()
    for i, rs in enumerate(ref_symbols):
        for j, gs in enumerate(gen_symbols):
            if rs != gs:
                cost[i, j] = np.inf

    if not np.all(np.isfinite(cost).any(axis=1)):
        return {
            "assignment_ok": False,
            "local_element_match_ok": False,
            "assignment_message": "At least one fixed atom has no generated atom with matching element",
            "best_fixed_geom_rmsd_A": nan(),
            "mean_fixed_atom_dist_A": nan(),
            "max_fixed_atom_dist_A": nan(),
            "matched_generated_heavy_indices_0based": [],
            "matched_generated_symbols": [],
            "matched_distances_A": [],
            "n_fixed_atoms_under_0p2A": 0,
            "n_fixed_atoms_under_0p5A": 0,
            "n_fixed_atoms_under_1p0A": 0,
        }

    # scipy cannot solve matrices with rows/cols all inf robustly in all versions.
    # Replace inf by a very large number only after checking row feasibility.
    large = 1e9
    finite_cost = np.where(np.isfinite(cost), cost, large)

    row_ind, col_ind = linear_sum_assignment(finite_cost)

    assignment = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_ref:
            assignment[int(r)] = int(c)

    if len(assignment) != n_ref:
        return {
            "assignment_ok": False,
            "local_element_match_ok": False,
            "assignment_message": "Incomplete fixed-atom assignment",
            "best_fixed_geom_rmsd_A": nan(),
            "mean_fixed_atom_dist_A": nan(),
            "max_fixed_atom_dist_A": nan(),
            "matched_generated_heavy_indices_0based": [],
            "matched_generated_symbols": [],
            "matched_distances_A": [],
            "n_fixed_atoms_under_0p2A": 0,
            "n_fixed_atoms_under_0p5A": 0,
            "n_fixed_atoms_under_1p0A": 0,
        }

    matched = [assignment[i] for i in range(n_ref)]
    matched_symbols = [gen_symbols[j] for j in matched]
    element_ok = all(ref_symbols[i] == matched_symbols[i] for i in range(n_ref))

    if not element_ok:
        return {
            "assignment_ok": False,
            "local_element_match_ok": False,
            "assignment_message": "Element mismatch in matched fixed atoms",
            "best_fixed_geom_rmsd_A": nan(),
            "mean_fixed_atom_dist_A": nan(),
            "max_fixed_atom_dist_A": nan(),
            "matched_generated_heavy_indices_0based": matched,
            "matched_generated_symbols": matched_symbols,
            "matched_distances_A": [],
            "n_fixed_atoms_under_0p2A": 0,
            "n_fixed_atoms_under_0p5A": 0,
            "n_fixed_atoms_under_1p0A": 0,
        }

    distances = [float(dmat[i, matched[i]]) for i in range(n_ref)]

    rmsd = float(np.sqrt(np.mean(np.square(distances))))
    mean_dist = float(np.mean(distances))
    max_dist = float(np.max(distances))

    return {
        "assignment_ok": True,
        "local_element_match_ok": True,
        "assignment_message": "",
        "best_fixed_geom_rmsd_A": rmsd,
        "mean_fixed_atom_dist_A": mean_dist,
        "max_fixed_atom_dist_A": max_dist,
        "matched_generated_heavy_indices_0based": matched,
        "matched_generated_symbols": matched_symbols,
        "matched_distances_A": distances,
        "n_fixed_atoms_under_0p2A": sum(d <= 0.2 for d in distances),
        "n_fixed_atoms_under_0p5A": sum(d <= 0.5 for d in distances),
        "n_fixed_atoms_under_1p0A": sum(d <= 1.0 for d in distances),
    }


def geom_features(coords):
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    rgyr = float(np.sqrt((centered * centered).sum(axis=1).mean()))

    if len(coords) >= 2:
        diff = coords[:, None, :] - coords[None, :, :]
        dmat = np.sqrt((diff * diff).sum(axis=2))
        dmax = float(dmat.max())
    else:
        dmax = 0.0

    cov = np.cov(centered.T) if len(coords) >= 3 else np.zeros((3, 3))
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))[::-1]
    pca = np.sqrt(vals)

    return {
        "centroid": centroid,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "rgyr": rgyr,
        "dmax": dmax,
        "pca1": float(pca[0]),
        "pca2": float(pca[1]),
        "pca3": float(pca[2]),
    }


def shape_descriptor_delta(a, b):
    va = np.asarray([a["pca1"], a["pca2"], a["pca3"], a["rgyr"], a["dmax"]], dtype=float)
    vb = np.asarray([b["pca1"], b["pca2"], b["pca3"], b["rgyr"], b["dmax"]], dtype=float)
    return float(np.linalg.norm(va - vb))


def centroid_dist(a, b):
    return float(np.linalg.norm(a["centroid"] - b["centroid"]))


def fingerprint(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return Chem.RDKFingerprint(mol)


def shape_distances(mol, A_mol, B_mol):
    out = {}
    try:
        out["shape_tanimoto_dist_to_A"] = float(rdShapeHelpers.ShapeTanimotoDist(mol, A_mol))
    except Exception:
        out["shape_tanimoto_dist_to_A"] = nan()

    try:
        out["shape_tanimoto_dist_to_B"] = float(rdShapeHelpers.ShapeTanimotoDist(mol, B_mol))
    except Exception:
        out["shape_tanimoto_dist_to_B"] = nan()

    try:
        out["shape_protrude_dist_to_A"] = float(rdShapeHelpers.ShapeProtrudeDist(mol, A_mol))
    except Exception:
        out["shape_protrude_dist_to_A"] = nan()

    try:
        out["shape_protrude_dist_to_B"] = float(rdShapeHelpers.ShapeProtrudeDist(mol, B_mol))
    except Exception:
        out["shape_protrude_dist_to_B"] = nan()

    return out


def init_failed_row(args, sample_idx, sanitize_msg):
    return {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "lambda_global": args.lambda_global,
        "lambda_local": args.lambda_local,
        "sample": sample_idx,
        "valid_rdkit_read": False,
        "sanitize_status": "READ_FAIL",
        "sanitize_message": sanitize_msg,
        "n_atoms": 0,
        "n_heavy": 0,
        "n_bonds": 0,
        "formula": "",
        "mol_wt": nan(),
        "assignment_ok": False,
        "assignment_message": "No molecule to evaluate",
        "local_element_match_ok": False,
        "n_fixed_expected": "",
        "fixed_atom_names": "",
        "fixed_ref_indices_0based": "",
        "best_fixed_geom_rmsd_A": nan(),
        "mean_fixed_atom_dist_A": nan(),
        "max_fixed_atom_dist_A": nan(),
        "n_fixed_atoms_under_0p2A": 0,
        "n_fixed_atoms_under_0p5A": 0,
        "n_fixed_atoms_under_1p0A": 0,
        "local_pass_rmsd_0p2A": False,
        "local_pass_all_atoms_0p2A": False,
        "matched_generated_heavy_indices_0based": "",
        "matched_generated_symbols": "",
        "matched_distances_A": "",
        "centroid_x": nan(),
        "centroid_y": nan(),
        "centroid_z": nan(),
        "centroid_dist_to_A": nan(),
        "centroid_dist_to_B": nan(),
        "closer_centroid": "",
        "shape_descriptor_delta_to_A": nan(),
        "shape_descriptor_delta_to_B": nan(),
        "closer_shape_descriptor": "",
        "rgyr": nan(),
        "rgyr_A": nan(),
        "rgyr_B": nan(),
        "dmax": nan(),
        "dmax_A": nan(),
        "dmax_B": nan(),
        "shape_tanimoto_dist_to_A": nan(),
        "shape_tanimoto_dist_to_B": nan(),
        "closer_shape_tanimoto": "",
        "shape_protrude_dist_to_A": nan(),
        "shape_protrude_dist_to_B": nan(),
        "closer_shape_protrude": "",
        "b_closer_than_A_shape_tanimoto": False,
        "b_closer_than_A_shape_protrude": False,
        "tanimoto_to_A": nan(),
        "tanimoto_to_B": nan(),
        "closer_chem": "",
        "dual_candidate_vs_A_strict": False,
        "sa_score": nan(),
        "nearest_neighbor_tanimoto_in_batch": nan(),
        "posecheck_clash_score": nan(),
        "posecheck_strain_energy": nan(),
        "docking_vina_score": nan(),
        "docking_gnina_score": nan(),
    }


def row_to_clean_dict(row, fields):
    clean = {}
    for k in fields:
        v = row.get(k, "")
        if isinstance(v, bool):
            v = bool_str(v)
        elif isinstance(v, float) and math.isnan(v):
            v = "NaN"
        clean[k] = v
    return clean


def mean_of(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, nan())
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--lambda_global", type=float, default=0.0)
    ap.add_argument("--lambda_local", type=float, default=1.0)
    ap.add_argument("--A_sdf", required=True)
    ap.add_argument("--B_sdf", required=True)
    ap.add_argument("--gen_sdf", required=True)
    ap.add_argument("--fixed_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = out_dir / f"{args.experiment_id}_official_phase0_metrics.tsv"
    out_json = out_dir / f"{args.experiment_id}_official_phase0_summary.json"
    out_md = out_dir / f"{args.experiment_id}_official_phase0_report.md"

    A_mol = read_single_mol(args.A_sdf, "A_sdf")
    B_mol = read_single_mol(args.B_sdf, "B_sdf")

    A_sanitize, A_sanitize_msg = sanitize_status(A_mol)
    B_sanitize, B_sanitize_msg = sanitize_status(B_mol)

    with open(args.fixed_json) as f:
        fixed = json.load(f)

    ref_fixed_coords, ref_fixed_symbols, ref_fixed_idx, fixed_names = get_fixed_reference(A_mol, fixed)

    A_coords, A_symbols, _ = get_heavy_coords_symbols(A_mol)
    B_coords, B_symbols, _ = get_heavy_coords_symbols(B_mol)

    A_geom = geom_features(A_coords)
    B_geom = geom_features(B_coords)

    A_fp = fingerprint(A_mol)
    B_fp = fingerprint(B_mol)

    records = read_generated_records(args.gen_sdf)
    rows = []

    for sample_idx, mol in records:
        if mol is None:
            rows.append(init_failed_row(args, sample_idx, "RDKit supplier returned None"))
            continue

        row = {}
        row["experiment_id"] = args.experiment_id
        row["mode"] = args.mode
        row["lambda_global"] = args.lambda_global
        row["lambda_local"] = args.lambda_local
        row["sample"] = sample_idx

        row["valid_rdkit_read"] = True
        s_status, s_msg = sanitize_status(mol)
        row["sanitize_status"] = s_status
        row["sanitize_message"] = s_msg

        row["n_atoms"] = mol.GetNumAtoms()
        row["n_heavy"] = mol.GetNumHeavyAtoms()
        row["n_bonds"] = mol.GetNumBonds()

        try:
            row["formula"] = rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            row["formula"] = ""

        try:
            row["mol_wt"] = float(Descriptors.MolWt(mol))
        except Exception:
            row["mol_wt"] = nan()

        try:
            coords, symbols, heavy_atom_indices = get_heavy_coords_symbols(mol)
            geom = geom_features(coords)

            local = assign_fixed_atoms_by_geometry(ref_fixed_coords, ref_fixed_symbols, coords, symbols)
            row.update(local)

            row["n_fixed_expected"] = len(ref_fixed_coords)
            row["fixed_atom_names"] = ",".join(fixed_names)
            row["fixed_ref_indices_0based"] = ",".join(map(str, ref_fixed_idx))
            row["matched_generated_heavy_indices_0based"] = ",".join(map(str, local["matched_generated_heavy_indices_0based"]))
            row["matched_generated_symbols"] = ",".join(local["matched_generated_symbols"])
            row["matched_distances_A"] = ",".join("{:.4f}".format(x) for x in local["matched_distances_A"])

            row["local_pass_rmsd_0p2A"] = (
                local["assignment_ok"]
                and local["local_element_match_ok"]
                and local["best_fixed_geom_rmsd_A"] <= LOCAL_RMSD_THRESHOLD_A
            )
            row["local_pass_all_atoms_0p2A"] = (
                local["assignment_ok"]
                and local["local_element_match_ok"]
                and local["n_fixed_atoms_under_0p2A"] == len(ref_fixed_coords)
            )

            row["centroid_x"] = geom["centroid_x"]
            row["centroid_y"] = geom["centroid_y"]
            row["centroid_z"] = geom["centroid_z"]
            row["centroid_dist_to_A"] = centroid_dist(geom, A_geom)
            row["centroid_dist_to_B"] = centroid_dist(geom, B_geom)
            row["closer_centroid"] = "A" if row["centroid_dist_to_A"] < row["centroid_dist_to_B"] else "B"

            row["shape_descriptor_delta_to_A"] = shape_descriptor_delta(geom, A_geom)
            row["shape_descriptor_delta_to_B"] = shape_descriptor_delta(geom, B_geom)
            row["closer_shape_descriptor"] = "A" if row["shape_descriptor_delta_to_A"] < row["shape_descriptor_delta_to_B"] else "B"

            row["rgyr"] = geom["rgyr"]
            row["rgyr_A"] = A_geom["rgyr"]
            row["rgyr_B"] = B_geom["rgyr"]
            row["dmax"] = geom["dmax"]
            row["dmax_A"] = A_geom["dmax"]
            row["dmax_B"] = B_geom["dmax"]

            row.update(shape_distances(mol, A_mol, B_mol))
            row["b_closer_than_A_shape_tanimoto"] = (
                not math.isnan(row["shape_tanimoto_dist_to_A"])
                and not math.isnan(row["shape_tanimoto_dist_to_B"])
                and row["shape_tanimoto_dist_to_B"] < row["shape_tanimoto_dist_to_A"]
            )
            row["b_closer_than_A_shape_protrude"] = (
                not math.isnan(row["shape_protrude_dist_to_A"])
                and not math.isnan(row["shape_protrude_dist_to_B"])
                and row["shape_protrude_dist_to_B"] < row["shape_protrude_dist_to_A"]
            )
            row["closer_shape_tanimoto"] = "B" if row["b_closer_than_A_shape_tanimoto"] else "A"
            row["closer_shape_protrude"] = "B" if row["b_closer_than_A_shape_protrude"] else "A"

            try:
                mfp = fingerprint(mol)
                row["tanimoto_to_A"] = float(DataStructs.TanimotoSimilarity(mfp, A_fp))
                row["tanimoto_to_B"] = float(DataStructs.TanimotoSimilarity(mfp, B_fp))
            except Exception:
                row["tanimoto_to_A"] = nan()
                row["tanimoto_to_B"] = nan()

            row["closer_chem"] = "A" if row["tanimoto_to_A"] >= row["tanimoto_to_B"] else "B"

            # Informative only. Real success with lambda_global > 0 must be evaluated against baseline aggregation.
            row["dual_candidate_vs_A_strict"] = (
                row["valid_rdkit_read"]
                and row["local_pass_all_atoms_0p2A"]
                and row["b_closer_than_A_shape_tanimoto"]
                and row["b_closer_than_A_shape_protrude"]
            )

            # Reserved schema columns for later physical/medchem validation.
            row["sa_score"] = nan()
            row["nearest_neighbor_tanimoto_in_batch"] = nan()
            row["posecheck_clash_score"] = nan()
            row["posecheck_strain_energy"] = nan()
            row["docking_vina_score"] = nan()
            row["docking_gnina_score"] = nan()

        except Exception as e:
            failed = init_failed_row(args, sample_idx, f"EVALUATION_FAIL: {repr(e)}")
            failed.update({
                "valid_rdkit_read": True,
                "sanitize_status": row.get("sanitize_status", "WARN"),
                "sanitize_message": row.get("sanitize_message", "") + " | " + repr(e),
                "n_atoms": row.get("n_atoms", 0),
                "n_heavy": row.get("n_heavy", 0),
                "n_bonds": row.get("n_bonds", 0),
                "formula": row.get("formula", ""),
                "mol_wt": row.get("mol_wt", nan()),
            })
            row = failed

        rows.append(row)

    fields = [
        "experiment_id", "mode", "lambda_global", "lambda_local", "sample",
        "valid_rdkit_read", "sanitize_status", "sanitize_message",
        "n_atoms", "n_heavy", "n_bonds", "formula", "mol_wt",
        "assignment_ok", "assignment_message", "local_element_match_ok",
        "n_fixed_expected", "fixed_atom_names", "fixed_ref_indices_0based",
        "best_fixed_geom_rmsd_A", "mean_fixed_atom_dist_A", "max_fixed_atom_dist_A",
        "n_fixed_atoms_under_0p2A", "n_fixed_atoms_under_0p5A", "n_fixed_atoms_under_1p0A",
        "local_pass_rmsd_0p2A", "local_pass_all_atoms_0p2A",
        "matched_generated_heavy_indices_0based", "matched_generated_symbols", "matched_distances_A",
        "centroid_x", "centroid_y", "centroid_z",
        "centroid_dist_to_A", "centroid_dist_to_B", "closer_centroid",
        "shape_descriptor_delta_to_A", "shape_descriptor_delta_to_B", "closer_shape_descriptor",
        "rgyr", "rgyr_A", "rgyr_B", "dmax", "dmax_A", "dmax_B",
        "shape_tanimoto_dist_to_A", "shape_tanimoto_dist_to_B", "closer_shape_tanimoto",
        "shape_protrude_dist_to_A", "shape_protrude_dist_to_B", "closer_shape_protrude",
        "b_closer_than_A_shape_tanimoto", "b_closer_than_A_shape_protrude",
        "tanimoto_to_A", "tanimoto_to_B", "closer_chem",
        "dual_candidate_vs_A_strict",
        "sa_score", "nearest_neighbor_tanimoto_in_batch",
        "posecheck_clash_score", "posecheck_strain_energy",
        "docking_vina_score", "docking_gnina_score",
    ]

    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row_to_clean_dict(row, fields))

    n_total = len(rows)
    n_read_fail = sum(not bool(r["valid_rdkit_read"]) for r in rows)

    summary = {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "lambda_global": args.lambda_global,
        "lambda_local": args.lambda_local,
        "generated_sdf": args.gen_sdf,
        "A_sdf": args.A_sdf,
        "B_sdf": args.B_sdf,
        "fixed_json": args.fixed_json,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "phase0_official_eval_v2",
        "notes": {
            "success_against_baseline": "Not computed here. Compare raw B-shape distances against lambda_global=0 in a separate aggregation script.",
            "dual_candidate_vs_A_strict": "Informative only: local pass plus B closer than A by both RDKit shape distances.",
            "local_pass_all_atoms_0p2A": "Primary local criterion: every fixed atom must be matched under 0.2 A with matching element.",
            "local_pass_rmsd_0p2A": "Secondary local criterion: aggregate fixed-block RMSD <= 0.2 A.",
        },
        "thresholds": {
            "local_rmsd_threshold_A": LOCAL_RMSD_THRESHOLD_A,
            "local_atom_distance_threshold_A": LOCAL_ATOM_DISTANCE_THRESHOLD_A,
        },
        "reference": {
            "A_sanitize_status": A_sanitize,
            "A_sanitize_message": A_sanitize_msg,
            "B_sanitize_status": B_sanitize,
            "B_sanitize_message": B_sanitize_msg,
            "A_heavy_atoms": A_mol.GetNumHeavyAtoms(),
            "B_heavy_atoms": B_mol.GetNumHeavyAtoms(),
            "n_fixed_atoms": len(ref_fixed_coords),
            "fixed_atom_names": fixed_names,
            "fixed_atom_symbols": ref_fixed_symbols,
        },
        "counts": {
            "n_sdf_records_total": n_total,
            "n_rdkit_read_fail": n_read_fail,
            "n_valid_rdkit_read": sum(bool(r["valid_rdkit_read"]) for r in rows),
            "n_sanitize_ok": sum(r["sanitize_status"] == "OK" for r in rows),
            "n_assignment_ok": sum(bool(r["assignment_ok"]) for r in rows),
            "n_local_element_match_ok": sum(bool(r["local_element_match_ok"]) for r in rows),
            "n_local_pass_rmsd_0p2A": sum(bool(r["local_pass_rmsd_0p2A"]) for r in rows),
            "n_local_pass_all_atoms_0p2A": sum(bool(r["local_pass_all_atoms_0p2A"]) for r in rows),
            "n_b_closer_than_A_shape_tanimoto": sum(bool(r["b_closer_than_A_shape_tanimoto"]) for r in rows),
            "n_b_closer_than_A_shape_protrude": sum(bool(r["b_closer_than_A_shape_protrude"]) for r in rows),
            "n_dual_candidate_vs_A_strict": sum(bool(r["dual_candidate_vs_A_strict"]) for r in rows),
        },
        "means_valid_rows": {
            "mean_best_fixed_geom_rmsd_A": mean_of([r for r in rows if r["valid_rdkit_read"]], "best_fixed_geom_rmsd_A"),
            "mean_shape_tanimoto_dist_to_A": mean_of([r for r in rows if r["valid_rdkit_read"]], "shape_tanimoto_dist_to_A"),
            "mean_shape_tanimoto_dist_to_B": mean_of([r for r in rows if r["valid_rdkit_read"]], "shape_tanimoto_dist_to_B"),
            "mean_shape_protrude_dist_to_A": mean_of([r for r in rows if r["valid_rdkit_read"]], "shape_protrude_dist_to_A"),
            "mean_shape_protrude_dist_to_B": mean_of([r for r in rows if r["valid_rdkit_read"]], "shape_protrude_dist_to_B"),
        },
        "outputs": {
            "metrics_tsv": str(out_tsv),
            "summary_json": str(out_json),
            "report_md": str(out_md),
        },
    }

    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    def md_table(rows_for_table, selected_fields):
        lines = []
        lines.append("| " + " | ".join(selected_fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(selected_fields)) + " |")
        for r in rows_for_table:
            vals = []
            for k in selected_fields:
                v = r.get(k, "")
                if isinstance(v, float):
                    vals.append("NaN" if math.isnan(v) else f"{v:.4f}")
                elif isinstance(v, bool):
                    vals.append(bool_str(v))
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md = []
    md.append(f"# Official Phase 0 evaluation v2 — {args.experiment_id}\n")
    md.append(f"Created: `{summary['created_at']}`\n")
    md.append("## Configuration\n")
    md.append("```text")
    md.append(f"mode = {args.mode}")
    md.append(f"lambda_global = {args.lambda_global}")
    md.append(f"lambda_local = {args.lambda_local}")
    md.append(f"A_sdf = {args.A_sdf}")
    md.append(f"B_sdf = {args.B_sdf}")
    md.append(f"gen_sdf = {args.gen_sdf}")
    md.append(f"fixed_json = {args.fixed_json}")
    md.append("```\n")
    md.append("## Summary counts\n")
    md.append("```json")
    md.append(json.dumps(summary["counts"], indent=2, sort_keys=True))
    md.append("```\n")
    md.append("## Mean metrics over valid rows\n")
    md.append("```json")
    md.append(json.dumps(summary["means_valid_rows"], indent=2, sort_keys=True))
    md.append("```\n")
    md.append("## Interpretation note\n")
    md.append("This evaluator does not decide success relative to baseline. It emits raw distances to A and B. Future lambda_global conditions must be compared against the lambda_global=0 baseline by an aggregation script.\n\n")
    md.append("## Core per-molecule metrics\n")
    md.append(md_table(rows, [
        "sample",
        "valid_rdkit_read",
        "n_heavy",
        "local_element_match_ok",
        "best_fixed_geom_rmsd_A",
        "n_fixed_atoms_under_0p2A",
        "local_pass_all_atoms_0p2A",
        "shape_tanimoto_dist_to_A",
        "shape_tanimoto_dist_to_B",
        "closer_shape_tanimoto",
        "shape_protrude_dist_to_A",
        "shape_protrude_dist_to_B",
        "closer_shape_protrude",
        "dual_candidate_vs_A_strict",
    ]))
    md.append("\n")
    out_md.write_text("\n".join(md))

    print("OFFICIAL_PHASE0_EVAL_V2_DONE")
    print(f"sdf_records_total={n_total}")
    print(f"rdkit_read_fail={n_read_fail}")
    print(f"metrics_tsv={out_tsv}")
    print(f"summary_json={out_json}")
    print(f"report_md={out_md}")
    print("counts=" + json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
