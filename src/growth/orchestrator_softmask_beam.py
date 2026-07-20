#!/usr/bin/env python3
"""Greedy/beam search over centered, order-safe soft-scaffold rigidity.

The requested increment is fixed at four atoms. Stage 1 uses hard inpainting
of the seven-atom warhead; later stages search rho in {1, .75, .5, .25, 0}.
Children are canonicalized as [parent in parent order] + [new atoms] before
the prefix-based hard/soft masks are constructed for the next stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers

from softmask_atom_order import AtomOrderError, canonicalize_rdkit_mol

import orchestrator_beam as base

PROJECT = os.environ.get("DC_PROJECT_ROOT", "/mnt/proyecto")
CKPT = os.environ.get(
    "DC_CHECKPOINT",
    f"{PROJECT}/artifacts/checkpoints/crossdocked_fullatom_cond.ckpt",
)
INPAINT_SOFT = os.environ.get(
    "DC_INPAINT_SOFTMASK",
    f"{PROJECT}/external/DiffSBDD/inpaint_softmask.py",
)
RUN_SEEDED = getattr(base, "RUN_SEEDED", f"{PROJECT}/src/generation/run_inpaint_seeded.py")

ADD_N = 4
RHO_VALUES = [1.0, 0.75, 0.50, 0.25, 0.0]
HARD_FIXED_COUNT = 7
SAMPLES_PER_ACTION = 10
N_GPUS = getattr(base, "N_GPUS", 3)
MAX_STAGES = getattr(base, "MAX_STAGES", 8)

PAIR_TO_A = base.PAIR_TO_A
TARGET_ATOMS = base.TARGET_ATOMS
TARGET_TOL = base.TARGET_TOL
WARHEAD_NAMES = base.WARHEAD_NAMES
ANCHOR_TOL = base.ANCHOR_TOL
EPS_TRADEOFF = base.EPS_TRADEOFF
CHAMFER_MIN = base.CHAMFER_MIN


def rho_tag(rho: float) -> str:
    return f"{int(round(float(rho) * 100)):03d}"


def load_parent_info(path: str | None):
    if path is None:
        return None
    supplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError(f"Could not read parent scaffold: {path}")
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=float)
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=int
    )
    if np.any(atomic_numbers <= 1):
        raise RuntimeError(
            f"Parent scaffold contains hydrogens; prefix tracking expects heavy-only SDF: {path}"
        )
    return {
        "mol": mol,
        "coords": coords,
        "atomic_numbers": atomic_numbers,
        "n_atoms": mol.GetNumAtoms(),
    }


def renumber_warhead_first(anchor_mol, warhead_local_indices):
    order_first = [int(i) for i in warhead_local_indices]
    if len(order_first) != HARD_FIXED_COUNT or len(set(order_first)) != HARD_FIXED_COUNT:
        raise ValueError(f"Invalid warhead local indices: {order_first}")
    used = set(order_first)
    order = order_first + [
        idx for idx in range(anchor_mol.GetNumAtoms()) if idx not in used
    ]
    return Chem.RenumberAtoms(anchor_mol, order)


def heavy_coords_excluding_prefix(mol, n_excluded=HARD_FIXED_COUNT):
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx < n_excluded or atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(idx)
        coords.append([p.x, p.y, p.z])
    return np.asarray(coords, dtype=float).reshape(-1, 3)


def audit_record_soft(
    aud,
    mol_raw,
    reference_a,
    reference_b,
    fixed_indices,
    parent_info,
):
    """Audit a child after restoring parent-first atom order."""

    mol, status = aud.sanitize_record(mol_raw)
    if mol is None or status != "OK" or mol.GetNumConformers() == 0:
        return None

    if parent_info is None:
        ref_conf = reference_a.GetConformer()
        parent_xyz = np.asarray(
            [
                list(ref_conf.GetAtomPosition(int(idx)))
                for idx in fixed_indices
            ],
            dtype=float,
        )
        parent_atomic_numbers = np.asarray(
            [
                reference_a.GetAtomWithIdx(int(idx)).GetAtomicNum()
                for idx in fixed_indices
            ],
            dtype=int,
        )
        order_failure = "warhead_match"
    else:
        parent_xyz = np.asarray(parent_info["coords"], dtype=float)
        parent_atomic_numbers = np.asarray(
            parent_info["atomic_numbers"],
            dtype=int,
        )
        order_failure = "parent_match"

    try:
        mol, parent_match = canonicalize_rdkit_mol(
            mol,
            parent_xyz,
            parent_atomic_numbers,
            hard_count=HARD_FIXED_COUNT,
            hard_tolerance=ANCHOR_TOL,
            soft_max_distance=None,
        )
    except AtomOrderError as exc:
        return {
            "anchor_valid": False,
            "reject_reason": order_failure,
            "order_error": str(exc),
        }

    fragments = aud.fragment_data(mol)
    assignments = aud.match_fixed_atoms(reference_a, mol, fixed_indices)
    if len(assignments) != len(fixed_indices):
        return {"anchor_valid": False, "reject_reason": "warhead_match"}

    assignments = sorted(assignments, key=lambda item: item[0])
    warhead_global = [int(generated_idx) for _, generated_idx, _ in assignments]
    warhead_dists = [float(distance) for _, _, distance in assignments]
    if max(warhead_dists) > ANCHOR_TOL:
        return {"anchor_valid": False, "reject_reason": "warhead_drift"}

    if warhead_global != list(range(HARD_FIXED_COUNT)):
        return {
            "anchor_valid": False,
            "reject_reason": "canonicalization_failed",
        }

    frag_ids = {fragments["atom_to_fragment"][idx] for idx in warhead_global}
    if len(frag_ids) != 1:
        return {"anchor_valid": False, "reject_reason": "warhead_split"}

    anchor_id = frag_ids.pop()
    anchor_global = tuple(int(i) for i in fragments["atom_fragments"][anchor_id])
    anchor_global_set = set(anchor_global)
    global_to_local = {
        global_idx: local_idx
        for local_idx, global_idx in enumerate(anchor_global)
    }
    warhead_local = [global_to_local[idx] for idx in warhead_global]

    parent_rmsd = 0.0
    soft_scaffold_rmsd = 0.0
    parent_max_displacement = 0.0

    if parent_info is not None:
        n_parent = int(parent_info["n_atoms"])
        if mol.GetNumAtoms() < n_parent:
            return {"anchor_valid": False, "reject_reason": "parent_truncated"}

        prefix = set(range(n_parent))
        if not prefix.issubset(anchor_global_set):
            return {"anchor_valid": False, "reject_reason": "parent_left_anchor"}

        generated_types = np.asarray(
            [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_parent)],
            dtype=int,
        )
        if not np.array_equal(generated_types, parent_info["atomic_numbers"]):
            return {"anchor_valid": False, "reject_reason": "parent_type_changed"}

        generated_coords = np.asarray(
            mol.GetConformer().GetPositions()[:n_parent],
            dtype=float,
        )
        displacement = generated_coords - parent_info["coords"]
        per_atom = np.sqrt((displacement**2).sum(axis=1))
        parent_rmsd = float(np.sqrt((per_atom**2).mean()))
        parent_max_displacement = float(per_atom.max())

        if n_parent > HARD_FIXED_COUNT:
            soft_scaffold_rmsd = float(
                np.sqrt((per_atom[HARD_FIXED_COUNT:] ** 2).mean())
            )

    anchor_mol_raw = fragments["fragment_molecules"][anchor_id]
    anchor_mol = renumber_warhead_first(anchor_mol_raw, warhead_local)
    anchor_heavy = sum(
        atom.GetAtomicNum() > 1 for atom in anchor_mol.GetAtoms()
    )

    try:
        tani_b = float(
            rdShapeHelpers.ShapeTanimotoDist(
                anchor_mol,
                reference_b,
                ignoreHs=True,
            )
        )
        prot_b = float(
            rdShapeHelpers.ShapeProtrudeDist(
                anchor_mol,
                reference_b,
                ignoreHs=True,
                allowReordering=False,
            )
        )
    except Exception:
        tani_b, prot_b = float("nan"), float("nan")

    return {
        "anchor_valid": True,
        "anchor_mol": anchor_mol,
        "anchor_heavy": int(anchor_heavy),
        "anchor_max_drift": max(warhead_dists),
        "connected": len(fragments["heavy_fragment_ids"]) == 1,
        "n_heavy_fragments": len(fragments["heavy_fragment_ids"]),
        "tani_B": tani_b,
        "prot_B": prot_b,
        "warhead_rmsd": float(np.sqrt(np.mean(np.square(warhead_dists)))),
        "parent_rmsd": parent_rmsd,
        "soft_scaffold_rmsd": soft_scaffold_rmsd,
        "parent_max_displacement": parent_max_displacement,
        "parent_match_rmsd": parent_match.rmsd,
        "parent_match_max_distance": parent_match.max_distance,
        "grown_xyz": heavy_coords_excluding_prefix(anchor_mol),
    }


def build_inpaint_cmd(
    seed,
    fix_atoms_arg,
    outfile,
    rho,
    b_ligand,
    lambda_global,
    scaffold_is_sdf,
    pdbfile,
    debug_path,
):
    fixed_args = fix_atoms_arg if not scaffold_is_sdf else [fix_atoms_arg]
    return [
        "python",
        RUN_SEEDED,
        "--seed",
        str(seed),
        INPAINT_SOFT,
        CKPT,
        "--pdbfile",
        pdbfile,
        "--ref_ligand",
        "A:404",
        "--fix_atoms",
        *fixed_args,
        "--outfile",
        outfile,
        "--n_samples",
        str(SAMPLES_PER_ACTION),
        "--resamplings",
        str(base.RESAMPLINGS),
        "--timesteps",
        str(base.TIMESTEPS),
        "--center",
        "ligand",
        "--lambda_global",
        str(lambda_global),
        "--guide_r",
        str(base.GUIDE_R),
        "--guide_clip",
        str(base.GUIDE_CLIP),
        "--b_shape_ligand",
        b_ligand,
        "--add_n_nodes",
        str(ADD_N),
        "--soft_scaffold_rho",
        str(rho),
        "--hard_fixed_count",
        str(HARD_FIXED_COUNT),
        "--softmask_debug",
        debug_path,
        "--sanitize",
    ]


def run_one_generation(
    gpu_id,
    seed,
    fix_atoms_arg,
    outfile,
    rho,
    b_ligand,
    lambda_global,
    scaffold_is_sdf,
    pdbfile,
    debug_path,
):
    env = dict(os.environ)
    vendor = str(Path(INPAINT_SOFT).resolve().parent)
    env["PYTHONPATH"] = (
        f"{PROJECT}/src:{PROJECT}/src/growth:{vendor}:"
        + env.get("PYTHONPATH", "")
    )
    env["PYTHONHASHSEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DC_GUIDE_SPACE"] = "x0"
    env["DC_GUIDE_FIELD"] = "shape"
    env["DC_GUIDE_ALPHA"] = "0.3"
    env["DC_GUIDE_DEBUG"] = "0"

    cmd = build_inpaint_cmd(
        seed,
        fix_atoms_arg,
        outfile,
        rho,
        b_ligand,
        lambda_global,
        scaffold_is_sdf,
        pdbfile,
        debug_path,
    )
    proc = subprocess.run(
        cmd,
        cwd=vendor,
        env=env,
        capture_output=True,
        text=True,
    )
    if (
        proc.returncode != 0
        or not Path(outfile).is_file()
        or Path(outfile).stat().st_size == 0
    ):
        raise RuntimeError(
            f"softmask_generation_failed gpu={gpu_id} seed={seed} "
            f"rho={rho} rc={proc.returncode} stderr={proc.stderr[-1200:]}"
        )
    return outfile


def generate_all_beams(
    beams,
    stage,
    out_dir,
    pdbfile,
    b_ligand,
    lambda_global,
    base_seed,
):
    actions = [1.0] if stage == 1 else RHO_VALUES
    jobs = []
    job_index = 0
    for beam_idx, beam in enumerate(beams):
        # Same seed for all rho values from a given parent/stage: paired noise.
        paired_seed = base_seed + stage * 100 + beam_idx * 7
        for rho in actions:
            gpu_id = job_index % N_GPUS
            tag = rho_tag(rho)
            sdf = str(
                out_dir
                / f"stage_{stage:02d}_beam_{beam_idx}_rho{tag}_generated.sdf"
            )
            debug = str(
                out_dir / f"stage_{stage:02d}_beam_{beam_idx}_rho{tag}_raw.json"
            )
            jobs.append((beam_idx, rho, gpu_id, sdf, debug, paired_seed, beam))
            job_index += 1

    by_gpu = defaultdict(list)
    for job in jobs:
        by_gpu[job[2]].append(job)

    def run_gpu_queue(gpu_id):
        completed = []
        for beam_idx, rho, gid, sdf, debug, seed, beam in by_gpu[gpu_id]:
            run_one_generation(
                gpu_id=gid,
                seed=seed,
                fix_atoms_arg=beam["fix_atoms_arg"],
                outfile=sdf,
                rho=rho,
                b_ligand=b_ligand,
                lambda_global=lambda_global,
                scaffold_is_sdf=beam["scaffold_is_sdf"],
                pdbfile=pdbfile,
                debug_path=debug,
            )
            completed.append((beam_idx, rho, sdf))
        return completed

    results = {}
    with ThreadPoolExecutor(max_workers=N_GPUS) as executor:
        futures = [executor.submit(run_gpu_queue, gpu) for gpu in by_gpu]
        for future in futures:
            for beam_idx, rho, sdf in future.result():
                results.setdefault(beam_idx, []).append((rho, sdf))
    return results


def save_checkpoint(out_dir, stage_completed, beams, graduated, rows, dead, config):
    state = {
        "stage_completed": stage_completed,
        "config": config,
        "dead_count": dead,
        "graduated_records": graduated,
        "traj_rows": rows,
        "beams": [
            {
                "fix_atoms_sdf": b["fix_atoms_arg"] if b["scaffold_is_sdf"] else None,
                "scaffold_is_sdf": b["scaffold_is_sdf"],
                "tani_B": b["tani_B"],
                "prot_B": b["prot_B"],
                "anchor_heavy": b["anchor_heavy"],
                "lineage": b["lineage"],
                "cumulative_gain": b.get("cumulative_gain", 0.0),
                "rho_schedule": b.get("rho_schedule", [1.0]),
            }
            for b in beams
        ],
    }
    tmp = out_dir / "beam_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(out_dir / "beam_state.json")


def load_checkpoint(out_dir):
    path = out_dir / "beam_state.json"
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except Exception:
        return None
    beams = []
    for item in state.get("beams", []):
        sdf = item.get("fix_atoms_sdf")
        if not sdf or not Path(sdf).is_file():
            return None
        beams.append(
            {
                "fix_atoms_arg": sdf,
                "scaffold_is_sdf": True,
                "tani_B": item["tani_B"],
                "prot_B": item["prot_B"],
                "anchor_heavy": item["anchor_heavy"],
                "lineage": item["lineage"],
                "cumulative_gain": item.get("cumulative_gain", 0.0),
                "rho_schedule": item.get("rho_schedule", [1.0]),
            }
        )
    return (
        state["stage_completed"] + 1,
        beams,
        state.get("graduated_records", []),
        state.get("traj_rows", []),
        state.get("dead_count", 0),
    )


def write_run_status(out_dir, status, graduated, dead_count, stages_run):
    (out_dir / "run_status.json").write_text(
        json.dumps(
            {
                "run_status": status,
                "n_graduated": len(graduated),
                "dead_count": dead_count,
                "stages_run": stages_run,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=list(PAIR_TO_A))
    parser.add_argument("--k_beams", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lambda_global", type=float, required=True)
    parser.add_argument("--pdbfile", required=True)
    parser.add_argument("--b_ligand", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if args.k_beams not in (1, 3):
        raise SystemExit("k_beams must be 1 (greedy) or 3 (beam)")
    if not Path(INPAINT_SOFT).is_file():
        raise SystemExit(f"Missing alternate sampler: {INPAINT_SOFT}")

    aud = base.load_auditor()
    a_local = PAIR_TO_A[args.pair]
    fixed_indices = aud.FIXED_INDICES[a_local]
    reference_a = aud.load_molecule(Path(aud.A_SDF[a_local]))
    reference_b = aud.load_molecule(Path(aud.B_SDF[args.pair.split("_")[1]]))
    target = TARGET_ATOMS[args.pair]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graduated").mkdir(exist_ok=True)

    config = {
        "pair": args.pair,
        "k_beams": args.k_beams,
        "seed": args.seed,
        "add_n": ADD_N,
        "rho_values": RHO_VALUES,
        "hard_fixed_count": HARD_FIXED_COUNT,
        "eps": EPS_TRADEOFF,
        "chamfer_min": CHAMFER_MIN,
        "target": target,
    }

    resumed = load_checkpoint(out_dir)
    if resumed:
        start_stage, beams, graduated_records, traj_rows, dead_count = resumed
        print(
            f"RESUMED stage={start_stage} beams={len(beams)} "
            f"graduated={len(graduated_records)}",
            flush=True,
        )
    else:
        start_stage = 1
        beams = [
            {
                "fix_atoms_arg": WARHEAD_NAMES[args.pair],
                "scaffold_is_sdf": False,
                "tani_B": None,
                "prot_B": None,
                "anchor_heavy": HARD_FIXED_COUNT,
                "lineage": "root",
                "cumulative_gain": 0.0,
                "rho_schedule": [],
            }
        ]
        graduated_records = []
        traj_rows = []
        dead_count = 0

    final_status = "stopped_max_stages"

    for stage in range(start_stage, MAX_STAGES + 1):
        if not beams:
            final_status = (
                "completed_graduated" if graduated_records else "completed_all_dead"
            )
            break

        generated = generate_all_beams(
            beams,
            stage,
            out_dir,
            args.pdbfile,
            args.b_ligand,
            args.lambda_global,
            args.seed,
        )

        candidates = []
        rejection_counts = defaultdict(int)
        for beam_idx, parent in enumerate(beams):
            parent_info = (
                load_parent_info(parent["fix_atoms_arg"])
                if parent["scaffold_is_sdf"]
                else None
            )
            for rho, sdf in generated.get(beam_idx, []):
                records = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)
                for mol in records:
                    audit = audit_record_soft(
                        aud,
                        mol,
                        reference_a,
                        reference_b,
                        fixed_indices,
                        parent_info,
                    )
                    if not audit or not audit.get("anchor_valid"):
                        rejection_counts[(rho, (audit or {}).get("reject_reason", "read"))] += 1
                        continue
                    if not (
                        np.isfinite(audit["tani_B"])
                        and np.isfinite(audit["prot_B"])
                    ):
                        rejection_counts[(rho, "shape_nan")] += 1
                        continue
                    if len(audit["grown_xyz"]) == 0:
                        rejection_counts[(rho, "no_growth")] += 1
                        continue
                    if audit["anchor_heavy"] <= parent["anchor_heavy"]:
                        rejection_counts[(rho, "not_larger")] += 1
                        continue

                    if parent["tani_B"] is None:
                        passed, gain = True, 0.0
                    else:
                        passed, gain, _, _ = base.compute_gate(
                            audit["tani_B"],
                            audit["prot_B"],
                            parent["tani_B"],
                            parent["prot_B"],
                            EPS_TRADEOFF,
                        )
                    if not passed:
                        rejection_counts[(rho, "shape_gate")] += 1
                        continue

                    audit["rho"] = float(rho)
                    audit["rho_schedule"] = parent.get("rho_schedule", []) + [
                        float(rho)
                    ]
                    audit["local_gain"] = gain
                    audit["absolute_quality"] = base.absolute_quality(
                        audit["tani_B"], audit["prot_B"]
                    )
                    audit["cumulative_gain"] = (
                        parent.get("cumulative_gain", 0.0) + gain
                    )
                    audit["parent_lineage"] = parent["lineage"]
                    candidates.append(audit)

        if not candidates:
            dead_count += len(beams)
            traj_rows.append(
                {
                    "stage": stage,
                    "event": "death_no_progress",
                    "n_beams_in": len(beams),
                    "n_candidates": 0,
                    "rejection_counts": json.dumps(
                        {f"rho={k[0]}:{k[1]}": v for k, v in rejection_counts.items()},
                        sort_keys=True,
                    ),
                }
            )
            beams = []
            save_checkpoint(
                out_dir,
                stage,
                beams,
                graduated_records,
                traj_rows,
                dead_count,
                config,
            )
            final_status = (
                "completed_graduated" if graduated_records else "completed_all_dead"
            )
            break

        chosen = base.select_beams(candidates, args.k_beams, CHAMFER_MIN)
        next_beams = []
        for chosen_idx, candidate in enumerate(chosen):
            anchor_sdf = str(
                out_dir / f"stage_{stage:02d}_beam_{chosen_idx}_anchor.sdf"
            )
            candidate["anchor_mol"].SetProp(
                "DC_RHO_SCHEDULE", json.dumps(candidate["rho_schedule"])
            )
            candidate["anchor_mol"].SetIntProp("DC_N_HARD", HARD_FIXED_COUNT)
            writer = Chem.SDWriter(anchor_sdf)
            writer.write(candidate["anchor_mol"])
            writer.close()

            graduation = base.graduation_status(
                candidate["anchor_heavy"], target, TARGET_TOL
            )
            row = {
                "stage": stage,
                "beam": chosen_idx,
                "rho": candidate["rho"],
                "rho_schedule": json.dumps(candidate["rho_schedule"]),
                "add_n": ADD_N,
                "parent_lineage": candidate["parent_lineage"],
                "anchor_heavy": candidate["anchor_heavy"],
                "connected": candidate["connected"],
                "n_heavy_fragments": candidate["n_heavy_fragments"],
                "warhead_rmsd": round(candidate["warhead_rmsd"], 5),
                "parent_rmsd": round(candidate["parent_rmsd"], 5),
                "soft_scaffold_rmsd": round(
                    candidate["soft_scaffold_rmsd"], 5
                ),
                "parent_max_displacement": round(
                    candidate["parent_max_displacement"], 5
                ),
                "tani_B": round(candidate["tani_B"], 4),
                "prot_B": round(candidate["prot_B"], 4),
                "local_gain": round(candidate["local_gain"], 4),
                "absolute_quality": round(candidate["absolute_quality"], 4),
                "cumulative_gain": round(candidate["cumulative_gain"], 4),
                "event": graduation,
                "n_beams_in": len(beams),
                "n_candidates": len(candidates),
            }

            if graduation == "oversize":
                row["event"] = "rejected_oversize"
                traj_rows.append(row)
                continue

            if graduation == "graduated":
                endpoint_idx = len(graduated_records)
                final_sdf = str(out_dir / "graduated" / f"final_{endpoint_idx}.sdf")
                writer = Chem.SDWriter(final_sdf)
                writer.write(candidate["anchor_mol"])
                writer.close()
                graduated_records.append(
                    {
                        "anchor_heavy": candidate["anchor_heavy"],
                        "tani_B": candidate["tani_B"],
                        "prot_B": candidate["prot_B"],
                        "component_graduated": True,
                        "strictly_connected_graduated": bool(
                            candidate["connected"]
                        ),
                        "lineage": candidate["parent_lineage"],
                        "rho": candidate["rho"],
                        "rho_schedule": candidate["rho_schedule"],
                        "warhead_rmsd": candidate["warhead_rmsd"],
                        "parent_rmsd": candidate["parent_rmsd"],
                        "soft_scaffold_rmsd": candidate[
                            "soft_scaffold_rmsd"
                        ],
                    }
                )
                traj_rows.append(row)
            else:
                traj_rows.append(row)
                next_beams.append(
                    {
                        "fix_atoms_arg": anchor_sdf,
                        "scaffold_is_sdf": True,
                        "tani_B": candidate["tani_B"],
                        "prot_B": candidate["prot_B"],
                        "anchor_heavy": candidate["anchor_heavy"],
                        "lineage": f"{candidate['parent_lineage']}>s{stage}b{chosen_idx}",
                        "cumulative_gain": candidate["cumulative_gain"],
                        "rho_schedule": candidate["rho_schedule"],
                    }
                )

        beams = next_beams
        save_checkpoint(
            out_dir,
            stage,
            beams,
            graduated_records,
            traj_rows,
            dead_count,
            config,
        )
        print(
            f"[stage {stage}] candidates={len(candidates)} chosen={len(chosen)} "
            f"graduated={len(graduated_records)} continuing={len(beams)}",
            flush=True,
        )
        if not beams:
            final_status = (
                "completed_graduated" if graduated_records else "completed_all_dead"
            )
            break

    fields = [
        "stage",
        "beam",
        "rho",
        "rho_schedule",
        "add_n",
        "parent_lineage",
        "anchor_heavy",
        "connected",
        "n_heavy_fragments",
        "warhead_rmsd",
        "parent_rmsd",
        "soft_scaffold_rmsd",
        "parent_max_displacement",
        "tani_B",
        "prot_B",
        "local_gain",
        "absolute_quality",
        "cumulative_gain",
        "event",
        "n_beams_in",
        "n_candidates",
        "rejection_counts",
    ]
    with (out_dir / "beam_trajectory.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(traj_rows)

    stages_run = max((row.get("stage", 0) for row in traj_rows), default=0)
    write_run_status(
        out_dir, final_status, graduated_records, dead_count, stages_run
    )
    print(
        f"SOFTMASK_DONE status={final_status} graduated={len(graduated_records)} "
        f"dead={dead_count} out_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
