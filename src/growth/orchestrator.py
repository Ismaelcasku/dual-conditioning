#!/usr/bin/env python3
"""Orquestador de UNA trayectoria de crecimiento estadificado.

VERSION 2 (Campana 1 / phase3): identico a la version validada de phase2 EXCEPTO
que n_samples se recibe como argumento explicito (--n_samples) en vez de
derivarse de la rama. Esto habilita la tercera rama A10:

  A1  (branch=A, n_samples=1):  deriva neutral, 1 muestra/etapa, seleccion B-blind.
  A10 (branch=A, n_samples=10): best-of-10 B-blind, 10 muestras/etapa, seleccion
                                por tamano de anchor SIN mirar B. Control del
                                confusor best-of-10.
  B   (branch=B, n_samples=10): dirigida a B, 10 muestras/etapa.

La LOGICA DE SELECCION no cambia: branch=A usa max(anchor_heavy) (estructural,
B-blind) para A1 y A10 por igual; branch=B usa progreso dirigido a B.
La distincion A1 vs A10 vive solo en n_samples y en la ruta de salida.

Criterio de parada: por n_stages (SIN cambios respecto a phase2).
Criterio de muerte: sin anchor valido -> death. La muerte es OBSERVABLE.

Salida (en el dir de la trayectoria):
  stage_XX_generated.sdf, stage_XX_scaffold.sdf, trajectory.tsv
"""

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdShapeHelpers

PROJECT = os.environ.get("DC_PROJECT_ROOT", "/mnt/proyecto")
CKPT = f"{PROJECT}/artifacts/checkpoints/crossdocked_fullatom_cond.ckpt"
INPAINT = f"{PROJECT}/external/DiffSBDD/inpaint.py"
RUN_SEEDED = f"{PROJECT}/src/generation/run_inpaint_seeded.py"
AUDITOR_PATH = f"{PROJECT}/src/analysis/single_shot/audit_single_shot.py"

RESAMPLINGS = 5
TIMESTEPS = 50
GUIDE_R = 1.7
GUIDE_CLIP = 1.0
ANCHOR_TOL = 0.2

PAIR_TO_A = {"x0434_x2193": "x0434", "x0874_x1093": "x0874"}
WARHEAD_NAMES = {
    "x0874_x1093": ["C02", "C04", "C05", "C06", "C07", "C08", "C09"],
    "x0434_x2193": ["C4", "C5", "C6", "N", "C1", "C2", "C3"],
}


def load_auditor():
    spec = importlib.util.spec_from_file_location("aud", AUDITOR_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_inpaint(seed, pdbfile, ref_ligand, fix_atoms_arg, outfile,
                n_samples, add_n_nodes, b_ligand, is_first_stage):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT}/src:{PROJECT}/external/DiffSBDD:" + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = str(seed)
    env["DC_GUIDE_SPACE"] = "x0"
    env["DC_GUIDE_FIELD"] = "shape"
    env["DC_GUIDE_ALPHA"] = "0.3"
    env["DC_GUIDE_DEBUG"] = "0"

    cmd = [
        "python", RUN_SEEDED, "--seed", str(seed), INPAINT, CKPT,
        "--pdbfile", pdbfile,
        "--ref_ligand", ref_ligand,
        "--fix_atoms", *fix_atoms_arg,
        "--outfile", outfile,
        "--n_samples", str(n_samples),
        "--resamplings", str(RESAMPLINGS),
        "--timesteps", str(TIMESTEPS),
        "--center", "ligand",
        "--lambda_global", str(LAMBDA_GLOBAL),
        "--guide_r", str(GUIDE_R),
        "--guide_clip", str(GUIDE_CLIP),
        "--b_shape_ligand", b_ligand,
        "--add_n_nodes", str(add_n_nodes),
        "--sanitize",
    ]
    proc = subprocess.run(cmd, cwd=f"{PROJECT}/external/DiffSBDD",
                          env=env, capture_output=True, text=True)
    ok = proc.returncode == 0 and Path(outfile).is_file() and Path(outfile).stat().st_size > 0
    return ok, proc.stderr


def audit_record(aud, mol_raw, reference_a, reference_b, fixed_indices):
    mol, status = aud.sanitize_record(mol_raw)
    if mol is None or status != "OK" or mol.GetNumConformers() == 0:
        return None

    fragments = aud.fragment_data(mol)
    assignments = aud.match_fixed_atoms(reference_a, mol, fixed_indices)
    if len(assignments) != len(fixed_indices):
        return {"anchor_valid": False}

    matched = [gi for _, gi, _ in assignments]
    dists = [d for _, _, d in assignments]
    frag_ids = {fragments["atom_to_fragment"][gi] for gi in matched}

    if len(frag_ids) != 1 or max(dists) > ANCHOR_TOL:
        return {"anchor_valid": False}

    anchor_id = frag_ids.pop()
    anchor_mol = fragments["fragment_molecules"][anchor_id]
    anchor_heavy = fragments["heavy_counts"][anchor_id]

    try:
        tani_B = float(rdShapeHelpers.ShapeTanimotoDist(anchor_mol, reference_b, ignoreHs=True))
        prot_B = float(rdShapeHelpers.ShapeProtrudeDist(anchor_mol, reference_b, ignoreHs=True))
    except Exception:
        tani_B, prot_B = float("nan"), float("nan")

    return {
        "anchor_valid": True,
        "anchor_mol": anchor_mol,
        "anchor_heavy": anchor_heavy,
        "anchor_max_drift": max(dists),
        "n_heavy_fragments": len(fragments["heavy_fragment_ids"]),
        "connected": len(fragments["heavy_fragment_ids"]) == 1,
        "tani_B": tani_B,
        "prot_B": prot_B,
    }


def select_scaffold(branch, audited, prev_tani_B, prev_prot_B):
    """branch=A: estructural B-blind (sirve A1 y A10). branch=B: dirigida."""
    valid = [a for a in audited if a and a.get("anchor_valid")]
    if not valid:
        return None, "death_no_anchor"

    if branch == "A":
        best = max(valid, key=lambda a: a["anchor_heavy"])
        return best, "ok_structural"

    if prev_tani_B is None:
        best = min(valid, key=lambda a: (a["tani_B"] + a["prot_B"]) / 2.0)
        return best, "ok_first_stage"

    progressing = [
        a for a in valid
        if a["tani_B"] <= prev_tani_B and a["prot_B"] <= prev_prot_B
    ]
    if not progressing:
        return None, "death_no_progress"
    best = min(progressing, key=lambda a: (a["tani_B"] + a["prot_B"]) / 2.0)
    return best, "ok_progress"


def main():
    global LAMBDA_GLOBAL

    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(PAIR_TO_A))
    ap.add_argument("--branch", required=True, choices=["A", "B"])
    ap.add_argument("--n_samples", type=int, required=True)  # NUEVO: explicito (1 o 10)
    ap.add_argument("--add_n", type=int, required=True)
    ap.add_argument("--n_stages", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lambda_global", type=float, required=True)
    ap.add_argument("--pdbfile", required=True)
    ap.add_argument("--b_ligand", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    LAMBDA_GLOBAL = args.lambda_global
    n_samples = args.n_samples  # CAMBIO: del arg, no derivado de branch

    aud = load_auditor()
    a_local = PAIR_TO_A[args.pair]
    fixed_indices = aud.FIXED_INDICES[a_local]
    reference_a = aud.load_molecule(Path(aud.A_SDF[a_local]))
    reference_b = aud.load_molecule(Path(aud.B_SDF[args.pair.split("_")[1]]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_rows = []

    fix_atoms_arg = WARHEAD_NAMES[args.pair]
    scaffold_is_sdf = False
    prev_tani_B, prev_prot_B = None, None

    for stage in range(1, args.n_stages + 1):
        gen_sdf = str(out_dir / f"stage_{stage:02d}_generated.sdf")

        ok, err = run_inpaint(
            seed=args.seed + stage,
            pdbfile=args.pdbfile,
            ref_ligand="A:404",
            fix_atoms_arg=fix_atoms_arg if not scaffold_is_sdf else [fix_atoms_arg],
            outfile=gen_sdf,
            n_samples=n_samples,
            add_n_nodes=args.add_n,
            b_ligand=args.b_ligand,
            is_first_stage=(stage == 1),
        )
        if not ok:
            traj_rows.append({"stage": stage, "gate": "death_generation_failed",
                              "n_samples": n_samples})
            print(f"[stage {stage}] generation FAILED: {err[:300]}", flush=True)
            break

        records = list(Chem.SDMolSupplier(gen_sdf, sanitize=False, removeHs=False))
        audited = [audit_record(aud, r, reference_a, reference_b, fixed_indices)
                   for r in records]

        chosen, gate = select_scaffold(args.branch, audited, prev_tani_B, prev_prot_B)

        n_valid = sum(1 for a in audited if a and a.get("anchor_valid"))
        row = {
            "stage": stage,
            "n_samples": len(records),
            "n_anchor_valid": n_valid,
            "gate": gate,
        }

        if chosen is None:
            traj_rows.append(row)
            print(f"[stage {stage}] {gate} -> trajectory ends", flush=True)
            break

        scaffold_sdf = str(out_dir / f"stage_{stage:02d}_scaffold.sdf")
        w = Chem.SDWriter(scaffold_sdf)
        w.write(chosen["anchor_mol"])
        w.close()

        row.update({
            "anchor_heavy": chosen["anchor_heavy"],
            "anchor_max_drift": round(chosen["anchor_max_drift"], 4),
            "connected": chosen["connected"],
            "n_heavy_fragments": chosen["n_heavy_fragments"],
            "tani_B": round(chosen["tani_B"], 4),
            "prot_B": round(chosen["prot_B"], 4),
            "scaffold_sdf": scaffold_sdf,
        })
        traj_rows.append(row)
        print(f"[stage {stage}] {gate} anchor={chosen['anchor_heavy']} "
              f"taniB={chosen['tani_B']:.3f} protB={chosen['prot_B']:.3f}", flush=True)

        fix_atoms_arg = scaffold_sdf
        scaffold_is_sdf = True
        prev_tani_B, prev_prot_B = chosen["tani_B"], chosen["prot_B"]

    all_fields = ["stage", "n_samples", "n_anchor_valid", "gate", "anchor_heavy",
                  "anchor_max_drift", "connected", "n_heavy_fragments",
                  "tani_B", "prot_B", "scaffold_sdf"]
    with (out_dir / "trajectory.tsv").open("w", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=all_fields, delimiter="\t", extrasaction="ignore")
        wr.writeheader()
        wr.writerows(traj_rows)

    print(f"TRAJECTORY_DONE stages_completed={len([r for r in traj_rows if r.get('scaffold_sdf')])} "
          f"out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
