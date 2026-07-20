#!/usr/bin/env python3
"""Orquestador de UNA trayectoria de BEAM/GREEDY de incremento variable (Campana 2).

Corre DENTRO del contenedor Singularity. Pide --gres=gpu:3 en SLURM (el greedy
k=1 tambien puede pedir 3, usa 1-3 GPUs segun beams vivos).

VARIANTES (--k_beams):
  greedy : k=1. Un solo camino, incremento variable. Greedy real.
  beam   : k=3. Beam search real (arbol): los k mejores hijos de CUALQUIER padre.
  Ambas comparten el MISMO gate de trade-off (eps=0.01). Solo cambia k, aislando
  el efecto del beam width. (Corrige critica 1: la antigua 'monotonic' con k=3 no
  era greedy.)

DISENO (todas las correcciones de la revision incorporadas):
  - Incrementos {1,2,3,4,5}, 10 muestras/incremento.
  - Paralelismo pool-por-GPU (ThreadPoolExecutor; una excepcion en un hilo hace
    fallar el job, NO fabrica una muerte falsa -- critica 10).
  - Score: se guardan local_gain (Delta vs padre), absolute_quality y
    cumulative_gain. La SELECCION de beams ordena por absolute_quality (estado
    absoluto), NO por mejora local (critica 3). El GATE usa Delta vs padre.
  - Gate trade-off: g_T >= -eps, g_P >= -eps, F=g_T+g_P > 0. (Renombrado: es un
    gate de trade-off con tolerancia por objetivo, NO credito diferido -- critica 4.)
  - Diversidad Chamfer sobre atomos crecidos, excluye los 7 fijos, sin alinear.
    Candidatos sin crecimiento conectado se rechazan ANTES (critica 7).
  - Retencion de scaffold parental: el hijo debe contener todos los pesados del
    anchor del padre dentro de tolerancia y en el mismo componente, y crecer
    (child_heavy > parent_heavy) -- critica 7.
  - Graduacion CON TECHO: target-tol <= h <= target+tol. Los que exceden se
    marcan rejected_oversize (critica 6). Dos endpoints: component_graduated y
    strictly_connected_graduated (punto cientifico final).
  - Protrude direccional: allowReordering=False, grid/vdW declarados (critica 8).
  - Guardia NaN antes del gate (critica 9).
  - Estados de terminacion explicitos y run_status en SUCCESS file; la guarda del
    SLURM mira run_status, no solo la existencia del TSV (critica 11).
  - CHECKPOINT con REANUDACION REAL: beam_state.json se lee al arrancar y se
    reconstruye el arbol para saltar etapas completas (critica 5).

Salida:
  stage_XX_beam_<b>_add<N>_generated.sdf   (muestras brutas)
  stage_XX_beam_<b>_anchor.sdf             (anchor del beam elegido)
  beam_state.json                          (checkpoint reanudable)
  graduated/final_<i>.sdf                  (candidatos finales)
  beam_trajectory.tsv                      (una fila por (etapa, beam))
  run_status.json                          (estado final + SUCCESS)
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
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

INCREMENTS = [1, 2, 3, 4, 5]
SAMPLES_PER_INCREMENT = 10
CHAMFER_MIN = 0.75
EPS_TRADEOFF = 0.01
N_GPUS = 3

# Shape evaluator params (declarados para casar con Fase 0)
SHAPE_GRID = 0.5        # grid spacing (A)
SHAPE_VDW = 1.0         # vdW scale
RETENTION_TOL = 0.5     # A, tolerancia para considerar un atomo del padre "retenido"

PAIR_TO_A = {"x0434_x2193": "x0434", "x0874_x1093": "x0874"}
TARGET_ATOMS = {"x0434_x2193": 16, "x0874_x1093": 19}
TARGET_TOL = 1
WARHEAD_NAMES = {
    "x0874_x1093": ["C02", "C04", "C05", "C06", "C07", "C08", "C09"],
    "x0434_x2193": ["C4", "C5", "C6", "N", "C1", "C2", "C3"],
}
MAX_STAGES = 8


def load_auditor():
    spec = importlib.util.spec_from_file_location("aud", AUDITOR_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ===========================================================================
# LOGICA PURA (testeable sin cluster)
# ===========================================================================

def chamfer_symmetric(G_i, G_j):
    """Chamfer simetrico entre dos nubes (Nx3, Mx3). Sin correspondencia ni
    alineamiento. Devuelve None si alguna esta vacia (el llamador ya deberia
    haber rechazado candidatos sin crecimiento -- critica 7)."""
    if len(G_i) == 0 or len(G_j) == 0:
        return None
    diff = G_i[:, None, :] - G_j[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())


def compute_gate(cand_tani, cand_prot, parent_tani, parent_prot, eps):
    """Gate de trade-off con tolerancia por objetivo. Convencion de DISTANCIAS
    (mas bajo = mas cerca de B). Devuelve (ok, F, g_T, g_P)."""
    g_T = parent_tani - cand_tani
    g_P = parent_prot - cand_prot
    F = g_T + g_P
    ok = (g_T >= -eps) and (g_P >= -eps) and (F > 0.0)
    return ok, F, g_T, g_P


def absolute_quality(tani, prot):
    """Calidad absoluta del estado: -(dist). Mas alto = mejor (mas cerca de B)."""
    return -(tani + prot)


def select_beams(candidates, k, chamfer_min):
    """Selecciona hasta k candidatos ordenando por ABSOLUTE_QUALITY (no por
    mejora local -- critica 3), con poda por diversidad Chamfer sobre atomos
    crecidos. candidates: dicts con 'absolute_quality', 'grown_xyz'."""
    ordered = sorted(candidates, key=lambda c: c["absolute_quality"], reverse=True)
    chosen = []
    for c in ordered:
        if len(chosen) >= k:
            break
        diverse = True
        for s in chosen:
            d = chamfer_symmetric(c["grown_xyz"], s["grown_xyz"])
            if d is not None and d <= chamfer_min:
                diverse = False
                break
        if diverse:
            chosen.append(c)
    return chosen


def graduation_status(anchor_heavy, target, tol):
    """Estado de graduacion CON techo (critica 6).
    Devuelve: 'graduated' | 'growing' | 'oversize'."""
    if anchor_heavy > target + tol:
        return "oversize"
    if anchor_heavy >= target - tol:
        return "graduated"
    return "growing"


def scaffold_retained(parent_heavy_xyz, child_anchor_xyz, tol):
    """Verifica que todos los pesados del anchor del padre esten en el hijo
    dentro de tolerancia (retencion acumulativa -- critica 7). parent_heavy_xyz
    y child_anchor_xyz son Nx3. Devuelve True si cada atomo del padre tiene un
    vecino del hijo a <= tol."""
    if len(parent_heavy_xyz) == 0:
        return True  # padre = warhead puro (etapa 1), nada extra que retener
    if len(child_anchor_xyz) == 0:
        return False
    diff = parent_heavy_xyz[:, None, :] - child_anchor_xyz[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return bool((d.min(axis=1) <= tol).all())


# ===========================================================================
# GENERACION Y AUDITORIA (necesitan cluster)
# ===========================================================================

def build_inpaint_cmd(seed, fix_atoms_arg, outfile, add_n_nodes, b_ligand,
                      lambda_global, scaffold_is_sdf, pdbfile):
    fa = fix_atoms_arg if not scaffold_is_sdf else [fix_atoms_arg]
    return [
        "python", RUN_SEEDED, "--seed", str(seed), INPAINT, CKPT,
        "--pdbfile", pdbfile, "--ref_ligand", "A:404",
        "--fix_atoms", *fa, "--outfile", outfile,
        "--n_samples", str(SAMPLES_PER_INCREMENT),
        "--resamplings", str(RESAMPLINGS), "--timesteps", str(TIMESTEPS),
        "--center", "ligand", "--lambda_global", str(lambda_global),
        "--guide_r", str(GUIDE_R), "--guide_clip", str(GUIDE_CLIP),
        "--b_shape_ligand", b_ligand, "--add_n_nodes", str(add_n_nodes),
        "--sanitize",
    ]


def run_one_generation(gpu_id, seed, fix_atoms_arg, outfile, add_n_nodes,
                       b_ligand, lambda_global, scaffold_is_sdf, pdbfile):
    """Ejecuta UNA generacion en una GPU. Lanza excepcion si falla tecnicamente
    (critica 10: no se enmascara como muerte)."""
    import subprocess
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT}/src:{PROJECT}/external/DiffSBDD:" + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DC_GUIDE_SPACE"] = "x0"
    env["DC_GUIDE_FIELD"] = "shape"
    env["DC_GUIDE_ALPHA"] = "0.3"
    env["DC_GUIDE_DEBUG"] = "0"
    cmd = build_inpaint_cmd(seed, fix_atoms_arg, outfile, add_n_nodes, b_ligand,
                            lambda_global, scaffold_is_sdf, pdbfile)
    proc = subprocess.run(cmd, cwd=f"{PROJECT}/external/DiffSBDD",
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(outfile).is_file() or Path(outfile).stat().st_size == 0:
        raise RuntimeError(f"generation_failed gpu={gpu_id} seed={seed} "
                           f"add_n={add_n_nodes} rc={proc.returncode} "
                           f"stderr={proc.stderr[:300]}")
    return outfile


def generate_all_beams(beams, stage, out_dir, pdbfile, b_ligand, lambda_global,
                       base_seed):
    """Pool por GPU con ThreadPoolExecutor. Cada trabajo (beam,add_n) va a una
    GPU (round-robin). future.result() propaga excepciones tecnicas al hilo
    principal -> el job FALLA en vez de fabricar muerte (critica 10)."""
    jobs = []  # (beam_idx, add_n, gpu_id, sdf, seed, beam)
    idx = 0
    for bi, beam in enumerate(beams):
        for add_n in INCREMENTS:
            gpu_id = idx % N_GPUS
            sdf = str(out_dir / f"stage_{stage:02d}_beam_{bi}_add{add_n}_generated.sdf")
            seed = base_seed + stage * 100 + bi * 7 + add_n
            jobs.append((bi, add_n, gpu_id, sdf, seed, beam))
            idx += 1

    results = {}
    # ejecutar: max N_GPUS en paralelo, cada GPU serializa su cola por el
    # round-robin de gpu_id + max_workers=N_GPUS agrupando por gpu.
    # Para garantizar <=1 proceso por GPU, agrupamos explicitamente por gpu_id.
    from collections import defaultdict
    by_gpu = defaultdict(list)
    for j in jobs:
        by_gpu[j[2]].append(j)

    def run_gpu_queue(gpu_id):
        out = []
        for (bi, add_n, gid, sdf, seed, beam) in by_gpu[gpu_id]:
            run_one_generation(  # lanza excepcion si falla
                gpu_id=gid, seed=seed, fix_atoms_arg=beam["fix_atoms_arg"],
                outfile=sdf, add_n_nodes=add_n, b_ligand=b_ligand,
                lambda_global=lambda_global,
                scaffold_is_sdf=beam["scaffold_is_sdf"], pdbfile=pdbfile)
            out.append((bi, add_n, sdf))
        return out

    with ThreadPoolExecutor(max_workers=N_GPUS) as ex:
        futures = {ex.submit(run_gpu_queue, g): g for g in by_gpu}
        for fut in futures:
            for (bi, add_n, sdf) in fut.result():  # propaga excepciones
                results.setdefault(bi, []).append((add_n, sdf))
    return results


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
        tani_B = float(rdShapeHelpers.ShapeTanimotoDist(
            anchor_mol, reference_b, ignoreHs=True))
        prot_B = float(rdShapeHelpers.ShapeProtrudeDist(
            anchor_mol, reference_b, ignoreHs=True, allowReordering=False))
    except Exception:
        tani_B, prot_B = float("nan"), float("nan")
    return {
        "anchor_valid": True, "anchor_mol": anchor_mol, "anchor_heavy": anchor_heavy,
        "anchor_max_drift": max(dists),
        "connected": len(fragments["heavy_fragment_ids"]) == 1,
        "n_heavy_fragments": len(fragments["heavy_fragment_ids"]),
        "tani_B": tani_B, "prot_B": prot_B,
        "_fixed_global": {gi for _, gi, _ in assignments},
    }


def anchor_grown_coords(anchor_mol, fixed_global):
    """Coords de pesados del anchor EXCLUYENDO los fijos."""
    conf = anchor_mol.GetConformer()
    coords = []
    for atom in anchor_mol.GetAtoms():
        if atom.GetIdx() in fixed_global or atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    return np.array(coords, dtype=float) if coords else np.zeros((0, 3))


def anchor_all_heavy_coords(anchor_mol):
    """Coords de TODOS los pesados del anchor (para retencion parental)."""
    conf = anchor_mol.GetConformer()
    coords = []
    for atom in anchor_mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    return np.array(coords, dtype=float) if coords else np.zeros((0, 3))


# ===========================================================================
# CHECKPOINT / REANUDACION (critica 5)
# ===========================================================================

def save_checkpoint(out_dir, stage_completed, beams, graduated_records,
                    traj_rows, dead_count, config):
    """Guarda estado COMPLETO para reanudacion real."""
    state = {
        "stage_completed": stage_completed,
        "config": config,
        "dead_count": dead_count,
        "graduated_records": graduated_records,  # metadatos, no mols
        "traj_rows": traj_rows,
        "beams": [
            {
                "fix_atoms_sdf": b["fix_atoms_arg"] if b["scaffold_is_sdf"] else None,
                "scaffold_is_sdf": b["scaffold_is_sdf"],
                "tani_B": b["tani_B"], "prot_B": b["prot_B"],
                "anchor_heavy": b["anchor_heavy"], "lineage": b["lineage"],
                "cumulative_gain": b.get("cumulative_gain", 0.0),
            }
            for b in beams
        ],
    }
    tmp = out_dir / "beam_state.json.tmp"
    with tmp.open("w") as h:
        json.dump(state, h, indent=2)
    tmp.replace(out_dir / "beam_state.json")  # escritura atomica


def load_checkpoint(out_dir):
    """Lee checkpoint si existe. Devuelve (start_stage, beams, graduated_records,
    traj_rows, dead_count) o None si no hay o esta corrupto."""
    p = out_dir / "beam_state.json"
    if not p.is_file():
        return None
    try:
        state = json.loads(p.read_text())
    except Exception:
        return None
    beams = []
    for b in state["beams"]:
        if b["fix_atoms_sdf"] is None:
            continue  # un beam-warhead no deberia persistir a mitad; se ignora
        if not Path(b["fix_atoms_sdf"]).is_file():
            return None  # scaffold SDF perdido -> no se puede reanudar limpio
        beams.append({
            "fix_atoms_arg": b["fix_atoms_sdf"], "scaffold_is_sdf": True,
            "tani_B": b["tani_B"], "prot_B": b["prot_B"],
            "anchor_heavy": b["anchor_heavy"], "lineage": b["lineage"],
            "cumulative_gain": b.get("cumulative_gain", 0.0),
        })
    return (state["stage_completed"] + 1, beams, state["graduated_records"],
            state["traj_rows"], state["dead_count"])


def write_run_status(out_dir, status, graduated_records, dead_count, stages_run):
    with (out_dir / "run_status.json").open("w") as h:
        json.dump({
            "run_status": status,          # completed_graduated | completed_all_dead
                                           # | stopped_max_stages | failed_generation
            "n_graduated": len(graduated_records),
            "dead_count": dead_count,
            "stages_run": stages_run,
        }, h, indent=2)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(PAIR_TO_A))
    ap.add_argument("--k_beams", type=int, required=True)  # 1=greedy, 3=beam
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lambda_global", type=float, required=True)
    ap.add_argument("--pdbfile", required=True)
    ap.add_argument("--b_ligand", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    k_beams = args.k_beams
    aud = load_auditor()
    a_local = PAIR_TO_A[args.pair]
    fixed_indices = aud.FIXED_INDICES[a_local]
    reference_a = aud.load_molecule(Path(aud.A_SDF[a_local]))
    reference_b = aud.load_molecule(Path(aud.B_SDF[args.pair.split("_")[1]]))
    target = TARGET_ATOMS[args.pair]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graduated").mkdir(exist_ok=True)

    config = {"pair": args.pair, "k_beams": k_beams, "seed": args.seed,
              "increments": INCREMENTS, "eps": EPS_TRADEOFF,
              "chamfer_min": CHAMFER_MIN, "target": target}

    # --- reanudacion ---
    resumed = load_checkpoint(out_dir)
    if resumed:
        start_stage, beams, graduated_records, traj_rows, dead_count = resumed
        print(f"RESUMED from stage {start_stage} ({len(beams)} beams, "
              f"{len(graduated_records)} graduated)", flush=True)
    else:
        start_stage = 1
        beams = [{
            "fix_atoms_arg": WARHEAD_NAMES[args.pair], "scaffold_is_sdf": False,
            "tani_B": None, "prot_B": None, "anchor_heavy": 7, "lineage": "root",
            "cumulative_gain": 0.0,
        }]
        graduated_records = []
        traj_rows = []
        dead_count = 0

    final_status = "stopped_max_stages"

    for stage in range(start_stage, MAX_STAGES + 1):
        if not beams:
            final_status = ("completed_graduated" if graduated_records
                            else "completed_all_dead")
            break

        gen = generate_all_beams(beams, stage, out_dir, args.pdbfile,
                                 args.b_ligand, args.lambda_global, args.seed)

        candidates = []
        for bi, beam in enumerate(beams):
            parent = beam
            parent_heavy_xyz = (np.zeros((0, 3)) if not parent["scaffold_is_sdf"]
                                else _load_anchor_heavy(parent["fix_atoms_arg"]))
            for add_n, sdf in gen.get(bi, []):
                records = list(Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False))
                for r in records:
                    a = audit_record(aud, r, reference_a, reference_b, fixed_indices)
                    if not a or not a.get("anchor_valid"):
                        continue
                    # guardia NaN (critica 9)
                    if not (np.isfinite(a["tani_B"]) and np.isfinite(a["prot_B"])):
                        continue
                    grown = anchor_grown_coords(a["anchor_mol"], a["_fixed_global"])
                    # rechazar sin crecimiento conectado (critica 7)
                    if len(grown) == 0:
                        continue
                    # crecimiento real vs padre
                    if a["anchor_heavy"] <= parent["anchor_heavy"]:
                        continue
                    # retencion del scaffold parental (critica 7)
                    child_heavy_xyz = anchor_all_heavy_coords(a["anchor_mol"])
                    if not scaffold_retained(parent_heavy_xyz, child_heavy_xyz,
                                             RETENTION_TOL):
                        continue
                    # gate trade-off (etapa 1: sin padre con forma -> solo validez)
                    if parent["tani_B"] is None:
                        F = 0.0; g_T = 0.0; g_P = 0.0; passed = True
                    else:
                        passed, F, g_T, g_P = compute_gate(
                            a["tani_B"], a["prot_B"],
                            parent["tani_B"], parent["prot_B"], EPS_TRADEOFF)
                    if not passed:
                        continue
                    a["local_gain"] = F
                    a["absolute_quality"] = absolute_quality(a["tani_B"], a["prot_B"])
                    a["cumulative_gain"] = parent.get("cumulative_gain", 0.0) + F
                    a["add_n"] = add_n
                    a["parent_lineage"] = beam["lineage"]
                    a["grown_xyz"] = grown
                    candidates.append(a)

        if not candidates:
            dead_count += len(beams)
            traj_rows.append({"stage": stage, "event": "death_no_progress",
                              "n_beams_in": len(beams), "n_candidates": 0})
            beams = []
            save_checkpoint(out_dir, stage, beams, graduated_records, traj_rows,
                            dead_count, config)
            final_status = "completed_all_dead" if not graduated_records else "completed_graduated"
            break

        chosen = select_beams(candidates, k_beams, CHAMFER_MIN)

        next_beams = []
        for ci, c in enumerate(chosen):
            anchor_sdf = str(out_dir / f"stage_{stage:02d}_beam_{ci}_anchor.sdf")
            w = Chem.SDWriter(anchor_sdf); w.write(c["anchor_mol"]); w.close()

            grad = graduation_status(c["anchor_heavy"], target, TARGET_TOL)
            row = {
                "stage": stage, "beam": ci, "add_n": c["add_n"],
                "parent_lineage": c["parent_lineage"],
                "anchor_heavy": c["anchor_heavy"], "connected": c["connected"],
                "n_heavy_fragments": c["n_heavy_fragments"],
                "tani_B": round(c["tani_B"], 4), "prot_B": round(c["prot_B"], 4),
                "local_gain": round(c["local_gain"], 4),
                "absolute_quality": round(c["absolute_quality"], 4),
                "cumulative_gain": round(c["cumulative_gain"], 4),
                "event": grad,
            }

            if grad == "oversize":
                row["event"] = "rejected_oversize"
                traj_rows.append(row)
                continue  # no propaga ni gradua
            if grad == "graduated":
                # dos endpoints: component vs strictly connected
                final_sdf = str(out_dir / "graduated" / f"final_{len(graduated_records)}.sdf")
                w = Chem.SDWriter(final_sdf); w.write(c["anchor_mol"]); w.close()
                graduated_records.append({
                    "anchor_heavy": c["anchor_heavy"],
                    "tani_B": c["tani_B"], "prot_B": c["prot_B"],
                    "component_graduated": True,
                    "strictly_connected_graduated": bool(c["connected"]),
                    "lineage": c["parent_lineage"],
                })
                traj_rows.append(row)
            else:  # growing
                traj_rows.append(row)
                next_beams.append({
                    "fix_atoms_arg": anchor_sdf, "scaffold_is_sdf": True,
                    "tani_B": c["tani_B"], "prot_B": c["prot_B"],
                    "anchor_heavy": c["anchor_heavy"],
                    "lineage": f"{c['parent_lineage']}>s{stage}b{ci}",
                    "cumulative_gain": c["cumulative_gain"],
                })

        beams = next_beams
        save_checkpoint(out_dir, stage, beams, graduated_records, traj_rows,
                        dead_count, config)
        print(f"[stage {stage}] candidates={len(candidates)} chosen={len(chosen)} "
              f"graduated={len(graduated_records)} continuing={len(beams)}", flush=True)

        if not beams:
            final_status = "completed_graduated" if graduated_records else "completed_all_dead"
            break

    # escribir trajectory + run_status
    fields = ["stage", "beam", "add_n", "parent_lineage", "anchor_heavy",
              "connected", "n_heavy_fragments", "tani_B", "prot_B",
              "local_gain", "absolute_quality", "cumulative_gain", "event",
              "n_beams_in", "n_candidates"]
    with (out_dir / "beam_trajectory.tsv").open("w", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        wr.writeheader(); wr.writerows(traj_rows)

    stages_run = max((r.get("stage", 0) for r in traj_rows), default=0)
    write_run_status(out_dir, final_status, graduated_records, dead_count, stages_run)

    print(f"BEAM_DONE status={final_status} graduated={len(graduated_records)} "
          f"dead={dead_count} out_dir={out_dir}", flush=True)


def _load_anchor_heavy(sdf_path):
    """Carga un scaffold SDF y devuelve coords de sus pesados (para retencion)."""
    try:
        m = next(iter(Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)))
        if m is None:
            return np.zeros((0, 3))
        return anchor_all_heavy_coords(m)
    except Exception:
        return np.zeros((0, 3))


if __name__ == "__main__":
    main()
