#!/usr/bin/env python3
"""Manifest de CAMPANA 1 v2 (phase3_fixed_highn): TRES ramas.

Incorpora el control A10 (best-of-10 B-blind) que aisla el confusor best-of-10
en la comparacion A vs B. Tres ramas:

  A1  (branch=A, n_samples=1):  deriva neutral pura, 3 replicas/semilla.
  A10 (branch=A, n_samples=10): best-of-10 seleccion B-blind, 3 replicas/semilla.
  B   (branch=B, n_samples=10): dirigida a B, 1/semilla.

Comparaciones que habilita:
  A1  vs A10 : efecto del best-of-10 estructural (sin mirar B).
  A10 vs B   : efecto de la DIRECCION hacia B (mismo muestreo, misma best-of-10;
               solo cambia si la seleccion mira a B). Aisla limpiamente el objetivo.

Ejes: 2 pares x 4 brazos (+3/+4/+5/+6) x 10 semillas.
n_stages parejo: +3->3, +4->3, +5->2, +6->2. Parada por n_stages (sin cambios).

Conteo:
  A1  = 2 x 4 x 10 x 3 = 240
  A10 = 2 x 4 x 10 x 3 = 240
  B   = 2 x 4 x 10 x 1 =  80
  total = 560 trayectorias.

Uso (contenedor, desde la raiz):
    python src/growth/make_campaign1_manifest.py \
        --out artifacts/phase3_fixed_highn/stage3_manifest.tsv
"""

import argparse
import csv
import glob
from pathlib import Path

ARMS = {3: 3, 4: 3, 5: 2, 6: 2}
SEEDS = [1101, 2202, 3303, 4404, 5505, 6606, 7707, 8808, 9909, 1010]
N_REPLICAS_A = 3   # aplica a A1 y A10 por igual
LAMBDA_GLOBAL = 20.0

PAIRS = [
    {"pair_id": "x0434_x2193", "A_local": "x0434", "B_global": "x2193"},
    {"pair_id": "x0874_x1093", "A_local": "x0874", "B_global": "x1093"},
]

OUT_ROOT = "artifacts/phase3_fixed_highn"

# definicion de ramas: (arm_label, branch, n_samples, n_replicas)
BRANCH_DEFS = [
    ("A1",  "A",  1, N_REPLICAS_A),
    ("A10", "A", 10, N_REPLICAS_A),
    ("B",   "B", 10, 1),
]


def resolve_one(pattern, desc):
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise SystemExit("ERROR: no se encontro {} con patron: {}".format(desc, pattern))
    if len(matches) > 1:
        raise SystemExit("ERROR: {} ambiguo:\n  {}".format(desc, "\n  ".join(matches)))
    return matches[0]


def resolve_pair_paths(pair):
    a_complex = resolve_one(
        "data/mpro/prepared/silvr_xchem_hits/{}__*/*_complex.pdb".format(pair["A_local"]),
        "complex de {}".format(pair["A_local"]))
    b_ligand = resolve_one(
        "data/mpro/prepared/silvr_xchem_hits/{}__*/*_ligand.sdf".format(pair["B_global"]),
        "ligando de {}".format(pair["B_global"]))
    return a_complex, b_ligand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/phase3_fixed_highn/stage3_manifest.tsv")
    args = ap.parse_args()

    rows = []
    task_id = 0

    for pair in PAIRS:
        a_complex, b_ligand = resolve_pair_paths(pair)
        for add_n, n_stages in ARMS.items():
            for seed in SEEDS:
                for arm_label, branch, n_samples, n_rep in BRANCH_DEFS:
                    for replica in range(1, n_rep + 1):
                        # eff_seed: para ramas con replicas (A1, A10) variar por replica.
                        # Para B (1 replica) usar seed base tal cual (como phase2).
                        if n_rep > 1:
                            eff_seed = seed + replica * 1000
                        else:
                            eff_seed = seed

                        if n_rep > 1:
                            out_dir = "{}/{}/arm_add{}/branch_{}/seed_{}/rep_{}".format(
                                OUT_ROOT, pair["pair_id"], add_n, arm_label, seed, replica)
                            exp_id = "stage3_{}_add{}_{}_seed{}_rep{}".format(
                                pair["pair_id"], add_n, arm_label, seed, replica)
                        else:
                            out_dir = "{}/{}/arm_add{}/branch_{}/seed_{}".format(
                                OUT_ROOT, pair["pair_id"], add_n, arm_label, seed)
                            exp_id = "stage3_{}_add{}_{}_seed{}".format(
                                pair["pair_id"], add_n, arm_label, seed)

                        rows.append({
                            "task_id": task_id, "pair_id": pair["pair_id"],
                            "A_local": pair["A_local"], "B_global": pair["B_global"],
                            "arm_label": arm_label, "branch": branch,
                            "add_n": add_n, "n_stages": n_stages,
                            "n_samples": n_samples, "seed": eff_seed, "replica": replica,
                            "lambda_global": LAMBDA_GLOBAL,
                            "experiment_id": exp_id, "out_dir": out_dir,
                            "A_complex": a_complex, "B_ligand": b_ligand,
                        })
                        task_id += 1

    fields = ["task_id", "pair_id", "A_local", "B_global", "arm_label", "branch",
              "add_n", "n_stages", "n_samples", "seed", "replica", "lambda_global",
              "experiment_id", "out_dir", "A_complex", "B_ligand"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    by_label = Counter(r["arm_label"] for r in rows)
    print("[ok] {} trayectorias escritas en {}".format(len(rows), out))
    for lbl in ["A1", "A10", "B"]:
        print("     rama {:>3}: {}".format(lbl, by_label[lbl]))
    print("     array SLURM: --array=0-{}".format(len(rows) - 1))


if __name__ == "__main__":
    main()
