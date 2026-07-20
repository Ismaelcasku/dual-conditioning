#!/usr/bin/env python3
"""Genera el manifest de trayectorias del experimento de crecimiento estadificado.

Una fila = una trayectoria = una tarea de SLURM.

Diseno:
  - Rama A (structural): 1 muestra/etapa, 5 replicas por (par,brazo,semilla).
                         Cada replica es una trayectoria independiente.
  - Rama B (b_directed): 10 muestras/etapa, 1 trayectoria por (par,brazo,semilla).
  - Brazos: +8 (2 etapas), +6 (3), +4 (4), +2 (8).
  - 2 pares, 3 semillas base.

Conteo:
  A: 2 pares x 4 brazos x 3 seeds x 5 replicas = 120
  B: 2 pares x 4 brazos x 3 seeds x 1          =  24
  total = 144 trayectorias.

Las rutas de A_complex y B_ligand se descubren por glob (como en la etapa 1).

Uso (dentro del contenedor, desde la raiz del proyecto):
    python codes/growth/make_stage2_manifest.py \
        --out artifacts/phase2_staged_growth/stage2_manifest.tsv
"""

import argparse
import csv
import glob
from pathlib import Path

# brazo -> n_stages
ARMS = {8: 2, 6: 3, 4: 4, 2: 8}
SEEDS = [1101, 2202, 3303]
N_REPLICAS_A = 5
LAMBDA_GLOBAL = 20.0

PAIRS = [
    {"pair_id": "x0434_x2193", "A_local": "x0434", "B_global": "x2193"},
    {"pair_id": "x0874_x1093", "A_local": "x0874", "B_global": "x1093"},
]

OUT_ROOT = "artifacts/phase2_staged_growth"


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
    ap.add_argument("--out", default="artifacts/phase2_staged_growth/stage2_manifest.tsv")
    args = ap.parse_args()

    rows = []
    task_id = 0

    for pair in PAIRS:
        a_complex, b_ligand = resolve_pair_paths(pair)

        for add_n, n_stages in ARMS.items():
            for seed in SEEDS:
                # --- rama A: 5 replicas, 1 muestra/etapa ---
                for replica in range(1, N_REPLICAS_A + 1):
                    # semilla efectiva distinta por replica para que no sean identicas
                    eff_seed = seed + replica * 1000
                    exp_id = "stage2_{}_add{}_A_seed{}_rep{}".format(
                        pair["pair_id"], add_n, seed, replica)
                    out_dir = "{}/{}/arm_add{}/branch_A/seed_{}/rep_{}".format(
                        OUT_ROOT, pair["pair_id"], add_n, seed, replica)
                    rows.append({
                        "task_id": task_id, "pair_id": pair["pair_id"],
                        "A_local": pair["A_local"], "B_global": pair["B_global"],
                        "branch": "A", "add_n": add_n, "n_stages": n_stages,
                        "n_samples": 1, "seed": eff_seed, "replica": replica,
                        "lambda_global": LAMBDA_GLOBAL,
                        "experiment_id": exp_id, "out_dir": out_dir,
                        "A_complex": a_complex, "B_ligand": b_ligand,
                    })
                    task_id += 1

                # --- rama B: 1 trayectoria, 10 muestras/etapa ---
                exp_id = "stage2_{}_add{}_B_seed{}".format(
                    pair["pair_id"], add_n, seed)
                out_dir = "{}/{}/arm_add{}/branch_B/seed_{}".format(
                    OUT_ROOT, pair["pair_id"], add_n, seed)
                rows.append({
                    "task_id": task_id, "pair_id": pair["pair_id"],
                    "A_local": pair["A_local"], "B_global": pair["B_global"],
                    "branch": "B", "add_n": add_n, "n_stages": n_stages,
                    "n_samples": 10, "seed": seed, "replica": 0,
                    "lambda_global": LAMBDA_GLOBAL,
                    "experiment_id": exp_id, "out_dir": out_dir,
                    "A_complex": a_complex, "B_ligand": b_ligand,
                })
                task_id += 1

    fields = ["task_id", "pair_id", "A_local", "B_global", "branch", "add_n",
              "n_stages", "n_samples", "seed", "replica", "lambda_global",
              "experiment_id", "out_dir", "A_complex", "B_ligand"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    n_a = sum(1 for r in rows if r["branch"] == "A")
    n_b = sum(1 for r in rows if r["branch"] == "B")
    print("[ok] {} trayectorias escritas en {}".format(len(rows), out))
    print("     rama A: {}  rama B: {}".format(n_a, n_b))
    print("     array SLURM: --array=0-{}".format(len(rows) - 1))
    print("     ejes: pares={} brazos={} seeds={} replicas_A={}".format(
        [p["pair_id"] for p in PAIRS], list(ARMS.keys()), SEEDS, N_REPLICAS_A))


if __name__ == "__main__":
    main()
