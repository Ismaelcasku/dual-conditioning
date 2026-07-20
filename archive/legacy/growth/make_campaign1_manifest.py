#!/usr/bin/env python3
"""Genera el manifest de trayectorias de la CAMPANA 1 (phase3_fixed_highn).

Variante de make_stage2_manifest.py: MISMA firma de columnas, mismo esquema de
eff_seed y resolucion de rutas por glob, para que stage2_staged_growth.slurm y
orchestrator.py funcionen SIN cambios. Solo cambian los ejes del barrido.

Campana 1 = repeticion a mayor n del experimento de incremento FIJO (phase2),
como baseline pareado para la Campana 2 (beam variable). Diferencias vs phase2:
  - Brazos +3,+4,+5,+6 (en vez de +2,+4,+6,+8).
  - n_stages parejo: +3->3, +4->3, +5->2, +6->2 (crecimiento 9-12 atomos).
  - 10 semillas (3 heredadas de phase2 + 7 nuevas) para potencia estadistica.
  - 3 replicas rama A (en vez de 5): 3 rep x 10 seeds = 30 trayectorias A/celda.
  - Criterio de parada SIN cambios: por n_stages (NO por target de atomos).
    El target de atomos es solo para la Campana 2.

Conteo:
  A: 2 pares x 4 brazos x 10 seeds x 3 replicas = 240
  B: 2 pares x 4 brazos x 10 seeds x 1          =  80
  total = 320 trayectorias.

IMPORTANTE: las semillas heredadas {1101,2202,3303} se conservan para que el
baseline sea pareado por semilla con phase2. NO reordenar ni renombrar.

Uso (dentro del contenedor, desde la raiz del proyecto):
    python codes/growth/make_campaign1_manifest.py \
        --out artifacts/phase3_fixed_highn/stage3_manifest.tsv
"""

import argparse
import csv
import glob
from pathlib import Path

# brazo -> n_stages  (mapeo parejo: crecimiento 9-12 atomos sobre warhead=7)
ARMS = {3: 3, 4: 3, 5: 2, 6: 2}
# 3 heredadas de phase2 + 7 nuevas. Verificado: eff_seed=seed+rep*1000 sin colision.
SEEDS = [1101, 2202, 3303, 4404, 5505, 6606, 7707, 8808, 9909, 1010]
N_REPLICAS_A = 3
LAMBDA_GLOBAL = 20.0

PAIRS = [
    {"pair_id": "x0434_x2193", "A_local": "x0434", "B_global": "x2193"},
    {"pair_id": "x0874_x1093", "A_local": "x0874", "B_global": "x1093"},
]

OUT_ROOT = "artifacts/phase3_fixed_highn"


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
                for replica in range(1, N_REPLICAS_A + 1):
                    eff_seed = seed + replica * 1000
                    exp_id = "stage3_{}_add{}_A_seed{}_rep{}".format(
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

                exp_id = "stage3_{}_add{}_B_seed{}".format(
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
