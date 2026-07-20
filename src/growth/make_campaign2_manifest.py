#!/usr/bin/env python3
"""Manifest de CAMPANA 2 v2 (phase4_evo_beam): greedy vs beam, incremento variable.

VARIANTES (corrige critica 1):
  greedy : k_beams=1. Greedy real de incremento variable.
  beam   : k_beams=3. Beam search real (arbol).
  Ambas comparten el MISMO gate trade-off (eps=0.01). Solo cambia k, aislando el
  efecto del beam width.

Ejes: 2 pares x 10 semillas x 2 variantes = 40 trayectorias.
--pilot N limita a las primeras N semillas.

Uso:
    python src/growth/make_campaign2_manifest.py \
        --out artifacts/phase4_evo_beam/beam_manifest.tsv
    python src/growth/make_campaign2_manifest.py \
        --out artifacts/phase4_evo_beam/beam_manifest_pilot.tsv --pilot 3
"""

import argparse, csv, glob
from pathlib import Path

SEEDS = [1101, 2202, 3303, 4404, 5505, 6606, 7707, 8808, 9909, 1010]
LAMBDA_GLOBAL = 20.0
VARIANTS = [("greedy", 1), ("beam", 3)]  # (label, k_beams)

PAIRS = [
    {"pair_id": "x0434_x2193", "A_local": "x0434", "B_global": "x2193"},
    {"pair_id": "x0874_x1093", "A_local": "x0874", "B_global": "x1093"},
]
OUT_ROOT = "artifacts/phase4_evo_beam"


def resolve_one(pattern, desc):
    m = sorted(glob.glob(pattern))
    if len(m) == 0:
        raise SystemExit("ERROR: no se encontro {} con patron: {}".format(desc, pattern))
    if len(m) > 1:
        raise SystemExit("ERROR: {} ambiguo:\n  {}".format(desc, "\n  ".join(m)))
    return m[0]


def resolve_pair_paths(pair):
    a = resolve_one("data/mpro/prepared/silvr_xchem_hits/{}__*/*_complex.pdb".format(pair["A_local"]),
                    "complex de {}".format(pair["A_local"]))
    b = resolve_one("data/mpro/prepared/silvr_xchem_hits/{}__*/*_ligand.sdf".format(pair["B_global"]),
                    "ligando de {}".format(pair["B_global"]))
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/phase4_evo_beam/beam_manifest.tsv")
    ap.add_argument("--pilot", type=int, default=None)
    args = ap.parse_args()

    seeds = SEEDS[:args.pilot] if args.pilot else SEEDS
    rows = []
    task_id = 0
    for pair in PAIRS:
        a_complex, b_ligand = resolve_pair_paths(pair)
        for seed in seeds:
            for variant_label, k_beams in VARIANTS:
                out_dir = "{}/{}/variant_{}/seed_{}".format(
                    OUT_ROOT, pair["pair_id"], variant_label, seed)
                exp_id = "beam_{}_{}_seed{}".format(pair["pair_id"], variant_label, seed)
                rows.append({
                    "task_id": task_id, "pair_id": pair["pair_id"],
                    "A_local": pair["A_local"], "B_global": pair["B_global"],
                    "variant": variant_label, "k_beams": k_beams,
                    "seed": seed, "lambda_global": LAMBDA_GLOBAL,
                    "experiment_id": exp_id, "out_dir": out_dir,
                    "A_complex": a_complex, "B_ligand": b_ligand,
                })
                task_id += 1

    fields = ["task_id", "pair_id", "A_local", "B_global", "variant", "k_beams",
              "seed", "lambda_global", "experiment_id", "out_dir",
              "A_complex", "B_ligand"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    bv = Counter(r["variant"] for r in rows)
    print("[ok] {} trayectorias en {}".format(len(rows), out))
    print("     greedy(k=1)={} beam(k=3)={}".format(bv["greedy"], bv["beam"]))
    print("     semillas: {} {}".format(len(seeds), "(PILOTO)" if args.pilot else "(completo)"))
    print("     array: --array=0-{}".format(len(rows) - 1))


if __name__ == "__main__":
    main()
