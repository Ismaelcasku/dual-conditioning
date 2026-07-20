#!/usr/bin/env python3
"""Analisis del experimento de crecimiento estadificado (etapa 2).

Apila los trajectory.tsv de las 144 trayectorias y responde:
  - A vs B: llega la seleccion dirigida a B (rama B) mas cerca de B que la
    estructural neutral (rama A)?
  - Granularidad: crecen los brazos finos (+2/+4) conectado y hacia B, mientras
    los gruesos (+8) fragmentan?
  - Trayectorias de forma: tani_B / protrude_B etapa a etapa, por brazo y rama.
  - Mortalidad: cuantas trayectorias mueren, cuando y por que.
  - Supervivencia del warhead: tasa de anchor valido por brazo.

Descubre los trajectory.tsv recorriendo el arbol de directorios y parseando la
estructura de rutas (.../<pair>/arm_add<N>/branch_<X>/seed_<s>[/rep_<r>]/).

Uso:
    python analyze_fixed_increment_campaign.py \
        --root artifacts/phase2_staged_growth \
        --out_dir artifacts/phase2_staged_growth/analysis
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_path(traj_path, root):
    """Extrae (pair, add_n, branch, seed, rep) de la ruta de un trajectory.tsv."""
    rel = traj_path.relative_to(root)
    parts = rel.parts
    pair = parts[0]
    add_n = int(re.match(r"arm_add(\d+)", parts[1]).group(1))
    branch = parts[2].replace("branch_", "")
    seed = parts[3].replace("seed_", "")
    rep = parts[4].replace("rep_", "") if len(parts) > 5 and parts[4].startswith("rep_") else "0"
    return pair, add_n, branch, seed, rep


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def read_trajectory(path):
    with open(path) as h:
        return list(csv.DictReader(h, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/phase2_staged_growth")
    ap.add_argument("--out_dir", default="artifacts/phase2_staged_growth/analysis")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(root.rglob("trajectory.tsv"))
    print("trayectorias encontradas: {}".format(len(traj_files)))

    # ------------------------------------------------------------------
    # 1. tabla larga: una fila por (trayectoria, etapa)
    # ------------------------------------------------------------------
    long_rows = []
    # resumen por trayectoria
    traj_summary = []

    for tf in traj_files:
        pair, add_n, branch, seed, rep = parse_path(tf, root)
        stages = read_trajectory(tf)

        # etapas efectivas (con scaffold escrito = crecio/continuo)
        completed = [s for s in stages if s.get("scaffold_sdf")]
        last = completed[-1] if completed else (stages[-1] if stages else None)

        # motivo de fin
        final_gate = stages[-1]["gate"] if stages else "empty"

        # progreso de forma: primera vs ultima etapa completada
        first_tani = to_float(completed[0]["tani_B"]) if completed else None
        last_tani = to_float(last["tani_B"]) if last and last.get("tani_B") else None
        first_anchor = to_int(completed[0]["anchor_heavy"]) if completed else None
        last_anchor = to_int(last["anchor_heavy"]) if last and last.get("anchor_heavy") else None

        traj_summary.append({
            "pair": pair, "add_n": add_n, "branch": branch, "seed": seed, "rep": rep,
            "n_stages_total": len(stages),
            "n_stages_completed": len(completed),
            "final_gate": final_gate,
            "first_anchor_heavy": first_anchor,
            "last_anchor_heavy": last_anchor,
            "anchor_growth": (last_anchor - first_anchor) if (first_anchor is not None and last_anchor is not None) else None,
            "first_tani_B": first_tani,
            "last_tani_B": last_tani,
            "tani_B_improvement": (first_tani - last_tani) if (first_tani is not None and last_tani is not None) else None,
            "last_connected": last.get("connected") if last else None,
        })

        for s in stages:
            long_rows.append({
                "pair": pair, "add_n": add_n, "branch": branch, "seed": seed, "rep": rep,
                "stage": s.get("stage"),
                "gate": s.get("gate"),
                "anchor_heavy": s.get("anchor_heavy"),
                "connected": s.get("connected"),
                "n_heavy_fragments": s.get("n_heavy_fragments"),
                "anchor_max_drift": s.get("anchor_max_drift"),
                "tani_B": s.get("tani_B"),
                "prot_B": s.get("prot_B"),
            })

    # escribir tablas
    long_fields = ["pair","add_n","branch","seed","rep","stage","gate",
                   "anchor_heavy","connected","n_heavy_fragments",
                   "anchor_max_drift","tani_B","prot_B"]
    with (out_dir / "all_stages_long.tsv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=long_fields, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(long_rows)

    sum_fields = list(traj_summary[0].keys()) if traj_summary else []
    with (out_dir / "trajectory_summary.tsv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=sum_fields, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(traj_summary)

    # ------------------------------------------------------------------
    # 2. resumen impreso por (rama, brazo)
    # ------------------------------------------------------------------
    def fmt(x, d=3):
        return "{:.{}f}".format(x, d) if isinstance(x, (int, float)) else "NA"

    print("\n" + "=" * 90)
    print("RESUMEN POR RAMA Y BRAZO (media sobre pares/seeds/replicas)")
    print("=" * 90)
    print("{:<6} {:<6} {:>6} {:>10} {:>12} {:>12} {:>12} {:>10}".format(
        "rama","add_n","n_tr","et_compl","anchor_fin","crece_anchor","mejora_B","muertes"))

    groups = defaultdict(list)
    for t in traj_summary:
        groups[(t["branch"], t["add_n"])].append(t)

    for (branch, add_n) in sorted(groups, key=lambda k: (k[0], -k[1])):
        g = groups[(branch, add_n)]
        n = len(g)
        et = [t["n_stages_completed"] for t in g]
        af = [t["last_anchor_heavy"] for t in g if t["last_anchor_heavy"] is not None]
        gr = [t["anchor_growth"] for t in g if t["anchor_growth"] is not None]
        mb = [t["tani_B_improvement"] for t in g if t["tani_B_improvement"] is not None]
        deaths = sum(1 for t in g if t["final_gate"].startswith("death"))
        print("{:<6} {:<6} {:>6} {:>10} {:>12} {:>12} {:>12} {:>10}".format(
            branch, add_n, n,
            fmt(sum(et)/len(et),1) if et else "NA",
            fmt(sum(af)/len(af),1) if af else "NA",
            fmt(sum(gr)/len(gr),1) if gr else "NA",
            fmt(sum(mb)/len(mb),3) if mb else "NA",
            deaths))

    print("\nColumnas:")
    print("  et_compl     = etapas completadas (media)")
    print("  anchor_fin   = atomos del anchor al final (media)")
    print("  crece_anchor = atomos ganados desde etapa 1 (media; >0 = crecio)")
    print("  mejora_B     = reduccion de tani_B (primera->ultima; >0 = se acerco a B)")
    print("  muertes      = trayectorias terminadas por death_*")

    # ------------------------------------------------------------------
    # 3. desglose de motivos de muerte
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MOTIVOS DE FIN DE TRAYECTORIA")
    print("=" * 60)
    gate_counts = defaultdict(int)
    for t in traj_summary:
        gate_counts[(t["branch"], t["final_gate"])] += 1
    for (branch, gate) in sorted(gate_counts):
        print("  rama {}  {:<28} {}".format(branch, gate, gate_counts[(branch, gate)]))

    print("\nTablas escritas en {}/".format(out_dir))
    print("  all_stages_long.tsv      (una fila por trayectoria-etapa)")
    print("  trajectory_summary.tsv   (una fila por trayectoria)")


if __name__ == "__main__":
    main()
