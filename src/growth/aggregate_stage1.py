#!/usr/bin/env python3
"""Agregacion de la auditoria de la ETAPA 1 por condicion.

Consume el TSV por molecula (stage1_audit_per_molecule.tsv) y produce tablas
resumidas por condicion, para responder que combinacion de
(par, lambda, resamplings, add_n_nodes) produce con mas frecuencia un
anchor_component conectado, grande y dirigido a B.

Reutiliza describe() y write_tsv() del auditor de campana; no reimplementa
estadistica.

Salidas
-------
    stage1_summary_by_condition.tsv   una fila por (par,lambda,resamp,add_n,seed)
    stage1_summary_by_grid.tsv        colapsando seeds: (par,lambda,resamp,add_n)
    + impresion ordenada de las mejores condiciones por criterio de seleccion

Uso (dentro del contenedor, desde la raiz del proyecto)
-------------------------------------------------------
    python src/growth/aggregate_stage1.py \
        --audit_tsv artifacts/phase1_fragment_growing/stage1_audit/stage1_audit_per_molecule.tsv \
        --out_dir   artifacts/phase1_fragment_growing/stage1_audit
"""

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path


AUDITOR_PATH = "src/analysis/single_shot/audit_single_shot.py"


def load_auditor(path):
    spec = importlib.util.spec_from_file_location("campaign_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_true(value):
    return str(value).strip().upper() == "TRUE"


def summarize_group(aud, rows):
    """Resumen de un grupo de moleculas (misma condicion o mismo punto de grid)."""
    auditable = [r for r in rows if is_true(r.get("audit_ok"))]
    n_aud = len(auditable)

    connected = [r for r in auditable if is_true(r.get("heavy_connected"))]
    anchor_same = [
        r for r in auditable if is_true(r.get("anchor_all_same_fragment"))
    ]
    anchor_is_parent = [
        r for r in auditable if is_true(r.get("anchor_component_is_parent"))
    ]

    def rate(subset):
        return len(subset) / n_aud if n_aud else 0.0

    result = {
        "n_records": len(rows),
        "n_auditable": n_aud,
        "n_heavy_connected": len(connected),
        "heavy_connected_rate": rate(connected),
        "n_anchor_same_fragment": len(anchor_same),
        "anchor_same_fragment_rate": rate(anchor_same),
        "n_anchor_is_parent": len(anchor_is_parent),
        "anchor_component_is_parent_rate": rate(anchor_is_parent),
    }

    # metricas continuas: anchor y su similitud a B (el observable clave)
    for metric in [
        "anchor_heavy_atoms",
        "anchor_heavy_fraction",
        "parent_heavy_fraction",
        "n_heavy_fragments",
        "fixed_match_max_distance",
    ]:
        stats = aud.describe([to_float(r.get(metric)) for r in auditable])
        result["{}_mean".format(metric)] = stats["mean"]
        result["{}_sd".format(metric)] = stats["sd"]

    # similitud a B: sobre las moleculas cuyo anchor tiene shape valido
    anchor_shape_ok = [r for r in auditable if is_true(r.get("anchor_shape_ok"))]
    for metric in ["anchor_tanimoto_B", "anchor_protrude_B",
                   "anchor_tanimoto_A", "anchor_protrude_A"]:
        stats = aud.describe([to_float(r.get(metric)) for r in anchor_shape_ok])
        result["{}_mean".format(metric)] = stats["mean"]
        result["{}_sd".format(metric)] = stats["sd"]

    # strict dual sobre anchor (B mas cerca que A en ambas metricas)
    result["anchor_strict_dual_count"] = sum(
        is_true(r.get("anchor_strict_dual")) for r in auditable
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--auditor", default=AUDITOR_PATH)
    args = parser.parse_args()

    aud = load_auditor(args.auditor)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.audit_tsv) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    # ------------------------------------------------------------------
    # Nivel 1: por condicion completa (incluye seed)
    # ------------------------------------------------------------------
    by_condition = defaultdict(list)
    for r in rows:
        key = (
            r["pair_id"],
            r["lambda_global"],
            r["resamplings"],
            r["add_n_nodes"],
            r["seed"],
        )
        by_condition[key].append(r)

    condition_rows = []
    for key in sorted(by_condition):
        pair_id, lam, resamp, add_n, seed = key
        summary = {
            "pair_id": pair_id,
            "lambda_global": lam,
            "resamplings": resamp,
            "add_n_nodes": add_n,
            "seed": seed,
        }
        summary.update(summarize_group(aud, by_condition[key]))
        condition_rows.append(summary)

    # ------------------------------------------------------------------
    # Nivel 2: colapsando seeds -> punto de grid
    # ------------------------------------------------------------------
    by_grid = defaultdict(list)
    for r in rows:
        key = (r["pair_id"], r["lambda_global"], r["resamplings"], r["add_n_nodes"])
        by_grid[key].append(r)

    grid_rows = []
    for key in sorted(by_grid):
        pair_id, lam, resamp, add_n = key
        summary = {
            "pair_id": pair_id,
            "lambda_global": lam,
            "resamplings": resamp,
            "add_n_nodes": add_n,
        }
        summary.update(summarize_group(aud, by_grid[key]))
        grid_rows.append(summary)

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------
    condition_fields = list(condition_rows[0].keys())
    grid_fields = list(grid_rows[0].keys())

    aud.write_tsv(
        out_dir / "stage1_summary_by_condition.tsv", condition_rows, condition_fields
    )
    aud.write_tsv(
        out_dir / "stage1_summary_by_grid.tsv", grid_rows, grid_fields
    )

    # ------------------------------------------------------------------
    # Impresion: mejores puntos de grid por par, ordenados por el criterio
    # de seleccion (anchor conectado y grande) y con similitud a B al lado.
    # ------------------------------------------------------------------
    def fmt(value):
        f = to_float(value)
        return "{:.3f}".format(f) if f is not None else "NA"

    print()
    print("STAGE1_AGGREGATION_DONE")
    print("conditions={} grid_points={}".format(len(condition_rows), len(grid_rows)))
    print()

    for pair_id in sorted({r["pair_id"] for r in grid_rows}):
        print("=" * 100)
        print("PAR {}: puntos de grid ordenados por anchor_heavy_atoms_mean".format(pair_id))
        print(
            "lambda  resamp  add_n   conn_rate  anchsame  anch=parent  "
            "anch_heavy  anch_TaniB  anch_ProtB  strictB"
        )
        pair_rows = [r for r in grid_rows if r["pair_id"] == pair_id]
        # ordenar por atomos en el anchor (criterio de seleccion elegido)
        pair_rows.sort(
            key=lambda r: (to_float(r.get("anchor_heavy_atoms_mean")) or 0.0),
            reverse=True,
        )
        for r in pair_rows:
            print(
                "{:>6}  {:>6}  {:>5}   {:>9}  {:>8}  {:>11}  {:>10}  {:>10}  {:>10}  {:>7}".format(
                    r["lambda_global"],
                    r["resamplings"],
                    r["add_n_nodes"],
                    fmt(r["heavy_connected_rate"]),
                    fmt(r["anchor_same_fragment_rate"]),
                    fmt(r["anchor_component_is_parent_rate"]),
                    fmt(r["anchor_heavy_atoms_mean"]),
                    fmt(r["anchor_tanimoto_B_mean"]),
                    fmt(r["anchor_protrude_B_mean"]),
                    r["anchor_strict_dual_count"],
                )
            )
        print()

    print("out_dir={}".format(out_dir))
    print()
    print("Nota: anch_TaniB y anch_ProtB son DISTANCIAS (menor = mas cerca de B).")
    print("      conn_rate = fraccion totalmente conectada;")
    print("      anchsame  = fraccion con warhead intacto en un solo fragmento;")
    print("      anch=parent = fraccion donde el warhead esta en el fragmento mayor.")


if __name__ == "__main__":
    main()
