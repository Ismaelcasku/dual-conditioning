#!/usr/bin/env python3
"""Genera el manifest TSV de la ETAPA 1 del piloto de crecimiento por etapas.

Extiende el esquema del manifest de la campana Fase 0 (exp06) con dos ejes
nuevos: `resamplings` y `add_n_nodes`. Mantiene identicas todas las demas
columnas y el mecanismo de generacion.

Ejes barridos
-------------
    pares        : x0874_x1093 (moderado), x0434_x2193 (dificil)
    lambda       : 10, 20
    resamplings  : 5, 10, 20
    add_n_nodes  : 2, 4, 6, 8, None   (None = el modelo decide el tamano)
    seeds        : 1101, 2202

Total: 2 pares x 2 lambda x 3 resamplings x 5 add_n_nodes x 2 seeds = 120 tareas.

Uso (desde la raiz del proyecto)
--------------------------------
    python src/growth/make_stage1_manifest.py \
        --out artifacts/phase1_fragment_growing/stage1_manifest.tsv

Las rutas de complex/ligando se descubren por glob contra data/mpro/prepared,
para no depender de transcribir a mano los nombres largos de fichero.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Definicion de pares. Las rutas se resuelven por glob a partir del codigo de
# fragmento (x0874, x1093, ...). El fixed_json de cada A se pasa explicito
# porque su nombre no sigue el patron de los ficheros prepared.
# ---------------------------------------------------------------------------

PAIRS = [
    {
        "pair_id": "x0874_x1093",
        "A_local": "x0874",
        "B_global": "x1093",
        "fixed_atoms": "C02 C04 C05 C06 C07 C08 C09",
        # RELLENAR/confirmar: nombre real del fixed_json de x0874
        "fixed_json": "data/mpro/manifests/b0_x0874_T54_fixed_atoms.json",
    },
    {
        "pair_id": "x0434_x2193",
        "A_local": "x0434",
        "B_global": "x2193",
        "fixed_atoms": "C4 C5 C6 N C1 C2 C3",
        # RELLENAR: nombre real del fixed_json de x0434 (el grep salio vacio).
        # Sustituir por la ruta que devuelva el diagnostico en el cluster.
        "fixed_json": "artifacts/fixed_atom_screen/x0434_blind/x0434_fixed_anchor_EVALUATOR_V2.json",
    },
]

LAMBDAS = [10.0, 20.0]
RESAMPLINGS = [5, 10, 20]
ADD_N_NODES = [2, 4, 6, 8, None]
SEEDS = [1101, 2202]

N_SAMPLES = 10
EXPERIMENT_PREFIX = "stage1"
OUT_ROOT = "artifacts/phase1_fragment_growing"


def resolve_one(pattern: str, description: str) -> str:
    """Devuelve la unica ruta que casa el patron, o aborta con mensaje claro."""
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise SystemExit(f"ERROR: no se encontro {description} con patron: {pattern}")
    if len(matches) > 1:
        raise SystemExit(
            f"ERROR: {description} ambiguo, {len(matches)} coincidencias:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def resolve_pair_paths(pair: dict) -> dict:
    """Descubre A_complex (del A) y B_ligand (del B) por glob."""
    a_complex = resolve_one(
        f"data/mpro/prepared/silvr_xchem_hits/{pair['A_local']}__*/*_complex.pdb",
        f"complex de {pair['A_local']}",
    )
    b_ligand = resolve_one(
        f"data/mpro/prepared/silvr_xchem_hits/{pair['B_global']}__*/*_ligand.sdf",
        f"ligando de {pair['B_global']}",
    )
    if not Path(pair["fixed_json"]).is_file():
        raise SystemExit(
            f"ERROR: fixed_json no existe para {pair['pair_id']}: {pair['fixed_json']}\n"
            "       Corrige la ruta en PAIRS antes de generar el manifest."
        )
    return {"A_complex": a_complex, "B_ligand": b_ligand}


def add_n_nodes_tag(value) -> str:
    return "auto" if value is None else f"n{value}"


def build_rows() -> list[dict]:
    rows = []
    task_id = 0
    for pair in PAIRS:
        paths = resolve_pair_paths(pair)
        for seed in SEEDS:
            for lam in LAMBDAS:
                for resamp in RESAMPLINGS:
                    for add_n in ADD_N_NODES:
                        ann_tag = add_n_nodes_tag(add_n)
                        exp_id = (
                            f"{EXPERIMENT_PREFIX}_{pair['pair_id']}_seed_{seed}"
                            f"_lambda_{lam}_resamp_{resamp}_{ann_tag}"
                        )
                        out_sdf = (
                            f"{OUT_ROOT}/{pair['pair_id']}/seed_{seed}/"
                            f"lambda_{lam}/resamp_{resamp}/{ann_tag}/{exp_id}_n{N_SAMPLES}.sdf"
                        )
                        rows.append(
                            {
                                "task_id": task_id,
                                "pair_id": pair["pair_id"],
                                "A_local": pair["A_local"],
                                "B_global": pair["B_global"],
                                "seed": seed,
                                "lambda_global": lam,
                                "resamplings": resamp,
                                # cadena vacia = pasar add_n_nodes=None (el modelo decide)
                                "add_n_nodes": "" if add_n is None else add_n,
                                "n_samples_requested": N_SAMPLES,
                                "experiment_id": exp_id,
                                "out_sdf": out_sdf,
                                "A_complex": paths["A_complex"],
                                "B_ligand": paths["B_ligand"],
                                "fixed_json": pair["fixed_json"],
                                "fixed_atoms": pair["fixed_atoms"],
                            }
                        )
                        task_id += 1
    return rows


COLUMNS = [
    "task_id",
    "pair_id",
    "A_local",
    "B_global",
    "seed",
    "lambda_global",
    "resamplings",
    "add_n_nodes",
    "n_samples_requested",
    "experiment_id",
    "out_sdf",
    "A_complex",
    "B_ligand",
    "fixed_json",
    "fixed_atoms",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/phase1_fragment_growing/stage1_manifest.tsv"),
    )
    args = parser.parse_args()

    rows = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {len(rows)} tareas escritas en {args.out}")
    print(f"     array SLURM: --array=0-{len(rows) - 1}%<concurrencia>")
    # resumen de ejes para verificacion visual
    print("     ejes:")
    print(f"       pares       = {[p['pair_id'] for p in PAIRS]}")
    print(f"       lambda      = {LAMBDAS}")
    print(f"       resamplings = {RESAMPLINGS}")
    print(f"       add_n_nodes = {ADD_N_NODES}")
    print(f"       seeds       = {SEEDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
