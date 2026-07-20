#!/usr/bin/env python3
"""Recomputa forma post-hoc del anchor contra A y contra B para todos los
scaffolds de Campana 1 (phase3_fixed_highn).

Motivacion: el pipeline de generacion solo midio forma contra B (guia + gate).
Para las figuras del plano A-B (tani_A vs tani_B) y el delta direccional
(tani_A - tani_B) hace falta la distancia de forma del anchor a A, que es una
metrica de analisis post-hoc (NO se uso para guiar). Se recomputa aqui sobre los
stage_XX_scaffold.sdf ya guardados. La generacion NO cambia.

Convencion (identica al auditor): ShapeTanimotoDist y ShapeProtrudeDist son
DISTANCIAS (menor = mas parecido). allowReordering=False para que Protrude sea
direccional (critica de Fase 0: RDKit puede reordenar por volumen).

Salida: all_stages_long_shapeAB.tsv con columnas:
  pair, add_n, branch, seed, rep, stage, anchor_heavy,
  tani_A, prot_A, tani_B, prot_B

Correr DENTRO del contenedor, desde la raiz del proyecto:
  singularity exec -B "$PWD:/mnt/proyecto" "$SANDBOX" bash -lc '
    cd /mnt/proyecto
    export PYTHONPATH=/mnt/proyecto/src:/mnt/proyecto/external/DiffSBDD
    python src/growth/recompute_shape_AB.py \
      --root artifacts/phase3_fixed_highn \
      --out  artifacts/phase3_fixed_highn/analysis/all_stages_long_shapeAB.tsv
  '
"""

import argparse
import csv
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdShapeHelpers

# rutas de referencia (del auditor)
A_SDF = {
    "x0434": "data/mpro/prepared/silvr_xchem_hits/x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_ligand.sdf",
    "x0874": "data/mpro/prepared/silvr_xchem_hits/x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_ligand.sdf",
}
B_SDF = {
    "x1093": "data/mpro/prepared/silvr_xchem_hits/x1093__5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20/5RF7_Mpro-x1093_AAR-POS-d2a4d1df-20_ligand.sdf",
    "x2193": "data/mpro/prepared/silvr_xchem_hits/x2193__5RHD_Mpro-x2193_AAR-POS-5507155c-1/5RHD_Mpro-x2193_AAR-POS-5507155c-1_ligand.sdf",
}
# par -> (A_local, B_global)
PAIR_PARTS = {
    "x0434_x2193": ("x0434", "x2193"),
    "x0874_x1093": ("x0874", "x1093"),
}


def load_mol(path):
    supp = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for m in supp:
        if m is not None:
            return m
    return None


def shape_dists(mol, ref):
    """Devuelve (tani, prot) distancias de forma; NaN si falla."""
    try:
        tani = float(rdShapeHelpers.ShapeTanimotoDist(mol, ref, ignoreHs=True))
        prot = float(rdShapeHelpers.ShapeProtrudeDist(mol, ref, ignoreHs=True,
                                                      allowReordering=False))
        return tani, prot
    except Exception:
        return float("nan"), float("nan")


def parse_scaffold_path(path, root):
    """Extrae (pair, add_n, branch, seed, rep, stage) de la ruta."""
    rel = Path(path).relative_to(root)
    parts = rel.parts
    # <pair>/arm_add<N>/branch_<X>/seed_<s>[/rep_<r>]/stage_XX_scaffold.sdf
    pair = parts[0]
    add_n = int(re.match(r"arm_add(\d+)", parts[1]).group(1))
    branch = parts[2].replace("branch_", "")
    seed = parts[3].replace("seed_", "")
    # rep puede o no estar (rama B no tiene rep_)
    if len(parts) >= 6 and parts[4].startswith("rep_"):
        rep = parts[4].replace("rep_", "")
        fname = parts[5]
    else:
        rep = "0"
        fname = parts[4]
    stage = int(re.match(r"stage_(\d+)_scaffold", fname).group(1))
    return pair, add_n, branch, seed, rep, stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/phase3_fixed_highn")
    ap.add_argument("--out", default="artifacts/phase3_fixed_highn/analysis/all_stages_long_shapeAB.tsv")
    args = ap.parse_args()

    root = Path(args.root)

    # precargar referencias A y B por par
    ref_A, ref_B = {}, {}
    for pair, (a_local, b_global) in PAIR_PARTS.items():
        ra = load_mol(A_SDF[a_local])
        rb = load_mol(B_SDF[b_global])
        if ra is None or rb is None:
            raise SystemExit(f"ERROR: no se pudo cargar ref A/B de {pair}")
        ref_A[pair] = ra
        ref_B[pair] = rb

    scaffolds = sorted(root.rglob("stage_*_scaffold.sdf"))
    print(f"scaffolds encontrados: {len(scaffolds)}")

    rows = []
    n_fail = 0
    for i, sc in enumerate(scaffolds):
        try:
            pair, add_n, branch, seed, rep, stage = parse_scaffold_path(sc, root)
        except Exception as e:
            n_fail += 1
            continue
        if pair not in PAIR_PARTS:
            continue
        mol = load_mol(sc)
        if mol is None:
            n_fail += 1
            continue
        anchor_heavy = mol.GetNumHeavyAtoms()
        tani_A, prot_A = shape_dists(mol, ref_A[pair])
        tani_B, prot_B = shape_dists(mol, ref_B[pair])
        rows.append({
            "pair": pair, "add_n": add_n, "branch": branch, "seed": seed,
            "rep": rep, "stage": stage, "anchor_heavy": anchor_heavy,
            "tani_A": round(tani_A, 4), "prot_A": round(prot_A, 4),
            "tani_B": round(tani_B, 4), "prot_B": round(prot_B, 4),
        })
        if (i + 1) % 200 == 0:
            print(f"  procesados {i+1}/{len(scaffolds)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "add_n", "branch", "seed", "rep", "stage", "anchor_heavy",
              "tani_A", "prot_A", "tani_B", "prot_B"]
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] {len(rows)} filas escritas en {out}")
    print(f"     fallos de parseo/carga: {n_fail}")
    # sanity: rango de tani_A y tani_B
    import statistics as st
    for col in ["tani_A", "tani_B"]:
        vals = [r[col] for r in rows if r[col] == r[col]]  # no NaN
        if vals:
            print(f"     {col}: min={min(vals):.3f} max={max(vals):.3f} "
                  f"media={st.mean(vals):.3f}")


if __name__ == "__main__":
    main()
