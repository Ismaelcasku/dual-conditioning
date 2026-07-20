#!/usr/bin/env python3
"""Auditoria de la ETAPA 1 del crecimiento por etapas.

Reutiliza la logica cientifica por-molecula del auditor de campana
(src/analysis/single_shot/audit_single_shot.py) SIN modificarlo: importa
sus funciones (fragment_data, match_fixed_atoms, interfragment_geometry,
add_shape_fields, sanitize_record, load_molecule, FIXED_INDICES, A_SDF, B_SDF)
y solo reimplementa el bucle de recorrido, adaptado al manifest de la etapa 1.

Diferencias respecto al main() original:
  - lee stage1_manifest.tsv (columnas: task_id, pair_id, ..., resamplings,
    add_n_nodes, out_sdf, ...), NO el manifest de 75 filas de Fase 0;
  - NO ejecuta el preflight de 75 filas ni el set de lambdas de Fase 0;
  - anade columnas de los ejes nuevos (resamplings, add_n_nodes) al TSV;
  - produce un TSV por molecula centrado en los observables de la etapa 1.

Uso (dentro del contenedor, desde la raiz del proyecto)
-------------------------------------------------------
    python src/growth/audit_stage1.py \
        --manifest artifacts/phase1_fragment_growing/stage1_manifest.tsv \
        --out_dir  artifacts/phase1_fragment_growing/stage1_audit

Requiere PYTHONPATH que incluya src/ y src/analysis/single_shot/ para poder importar el
auditor de campana (igual que el resto de scripts del proyecto).
"""

import argparse
import csv
import importlib.util
from pathlib import Path


AUDITOR_PATH = "src/analysis/single_shot/audit_single_shot.py"


def load_auditor(path):
    """Importa el modulo auditor sin ejecutar su main() (tiene el guard)."""
    spec = importlib.util.spec_from_file_location("campaign_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_one_record(aud, full_mol_raw, reference_a, reference_b, fixed_indices):
    """Aplica la logica por-molecula del auditor de campana a un registro.

    Devuelve un dict con las metricas por molecula, o None si el registro
    no es auditable (lectura/sanitize fallida).
    """
    full_mol, sanitize_status = aud.sanitize_record(full_mol_raw)

    row = {"sanitize_status": sanitize_status, "audit_ok": False}

    if (
        full_mol is None
        or sanitize_status != "OK"
        or full_mol.GetNumConformers() == 0
    ):
        return row

    fragments = aud.fragment_data(full_mol)
    heavy_fragment_ids = fragments["heavy_fragment_ids"]
    parent_id = fragments["parent_id"]

    total_heavy = full_mol.GetNumHeavyAtoms()
    parent_heavy = fragments["heavy_counts"][parent_id]

    assignments = aud.match_fixed_atoms(reference_a, full_mol, fixed_indices)

    matched_generated_indices = [gi for _, gi, _ in assignments]
    assignment_distances = [d for _, _, d in assignments]
    matched_fragment_ids = [
        fragments["atom_to_fragment"][gi] for gi in matched_generated_indices
    ]

    anchor_all_matched = len(assignments) == len(fixed_indices)
    anchor_all_same_fragment = (
        anchor_all_matched and len(set(matched_fragment_ids)) == 1
    )
    anchor_id = matched_fragment_ids[0] if anchor_all_same_fragment else None
    anchor_mol = (
        fragments["fragment_molecules"][anchor_id]
        if anchor_id is not None
        else None
    )
    anchor_heavy = (
        fragments["heavy_counts"][anchor_id] if anchor_id is not None else 0
    )

    row.update(
        {
            "audit_ok": True,
            "n_atoms_full": full_mol.GetNumAtoms(),
            "n_heavy_full": total_heavy,
            "n_total_fragments": len(fragments["atom_fragments"]),
            "n_heavy_fragments": len(heavy_fragment_ids),
            "heavy_connected": len(heavy_fragment_ids) == 1,
            "parent_fragment_id": parent_id,
            "parent_heavy_atoms": parent_heavy,
            "parent_heavy_fraction": (
                parent_heavy / total_heavy if total_heavy else 0.0
            ),
            "fixed_atoms_expected": len(fixed_indices),
            "fixed_atoms_matched": len(assignments),
            "fixed_atoms_within_0p2A": sum(
                d <= 0.2 for d in assignment_distances
            ),
            "fixed_match_max_distance": (
                max(assignment_distances) if assignment_distances else ""
            ),
            "anchor_all_same_fragment": anchor_all_same_fragment,
            "anchor_fragment_id": anchor_id if anchor_id is not None else "",
            "anchor_heavy_atoms": anchor_heavy,
            "anchor_heavy_fraction": (
                anchor_heavy / total_heavy if total_heavy else 0.0
            ),
            "anchor_component_is_parent": (
                anchor_id == parent_id if anchor_id is not None else False
            ),
        }
    )

    row.update(aud.interfragment_geometry(full_mol, fragments))

    aud.add_shape_fields(row, "full", full_mol, reference_a, reference_b)
    parent_mol = fragments["fragment_molecules"][parent_id]
    aud.add_shape_fields(row, "parent", parent_mol, reference_a, reference_b)
    aud.add_shape_fields(row, "anchor", anchor_mol, reference_a, reference_b)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--auditor", default=AUDITOR_PATH)
    args = parser.parse_args()

    aud = load_auditor(args.auditor)

    # Referencias A/B: reutilizamos las rutas y el loader del auditor de campana
    references_a = {
        key: aud.load_molecule(Path(path)) for key, path in aud.A_SDF.items()
    }
    references_b = {
        key: aud.load_molecule(Path(path)) for key, path in aud.B_SDF.items()
    }

    from rdkit import Chem

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open() as handle:
        design = list(csv.DictReader(handle, delimiter="\t"))

    molecule_rows = []

    for condition_number, condition in enumerate(design, start=1):
        sdf_path = Path(condition["out_sdf"])
        pair_id = condition["pair_id"]
        a_local = condition["A_local"]
        b_global = condition["B_global"]

        reference_a = references_a[a_local]
        reference_b = references_b[b_global]
        fixed_indices = aud.FIXED_INDICES[a_local]

        if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
            print(
                "[{}/{}] MISSING {}".format(
                    condition_number, len(design), sdf_path
                ),
                flush=True,
            )
            continue

        records = list(
            Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        )

        print(
            "[{}/{}] {} seed={} lambda={} resamp={} add_n={} records={}".format(
                condition_number,
                len(design),
                pair_id,
                condition["seed"],
                condition["lambda_global"],
                condition["resamplings"],
                condition["add_n_nodes"] if condition["add_n_nodes"] else "auto",
                len(records),
            ),
            flush=True,
        )

        for sample, original in enumerate(records, start=1):
            base = {
                "task_id": condition["task_id"],
                "pair_id": pair_id,
                "A_local": a_local,
                "B_global": b_global,
                "seed": condition["seed"],
                "lambda_global": condition["lambda_global"],
                "resamplings": condition["resamplings"],
                "add_n_nodes": (
                    condition["add_n_nodes"]
                    if condition["add_n_nodes"]
                    else "auto"
                ),
                "experiment_id": condition["experiment_id"],
                "sample": sample,
                "sdf_path": str(sdf_path),
            }

            metrics = audit_one_record(
                aud, original, reference_a, reference_b, fixed_indices
            )
            base.update(metrics)
            molecule_rows.append(base)

    # ------------------------------------------------------------------
    # Escritura del TSV por molecula. Campos: ejes nuevos primero, luego
    # las metricas cientificas (mismos nombres que el auditor de campana).
    # ------------------------------------------------------------------
    fields = [
        "task_id",
        "pair_id",
        "A_local",
        "B_global",
        "seed",
        "lambda_global",
        "resamplings",
        "add_n_nodes",
        "experiment_id",
        "sample",
        "sdf_path",
        "sanitize_status",
        "audit_ok",
        "n_atoms_full",
        "n_heavy_full",
        "n_total_fragments",
        "n_heavy_fragments",
        "heavy_connected",
        "parent_fragment_id",
        "parent_heavy_atoms",
        "parent_heavy_fraction",
        "fixed_atoms_expected",
        "fixed_atoms_matched",
        "fixed_atoms_within_0p2A",
        "fixed_match_max_distance",
        "anchor_all_same_fragment",
        "anchor_fragment_id",
        "anchor_heavy_atoms",
        "anchor_heavy_fraction",
        "anchor_component_is_parent",
        "minimum_heavy_interfragment_distance",
        "minimum_covalent_radius_ratio",
        "closest_atom_1",
        "closest_atom_2",
        "closest_atom_1_symbol",
        "closest_atom_2_symbol",
        "closest_pair_both_valence_headroom",
        "interfragment_class",
        "full_shape_ok",
        "full_tanimoto_A",
        "full_tanimoto_B",
        "full_protrude_A",
        "full_protrude_B",
        "full_strict_dual",
        "parent_shape_ok",
        "parent_tanimoto_A",
        "parent_tanimoto_B",
        "parent_protrude_A",
        "parent_protrude_B",
        "parent_strict_dual",
        "anchor_shape_ok",
        "anchor_tanimoto_A",
        "anchor_tanimoto_B",
        "anchor_protrude_A",
        "anchor_protrude_B",
        "anchor_strict_dual",
    ]

    # reutilizamos el escritor del auditor de campana para el mismo formato
    aud.write_tsv(
        out_dir / "stage1_audit_per_molecule.tsv", molecule_rows, fields
    )

    n_ok = sum(1 for r in molecule_rows if r.get("audit_ok"))
    n_connected = sum(1 for r in molecule_rows if r.get("heavy_connected"))
    n_anchor_same = sum(
        1 for r in molecule_rows if r.get("anchor_all_same_fragment")
    )

    print()
    print("STAGE1_AUDIT_DONE")
    print("molecules_total={}".format(len(molecule_rows)))
    print("auditable={}".format(n_ok))
    print("heavy_connected={}".format(n_connected))
    print("anchor_all_same_fragment={}".format(n_anchor_same))
    print("out_dir={}".format(out_dir))


if __name__ == "__main__":
    main()
