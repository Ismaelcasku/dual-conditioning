#!/usr/bin/env python3
"""Run the corrected per-record audit over a campaign manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from rdkit import Chem

from dual_conditioning.evaluation.audit import audit_record
from dual_conditioning.evaluation.shape import ShapeMetricConfig


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def derive_a_ligand(row: dict[str, str]) -> str:
    if row.get("A_ligand"):
        return row["A_ligand"]
    complex_path = row.get("A_complex", "")
    if complex_path.endswith("_complex.pdb"):
        return complex_path.removesuffix("_complex.pdb") + "_ligand.sdf"
    raise ValueError("manifest requires A_ligand or an A_complex ending in _complex.pdb")


def load_first(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise RuntimeError(f"no readable molecule in {path}")
    Chem.SanitizeMol(molecule)
    return molecule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-protrude-reordering", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    metric_config = ShapeMetricConfig(
        allow_protrude_reordering=args.allow_protrude_reordering
    )
    output_rows: list[dict[str, object]] = []
    reference_cache: dict[Path, Chem.Mol] = {}

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

    for design in manifest_rows:
        generated_path = resolve(root, design.get("release_sdf_path") or design["sdf_path"])
        a_path = resolve(root, derive_a_ligand(design))
        b_path = resolve(root, design["B_ligand"])
        fixed_path = resolve(root, design["fixed_json"])
        for path in (a_path, b_path):
            if path not in reference_cache:
                reference_cache[path] = load_first(path)
        fixed_indices = json.loads(fixed_path.read_text(encoding="utf-8"))["fixed_atom_indices_0based"]
        supplier = Chem.SDMolSupplier(str(generated_path), sanitize=False, removeHs=False)
        for record_index, molecule in enumerate(supplier, start=1):
            base = {
                key: design.get(key, "")
                for key in ("grid_id", "pair_id", "A_local", "B_global", "seed", "lambda_global", "experiment_id")
            }
            base["record_index"] = record_index
            if molecule is None:
                base.update({"read_ok": False, "error": "RDKit supplier returned None"})
                output_rows.append(base)
                continue
            copied = Chem.Mol(molecule)
            try:
                Chem.SanitizeMol(copied)
                base["sanitize_ok"] = True
            except Exception as exc:
                base["sanitize_ok"] = False
                base["sanitize_error"] = repr(exc)
            try:
                base.update(audit_record(
                    copied,
                    reference_cache[a_path],
                    reference_cache[b_path],
                    fixed_indices,
                    metric_config=metric_config,
                ))
                base["read_ok"] = True
            except Exception as exc:
                base.update({"read_ok": True, "error": f"AUDIT_FAIL: {exc!r}"})
            output_rows.append(base)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in output_rows for field in row})
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
