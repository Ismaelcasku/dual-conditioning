#!/usr/bin/env python3
"""Audit one generated SDF against A, B, and a frozen local anchor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from rdkit import Chem

from dual_conditioning.evaluation.audit import audit_record
from dual_conditioning.evaluation.shape import ShapeMetricConfig


def load_first_molecule(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise RuntimeError(f"no readable molecule in {path}")
    Chem.SanitizeMol(molecule)
    if molecule.GetNumConformers() == 0:
        raise RuntimeError(f"molecule has no conformer: {path}")
    return molecule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference-a", type=Path, required=True)
    parser.add_argument("--reference-b", type=Path, required=True)
    parser.add_argument("--fixed-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-threshold", type=float, default=0.2)
    parser.add_argument("--bond-ratio-threshold", type=float, default=1.25)
    parser.add_argument("--close-ratio-threshold", type=float, default=1.75)
    parser.add_argument(
        "--allow-protrude-reordering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True reproduces the frozen evaluator; false gives fixed-direction protrusion.",
    )
    args = parser.parse_args()

    reference_a = load_first_molecule(args.reference_a)
    reference_b = load_first_molecule(args.reference_b)
    fixed = json.loads(args.fixed_json.read_text(encoding="utf-8"))
    fixed_indices = fixed["fixed_atom_indices_0based"]
    metric_config = ShapeMetricConfig(
        allow_protrude_reordering=args.allow_protrude_reordering
    )

    supplier = Chem.SDMolSupplier(str(args.generated), sanitize=False, removeHs=False)
    rows: list[dict[str, object]] = []
    for record_index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            rows.append({
                "record_index": record_index,
                "read_ok": False,
                "sanitize_ok": False,
                "error": "RDKit supplier returned None",
            })
            continue
        copied = Chem.Mol(molecule)
        sanitize_ok = True
        error = ""
        try:
            Chem.SanitizeMol(copied)
        except Exception as exc:
            sanitize_ok = False
            error = repr(exc)
        try:
            row = audit_record(
                copied,
                reference_a,
                reference_b,
                fixed_indices,
                metric_config=metric_config,
                local_threshold_angstrom=args.local_threshold,
                bond_ratio_threshold=args.bond_ratio_threshold,
                close_ratio_threshold=args.close_ratio_threshold,
            )
            row.update({
                "record_index": record_index,
                "read_ok": True,
                "sanitize_ok": sanitize_ok,
                "error": error,
            })
        except Exception as exc:
            row = {
                "record_index": record_index,
                "read_ok": True,
                "sanitize_ok": sanitize_ok,
                "error": f"AUDIT_FAIL: {exc!r}",
            }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    preferred = ["record_index", "read_ok", "sanitize_ok", "error"]
    fields = preferred + [field for field in fields if field not in preferred]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
