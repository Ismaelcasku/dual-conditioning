#!/usr/bin/env python3
"""
Level-B smoke test (STUB — requires GPU, DiffSBDD, checkpoint, and a reference
trajectory under reproduce/smoke_test/reference/). See smoke_test/README.md.

Once reference/ and tolerances.json exist, this script:
  - runs the pinned reference command,
  - recomputes shape with the manuscript convention (allowReordering=False),
  - compares per-stage tani_B/prot_B to the reference within tolerance.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "smoke_test" / "reference"
TOL = HERE / "smoke_test" / "tolerances.json"


def main() -> int:
    if not REF.exists() or not TOL.exists():
        print("Level-B smoke test not configured yet.")
        print(f"  Missing: {REF} and/or {TOL}")
        print("  See reproduce/smoke_test/README.md to populate the reference.")
        return 2  # 'skipped', not a failure
    # TODO: implement once reference data is provided.
    print("Reference present. Implement generation+compare here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
