#!/usr/bin/env python3
"""Run the local regression suite for the two adaptive growth phases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def run(path: Path) -> None:
    print(f"\n[run] {path.relative_to(PROJECT)}", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=PROJECT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase4-only",
        action="store_true",
        help="Run only the CPU-compatible variable-increment logic tests.",
    )
    args = parser.parse_args()

    run(PROJECT / "tests" / "test_phase4_beam_logic.py")
    if not args.phase4_only:
        run(PROJECT / "tests" / "test_softmask_core.py")
        run(PROJECT / "tests" / "test_softmask_order.py")

    print("\nAdaptive-search verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
