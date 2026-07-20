#!/usr/bin/env python3
"""Verify the pinned DiffSBDD checkout, guidance patches, and soft-mask overlay."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

PINNED_COMMIT = "5d0d38d16c8932a0339fd2ce3f67ade98bbdff27"
EXPECTED = {
    "equivariant_diffusion/conditional_model.py":
        "d8fcced3e17352357f18a1a4cd737f1b5f2caa48117d2d3ba66e1bbb5e20e441",
    "inpaint.py":
        "659c328fde62ad59fe913938e66f40179857a17267c8d9158c8da6f1d3f305d6",
    "lightning_modules.py":
        "ae521cc7fae48c27582719b8881f83a094f815308236b0c38cd9e46bdb47c0b9",
    "equivariant_diffusion/conditional_model_softmask.py":
        "e8cd642f4fb1314540781c7eca960a22afcf393d634c3b0744dfbfcce99f5059",
    "inpaint_softmask.py":
        "61b5a4731395e9a724ad58a2f9b390d0c03d88129e501ea38cd175a6407cfc98",
    "softmask_atom_order.py":
        "f9a5aac8d6b25038c95a59be872a93e0d5b08453eb32fe22dc6d43c8fc34a69f",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkout",
        nargs="?",
        type=Path,
        default=Path("external/DiffSBDD"),
        help="Path to the patched DiffSBDD checkout.",
    )
    args = parser.parse_args()
    root = args.checkout.resolve()

    if not root.is_dir():
        raise SystemExit(f"DiffSBDD checkout not found: {root}")

    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"Unexpected DiffSBDD HEAD: {head} != {PINNED_COMMIT}")

    failed = False
    for rel, expected in EXPECTED.items():
        path = root / rel
        if not path.is_file():
            print(f"MISSING  {rel}")
            failed = True
            continue
        observed = sha256(path)
        status = "OK" if observed == expected else "MISMATCH"
        print(f"{status:8s} {rel}  {observed}")
        failed |= observed != expected

    if failed:
        return 1

    print("DiffSBDD guidance and soft-mask verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
