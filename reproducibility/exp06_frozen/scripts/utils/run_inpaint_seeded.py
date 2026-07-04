#!/usr/bin/env python3

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Python, NumPy and PyTorch before executing DiffSBDD."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("script", help="Python script to execute")

    args, script_args = parser.parse_known_args()

    seed = int(args.seed)
    script = Path(args.script).resolve()

    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Favorece reproducibilidad sin forzar operaciones deterministas que
    # podrían no estar implementadas por el modelo.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(
        f"[SEEDED_RUN] seed={seed} "
        f"python_hash_seed={os.environ.get('PYTHONHASHSEED')} "
        f"cuda_available={torch.cuda.is_available()} "
        f"script={script}",
        flush=True,
    )

    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *script_args]

    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
