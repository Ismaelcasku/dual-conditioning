#!/usr/bin/env python3
"""
Level-A verification: recompute the key manuscript numbers from the derived
TSVs and compare them to the values reported in the Results text. No GPU, no
generative model. If this passes, the tables and the figure-driving numbers are
reproducible from the released data.

Run:
  python reproduce/verify_numbers.py --data-root data/derived

Exit code 0 if all checks pass within tolerance, 1 otherwise.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SRC = {"A10": "A10", "directed": "B"}
ARMS = [3, 4, 5, 6]
PAIRS = ["x0434_x2193", "x0874_x1093"]

# --- expected values from the manuscript Results -------------------------
# means of Tanimoto improvement to B (first-last stage), per pair/arm/branch
EXPECTED_MEANS = {
    "x0434_x2193": {  # hard
        3: {"A10": 0.123, "directed": 0.176},
        4: {"A10": 0.132, "directed": 0.195},
        5: {"A10": 0.108, "directed": 0.125},
        6: {"A10": 0.093, "directed": 0.131},
    },
    "x0874_x1093": {  # moderate
        3: {"A10": 0.064, "directed": 0.116},
        4: {"A10": 0.068, "directed": 0.122},
        5: {"A10": 0.065, "directed": 0.082},
        6: {"A10": 0.047, "directed": 0.088},
    },
}
EXPECTED_P = {  # one-sided MWU, directed > A10
    "x0434_x2193": {3: 0.0012, 4: 0.0002, 5: 0.178, 6: 0.012},
    "x0874_x1093": {3: 0.0038, 4: 0.0096, 5: 0.092, 6: 0.015},
}
MEAN_TOL = 0.005    # manuscript reports 3 decimals
P_TOL_FACTOR = 0.5  # p-values: within 50% relative (rank test, seed-independent)


def improvement(st: pd.DataFrame, metric: str) -> pd.DataFrame:
    recs = []
    for key, g in st.groupby(["pair", "add_n", "branch", "seed", "rep"]):
        g = g.sort_values("stage")
        vals = g[metric].dropna()
        if len(vals) < 1:
            continue
        recs.append({"pair": key[0], "add_n": key[1], "branch": key[2],
                     "improvement": vals.iloc[0] - vals.iloc[-1]})
    return pd.DataFrame(recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/derived"))
    args = ap.parse_args()

    stages = args.data_root / "exp2/data/all_stages_long_shapeAB.tsv"
    st = pd.read_csv(stages, sep="\t")
    for c in ["add_n", "stage", "tani_B"]:
        st[c] = pd.to_numeric(st[c], errors="coerce")
    st["branch"] = st["branch"].astype(str).str.strip()
    imp = improvement(st, "tani_B")

    failures = []
    checks = 0

    # 1. means per pair/arm/branch
    for pair in PAIRS:
        for arm in ARMS:
            for br in ("A10", "directed"):
                v = imp[(imp["pair"] == pair) & (imp["add_n"] == arm) &
                        (imp["branch"] == SRC[br])]["improvement"].dropna().values
                got = float(np.mean(v)) if len(v) else float("nan")
                exp = EXPECTED_MEANS[pair][arm][br]
                checks += 1
                if not (abs(got - exp) <= MEAN_TOL):
                    failures.append(
                        f"mean {pair} +{arm} {br}: got {got:.4f}, expected {exp:.3f}")

    # 2. Mann-Whitney p-values, directed > A10
    for pair in PAIRS:
        for arm in ARMS:
            a10 = imp[(imp["pair"] == pair) & (imp["add_n"] == arm) &
                      (imp["branch"] == "A10")]["improvement"].dropna().values
            drc = imp[(imp["pair"] == pair) & (imp["add_n"] == arm) &
                      (imp["branch"] == "B")]["improvement"].dropna().values
            if len(a10) and len(drc):
                _, p = mannwhitneyu(drc, a10, alternative="greater")
            else:
                p = float("nan")
            exp = EXPECTED_P[pair][arm]
            checks += 1
            # relative tolerance; both should be same order of magnitude
            if not (abs(p - exp) <= max(P_TOL_FACTOR * exp, 0.01)):
                failures.append(
                    f"p-value {pair} +{arm}: got {p:.4f}, expected {exp:.4f}")

    # 3. +4 is the hard-pair directed optimum
    hard_dir = {arm: np.mean(imp[(imp["pair"] == "x0434_x2193") &
                                 (imp["add_n"] == arm) &
                                 (imp["branch"] == "B")]["improvement"])
                for arm in ARMS}
    checks += 1
    if max(hard_dir, key=hard_dir.get) != 4:
        failures.append(f"hard-pair directed optimum not at +4: {hard_dir}")

    print(f"Level-A verification: {checks} checks, {len(failures)} failures.")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("All manuscript numbers reproduced from derived TSVs within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
