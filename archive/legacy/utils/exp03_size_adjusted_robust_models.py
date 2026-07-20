#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path.cwd()
OUTDIR = (
    ROOT
    / "artifacts/reports/phase0_official_eval"
    / "exp03_shape_replicate_n20_stratified"
)
INPUT = OUTDIR / "exp03_combined_valid_rows.tsv"

METRICS = {
    "ShapeTaniB": "shape_tanimoto_dist_to_B",
    "ProtrudeB": "shape_protrude_dist_to_B",
}


def f(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def load_data():
    rows = []

    with INPUT.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            lam = f(row.get("lambda_global"))
            n_heavy = f(row.get("n_heavy"))

            if lam is None or n_heavy is None:
                continue

            record = {
                "lambda": lam,
                "n_heavy": n_heavy,
                "medium": 1.0 if 16 <= n_heavy <= 22 else 0.0,
                "large": 1.0 if n_heavy >= 23 else 0.0,
            }

            for label, column in METRICS.items():
                record[label] = f(row.get(column))

            rows.append(record)

    return rows


def ols_hc3(X, y, names):
    """
    OLS with HC3 heteroscedasticity-robust standard errors.
    Uses pseudoinverse, so it remains stable with near-collinear designs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    beta = np.linalg.pinv(X) @ y
    fitted = X @ beta
    residuals = y - fitted

    xtx_inv = np.linalg.pinv(X.T @ X)
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    leverage = np.clip(leverage, 0.0, 0.999999)

    hc3_scale = (residuals / (1.0 - leverage)) ** 2
    meat = X.T @ (X * hc3_scale[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    variances = np.clip(np.diag(covariance), 0.0, None)
    standard_errors = np.sqrt(variances)

    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(residuals ** 2)
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")

    output = []

    for name, estimate, se in zip(names, beta, standard_errors):
        output.append({
            "coefficient": name,
            "estimate": float(estimate),
            "robust_se_hc3": float(se),
            "ci95_low": float(estimate - 1.96 * se),
            "ci95_high": float(estimate + 1.96 * se),
            "n": len(y),
            "r_squared": float(r_squared),
        })

    return output


def fit_models(rows):
    results = []

    for metric in METRICS:
        valid = [r for r in rows if r[metric] is not None]
        y = np.asarray([r[metric] for r in valid], dtype=float)

        lambdas = np.asarray([r["lambda"] for r in valid])
        n_heavy = np.asarray([r["n_heavy"] for r in valid], dtype=float)
        n_centered = n_heavy - np.mean(n_heavy)

        i50 = (lambdas == 50.0).astype(float)
        i100 = (lambdas == 100.0).astype(float)

        # Model 1: separate lambda groups plus continuous molecular size.
        X_continuous = np.column_stack([
            np.ones(len(valid)),
            i50,
            i100,
            n_centered,
        ])

        continuous = ols_hc3(
            X_continuous,
            y,
            ["Intercept", "lambda_50_vs_0", "lambda_100_vs_0", "n_heavy"],
        )

        for row in continuous:
            row["metric"] = metric
            row["model"] = "lambda_groups_plus_continuous_n_heavy"
            results.append(row)

        # Model 2: separate lambda groups plus size-stratum fixed effects.
        medium = np.asarray([r["medium"] for r in valid], dtype=float)
        large = np.asarray([r["large"] for r in valid], dtype=float)

        X_strata = np.column_stack([
            np.ones(len(valid)),
            i50,
            i100,
            medium,
            large,
        ])

        strata = ols_hc3(
            X_strata,
            y,
            [
                "Intercept_small",
                "lambda_50_vs_0",
                "lambda_100_vs_0",
                "medium_vs_small",
                "large_vs_small",
            ],
        )

        for row in strata:
            row["metric"] = metric
            row["model"] = "lambda_groups_plus_size_strata"
            results.append(row)

        # Model 3: linear lambda trend per +50 units plus continuous size.
        lambda_per_50 = lambdas / 50.0

        X_trend = np.column_stack([
            np.ones(len(valid)),
            lambda_per_50,
            n_centered,
        ])

        trend = ols_hc3(
            X_trend,
            y,
            ["Intercept", "lambda_per_50", "n_heavy"],
        )

        for row in trend:
            row["metric"] = metric
            row["model"] = "linear_lambda_trend_plus_continuous_n_heavy"
            results.append(row)

    return results


def write_results(results):
    path = OUTDIR / "exp03_size_adjusted_robust_models.tsv"

    fields = [
        "metric",
        "model",
        "coefficient",
        "estimate",
        "robust_se_hc3",
        "ci95_low",
        "ci95_high",
        "n",
        "r_squared",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    return path


def write_summary(results):
    path = OUTDIR / "exp03_size_adjusted_key_effects.tsv"

    selected = [
        row for row in results
        if row["coefficient"] in {
            "lambda_50_vs_0",
            "lambda_100_vs_0",
            "lambda_per_50",
            "n_heavy",
        }
    ]

    fields = [
        "metric",
        "model",
        "coefficient",
        "estimate",
        "robust_se_hc3",
        "ci95_low",
        "ci95_high",
        "n",
        "r_squared",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

    return path, selected


def main():
    rows = load_data()

    if len(rows) != 57:
        print(f"WARNING: expected 57 valid rows, found {len(rows)}")

    results = fit_models(rows)
    full_path = write_results(results)
    key_path, selected = write_summary(results)

    print(f"ROBUST_MODELS_DONE rows={len(rows)}")
    print(f"full={full_path}")
    print(f"key={key_path}")

    print("\n=== KEY ADJUSTED EFFECTS ===")

    for row in selected:
        if row["coefficient"] not in {
            "lambda_50_vs_0",
            "lambda_100_vs_0",
            "lambda_per_50",
        }:
            continue

        direction = (
            "toward_B"
            if row["estimate"] < 0
            else "away_from_B"
        )

        print(
            f"{row['metric']}\t"
            f"{row['model']}\t"
            f"{row['coefficient']}\t"
            f"estimate={row['estimate']:.4f}\t"
            f"CI95=[{row['ci95_low']:.4f},"
            f"{row['ci95_high']:.4f}]\t"
            f"{direction}"
        )


if __name__ == "__main__":
    main()
