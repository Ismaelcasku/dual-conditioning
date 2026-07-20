#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()

OUTDIR = (
    ROOT
    / "artifacts/reports/phase0_official_eval"
    / "exp03_shape_replicate_n20_stratified"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    0.0: ROOT / (
        "artifacts/reports/phase0_official_eval/"
        "exp03_shape_replicate_n20_lambda_0.0/"
        "exp03_shape_replicate_n20_lambda_0.0_official_phase0_metrics.tsv"
    ),
    50.0: ROOT / (
        "artifacts/reports/phase0_official_eval/"
        "exp03_shape_replicate_n20_lambda_50.0/"
        "exp03_shape_replicate_n20_lambda_50.0_official_phase0_metrics.tsv"
    ),
    100.0: ROOT / (
        "artifacts/reports/phase0_official_eval/"
        "exp03_shape_replicate_n20_lambda_100.0/"
        "exp03_shape_replicate_n20_lambda_100.0_official_phase0_metrics.tsv"
    ),
}

METRICS = [
    "shape_tanimoto_dist_to_B",
    "shape_protrude_dist_to_B",
    "shape_tanimoto_dist_to_A",
    "shape_protrude_dist_to_A",
]

SIZE_ORDER = [
    "Small (≤15)",
    "Medium (16–22)",
    "Large (≥23)",
]


def size_bin(n_heavy):
    if n_heavy <= 15:
        return "Small (≤15)"
    if n_heavy <= 22:
        return "Medium (16–22)"
    return "Large (≥23)"


def safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rankdata(values):
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1

        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = average_rank
        i = j + 1

    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")

    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def ols_adjusted(rows, metric):
    """
    Descriptive OLS:
        metric = intercept
               + beta_lambda50 * (lambda_global / 50)
               + beta_nheavy * (n_heavy - mean_n_heavy)

    beta_lambda50 is the adjusted change in distance for each +50 lambda units.
    Negative beta means greater similarity to the reference.
    """
    valid = [
        r for r in rows
        if r.get(metric) is not None
        and r.get("n_heavy") is not None
        and r.get("lambda_global") is not None
    ]

    n = len(valid)
    if n < 5:
        return None

    y = np.asarray([r[metric] for r in valid], dtype=float)
    lambda50 = np.asarray([r["lambda_global"] / 50.0 for r in valid], dtype=float)
    nheavy = np.asarray([r["n_heavy"] for r in valid], dtype=float)
    nheavy_centered = nheavy - nheavy.mean()

    X = np.column_stack([
        np.ones(n),
        lambda50,
        nheavy_centered,
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted

    p = X.shape[1]
    dof = n - p
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    if dof > 0:
        sigma2 = rss / dof
        covariance = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(covariance))
    else:
        se = np.full(p, np.nan)

    return {
        "n": n,
        "dof": dof,
        "intercept": float(beta[0]),
        "beta_lambda_per_50": float(beta[1]),
        "beta_lambda_per_50_se": float(se[1]),
        "beta_lambda_per_50_ci_low": float(beta[1] - 1.96 * se[1]),
        "beta_lambda_per_50_ci_high": float(beta[1] + 1.96 * se[1]),
        "beta_n_heavy": float(beta[2]),
        "beta_n_heavy_se": float(se[2]),
        "beta_n_heavy_ci_low": float(beta[2] - 1.96 * se[2]),
        "beta_n_heavy_ci_high": float(beta[2] + 1.96 * se[2]),
        "r_squared": float(r2),
        "mean_n_heavy": float(nheavy.mean()),
    }


def load_rows():
    combined = []

    for lambda_value, path in RUNS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics TSV: {path}")

        with path.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")

            for row in reader:
                if row.get("valid_rdkit_read") != "TRUE":
                    continue

                n_heavy = safe_float(row.get("n_heavy"))
                if n_heavy is None:
                    continue

                record = {
                    "lambda_global": lambda_value,
                    "sample": int(row["sample"]),
                    "n_heavy": n_heavy,
                    "size_bin": size_bin(n_heavy),
                    "local_pass": row.get("local_pass_all_atoms_0p2A") == "TRUE",
                }

                for metric in METRICS:
                    record[metric] = safe_float(row.get(metric))

                combined.append(record)

    return combined


def write_combined(rows):
    path = OUTDIR / "exp03_combined_valid_rows.tsv"
    fields = [
        "lambda_global",
        "sample",
        "n_heavy",
        "size_bin",
        "local_pass",
        *METRICS,
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return path


def stratified_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["lambda_global"], row["size_bin"])].append(row)

    output = []

    for lambda_value in sorted(RUNS):
        for bin_name in SIZE_ORDER:
            group = grouped.get((lambda_value, bin_name), [])
            record = {
                "lambda_global": lambda_value,
                "size_bin": bin_name,
                "n": len(group),
                "local_pass_n": sum(r["local_pass"] for r in group),
            }

            heavy = [r["n_heavy"] for r in group]
            record["n_heavy_mean"] = float(np.mean(heavy)) if heavy else None
            record["n_heavy_median"] = float(np.median(heavy)) if heavy else None

            for metric in METRICS:
                values = [r[metric] for r in group if r[metric] is not None]

                record[f"{metric}_mean"] = float(np.mean(values)) if values else None
                record[f"{metric}_median"] = float(np.median(values)) if values else None
                record[f"{metric}_sd"] = sample_sd(values) if values else None
                record[f"{metric}_q25"] = percentile(values, 25) if values else None
                record[f"{metric}_q75"] = percentile(values, 75) if values else None

            output.append(record)

    return output


def write_stratified_summary(summary):
    path = OUTDIR / "exp03_stratified_summary.tsv"

    fields = [
        "lambda_global",
        "size_bin",
        "n",
        "local_pass_n",
        "n_heavy_mean",
        "n_heavy_median",
    ]

    for metric in METRICS:
        fields.extend([
            f"{metric}_mean",
            f"{metric}_median",
            f"{metric}_sd",
            f"{metric}_q25",
            f"{metric}_q75",
        ])

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)

    return path


def pairwise_within_bins(summary):
    indexed = {
        (r["lambda_global"], r["size_bin"]): r
        for r in summary
    }

    rows = []

    for bin_name in SIZE_ORDER:
        baseline = indexed.get((0.0, bin_name))

        for lambda_value in [50.0, 100.0]:
            guided = indexed.get((lambda_value, bin_name))

            record = {
                "size_bin": bin_name,
                "lambda_global": lambda_value,
                "baseline_n": baseline["n"] if baseline else 0,
                "guided_n": guided["n"] if guided else 0,
            }

            for metric in [
                "shape_tanimoto_dist_to_B",
                "shape_protrude_dist_to_B",
            ]:
                base_mean = baseline.get(f"{metric}_mean") if baseline else None
                guided_mean = guided.get(f"{metric}_mean") if guided else None

                record[f"{metric}_baseline_mean"] = base_mean
                record[f"{metric}_guided_mean"] = guided_mean
                record[f"{metric}_delta_vs_baseline"] = (
                    guided_mean - base_mean
                    if base_mean is not None and guided_mean is not None
                    else None
                )

            rows.append(record)

    return rows


def write_pairwise(rows):
    path = OUTDIR / "exp03_within_size_bin_deltas.tsv"

    fields = [
        "size_bin",
        "lambda_global",
        "baseline_n",
        "guided_n",
        "shape_tanimoto_dist_to_B_baseline_mean",
        "shape_tanimoto_dist_to_B_guided_mean",
        "shape_tanimoto_dist_to_B_delta_vs_baseline",
        "shape_protrude_dist_to_B_baseline_mean",
        "shape_protrude_dist_to_B_guided_mean",
        "shape_protrude_dist_to_B_delta_vs_baseline",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return path


def correlation_table(rows):
    output = []

    for lambda_value in ["all", 0.0, 50.0, 100.0]:
        subset = (
            rows
            if lambda_value == "all"
            else [r for r in rows if r["lambda_global"] == lambda_value]
        )

        for metric in METRICS:
            paired = [
                (r["n_heavy"], r[metric])
                for r in subset
                if r[metric] is not None
            ]

            x = [p[0] for p in paired]
            y = [p[1] for p in paired]
            pearson, spearman = correlation(x, y)

            output.append({
                "lambda_global": lambda_value,
                "metric": metric,
                "n": len(paired),
                "pearson_r": pearson,
                "spearman_rho": spearman,
            })

    return output


def write_correlations(rows):
    path = OUTDIR / "exp03_nheavy_correlations.tsv"
    fields = [
        "lambda_global",
        "metric",
        "n",
        "pearson_r",
        "spearman_rho",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return path


def adjusted_models(rows):
    minima = {
        lam: min(r["n_heavy"] for r in rows if r["lambda_global"] == lam)
        for lam in RUNS
    }
    maxima = {
        lam: max(r["n_heavy"] for r in rows if r["lambda_global"] == lam)
        for lam in RUNS
    }

    common_low = max(minima.values())
    common_high = min(maxima.values())

    common_rows = [
        r for r in rows
        if common_low <= r["n_heavy"] <= common_high
    ]

    output = []

    for dataset_name, dataset in [
        ("all_valid_rows", rows),
        (f"common_size_support_{common_low:.0f}_{common_high:.0f}", common_rows),
    ]:
        for metric in [
            "shape_tanimoto_dist_to_B",
            "shape_protrude_dist_to_B",
        ]:
            result = ols_adjusted(dataset, metric)
            if result is None:
                continue

            result["dataset"] = dataset_name
            result["metric"] = metric
            result["common_size_low"] = common_low
            result["common_size_high"] = common_high
            output.append(result)

    return output, common_low, common_high


def write_adjusted_models(rows):
    path = OUTDIR / "exp03_adjusted_lambda_effects.tsv"

    fields = [
        "dataset",
        "metric",
        "n",
        "dof",
        "intercept",
        "beta_lambda_per_50",
        "beta_lambda_per_50_se",
        "beta_lambda_per_50_ci_low",
        "beta_lambda_per_50_ci_high",
        "beta_n_heavy",
        "beta_n_heavy_se",
        "beta_n_heavy_ci_low",
        "beta_n_heavy_ci_high",
        "r_squared",
        "mean_n_heavy",
        "common_size_low",
        "common_size_high",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return path


def make_plot(summary, metric, ylabel, stem):
    lookup = {
        (r["lambda_global"], r["size_bin"]): r
        for r in summary
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for bin_name in SIZE_ORDER:
        xs = []
        ys = []
        yerr = []

        for lambda_value in sorted(RUNS):
            row = lookup[(lambda_value, bin_name)]
            mean_value = row.get(f"{metric}_mean")

            if row["n"] == 0 or mean_value is None:
                continue

            xs.append(lambda_value)
            ys.append(mean_value)
            yerr.append(row.get(f"{metric}_sd") or 0.0)

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                capsize=4,
                linewidth=1.8,
                label=bin_name,
            )

    ax.set_xlabel("Global guidance strength (λ)")
    ax.set_ylabel(ylabel)
    ax.set_title("Endpoint similarity to B stratified by molecular size")
    ax.set_xticks([0, 50, 100])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    png = OUTDIR / f"{stem}.png"
    pdf = OUTDIR / f"{stem}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def fmt(value, digits=3):
    if value is None:
        return "NA"
    try:
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(
    rows,
    summary,
    pairwise,
    correlations,
    adjusted,
    common_low,
    common_high,
    output_paths,
):
    report = OUTDIR / "exp03_stratified_size_report.md"

    lines = [
        "# Exp03 high-lambda replicate: size-stratified analysis\n\n",
        "Lower shape distances indicate greater similarity to B.\n\n",
        "## Dataset\n\n",
        f"- Valid molecules: {len(rows)}\n",
        f"- Common molecular-size support across λ groups: "
        f"`{common_low:.0f}–{common_high:.0f}` heavy atoms\n",
        "- Size strata: `≤15`, `16–22`, and `≥23` heavy atoms\n\n",
        "## Stratified summary\n\n",
        "| λ | Size stratum | n | ShapeTaniB mean | ShapeTaniB median | "
        "ProtrudeB mean | ProtrudeB median |\n",
        "|---:|---|---:|---:|---:|---:|---:|\n",
    ]

    for row in summary:
        lines.append(
            f"| {row['lambda_global']:.0f} | {row['size_bin']} | {row['n']} | "
            f"{fmt(row['shape_tanimoto_dist_to_B_mean'])} | "
            f"{fmt(row['shape_tanimoto_dist_to_B_median'])} | "
            f"{fmt(row['shape_protrude_dist_to_B_mean'])} | "
            f"{fmt(row['shape_protrude_dist_to_B_median'])} |\n"
        )

    lines.extend([
        "\n## Within-size-stratum changes versus λ=0\n\n",
        "Negative deltas indicate improvement toward B.\n\n",
        "| Size stratum | λ | Baseline n | Guided n | "
        "Δ ShapeTaniB | Δ ProtrudeB |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])

    for row in pairwise:
        lines.append(
            f"| {row['size_bin']} | {row['lambda_global']:.0f} | "
            f"{row['baseline_n']} | {row['guided_n']} | "
            f"{fmt(row['shape_tanimoto_dist_to_B_delta_vs_baseline'])} | "
            f"{fmt(row['shape_protrude_dist_to_B_delta_vs_baseline'])} |\n"
        )

    lines.extend([
        "\n## Size-adjusted descriptive regressions\n\n",
        "Model: `distance = intercept + βλ × (λ/50) + βsize × centered n_heavy`.\n\n",
        "A negative `βλ` indicates improvement toward B after adjusting for molecular size. "
        "Confidence intervals are approximate OLS 95% intervals.\n\n",
        "| Dataset | Metric | n | βλ per +50 | Approx. 95% CI | "
        "β per heavy atom | R² |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ])

    for row in adjusted:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['n']} | "
            f"{fmt(row['beta_lambda_per_50'])} | "
            f"[{fmt(row['beta_lambda_per_50_ci_low'])}, "
            f"{fmt(row['beta_lambda_per_50_ci_high'])}] | "
            f"{fmt(row['beta_n_heavy'])} | "
            f"{fmt(row['r_squared'])} |\n"
        )

    lines.extend([
        "\n## Interpretation constraints\n\n",
        "- This is a descriptive analysis of independently generated molecules, not a paired design.\n",
        "- Sparse strata must not be overinterpreted; inspect `n` before comparing means.\n",
        "- A persistent negative λ coefficient after adjusting for `n_heavy` supports genuine "
        "shape steering beyond simple molecular growth.\n",
        "- If the λ coefficient approaches zero after adjustment, much of the endpoint improvement "
        "may be mediated by molecular size.\n\n",
        "## Outputs\n\n",
    ])

    for label, path in output_paths.items():
        lines.append(f"- {label}: `{path.relative_to(ROOT)}`\n")

    report.write_text("".join(lines))
    return report


def main():
    rows = load_rows()
    combined_path = write_combined(rows)

    summary = stratified_summary(rows)
    summary_path = write_stratified_summary(summary)

    pairwise = pairwise_within_bins(summary)
    pairwise_path = write_pairwise(pairwise)

    correlations = correlation_table(rows)
    correlations_path = write_correlations(correlations)

    adjusted, common_low, common_high = adjusted_models(rows)
    adjusted_path = write_adjusted_models(adjusted)

    tani_png, tani_pdf = make_plot(
        summary,
        metric="shape_tanimoto_dist_to_B",
        ylabel="ShapeTanimoto distance to B",
        stem="exp03_shape_tanimoto_B_by_size",
    )

    protrude_png, protrude_pdf = make_plot(
        summary,
        metric="shape_protrude_dist_to_B",
        ylabel="ShapeProtrude distance to B",
        stem="exp03_shape_protrude_B_by_size",
    )

    output_paths = {
        "Combined valid rows": combined_path,
        "Stratified summary": summary_path,
        "Within-bin deltas": pairwise_path,
        "Size correlations": correlations_path,
        "Adjusted λ effects": adjusted_path,
        "ShapeTanimoto figure PNG": tani_png,
        "ShapeTanimoto figure PDF": tani_pdf,
        "ShapeProtrude figure PNG": protrude_png,
        "ShapeProtrude figure PDF": protrude_pdf,
    }

    report = write_report(
        rows,
        summary,
        pairwise,
        correlations,
        adjusted,
        common_low,
        common_high,
        output_paths,
    )

    print("STRATIFIED_SIZE_ANALYSIS_DONE")
    print(f"valid_rows={len(rows)}")
    print(f"common_size_support={common_low:.0f}-{common_high:.0f}")
    print(f"report={report}")

    print("\n=== SIZE-ADJUSTED LAMBDA EFFECTS ===")
    for row in adjusted:
        print(
            f"{row['dataset']}\t{row['metric']}\t"
            f"n={row['n']}\t"
            f"beta_lambda_per_50={row['beta_lambda_per_50']:.4f}\t"
            f"CI=[{row['beta_lambda_per_50_ci_low']:.4f},"
            f"{row['beta_lambda_per_50_ci_high']:.4f}]\t"
            f"beta_nheavy={row['beta_n_heavy']:.4f}\t"
            f"R2={row['r_squared']:.3f}"
        )


if __name__ == "__main__":
    main()
