#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--expected",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    files = sorted(
        root.glob(
            "*/summary.tsv"
        )
    )

    if not files:
        raise RuntimeError(
            f"No summaries found under {root}"
        )

    dataframe = pd.concat(
        [
            pd.read_csv(
                path,
                sep="\t",
            )
            for path in files
        ],
        ignore_index=True,
    )

    dataframe = dataframe.sort_values(
        by=[
            "warhead_pass_0p2A",
            "delta_shape_tanimoto_similarity_B",
            "after_clashes",
            "after_strain",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        na_position="last",
    )

    complete_tsv = (
        root
        / "complete_summary.tsv"
    )

    complete_csv = (
        root
        / "complete_summary.csv"
    )

    compact_tsv = (
        root
        / "compact_summary.tsv"
    )

    dataframe.to_csv(
        complete_tsv,
        sep="\t",
        index=False,
    )

    dataframe.to_csv(
        complete_csv,
        index=False,
    )

    compact_columns = [
        "candidate_id",
        "group",
        "warhead_rmsd_A",
        "warhead_pass_0p2A",
        "ligand_heavy_rmsd_A",
        "before_clashes",
        "after_clashes",
        "delta_clashes",
        "before_strain",
        "after_strain",
        "delta_strain",
        "before_shape_tanimoto_similarity_B",
        "after_shape_tanimoto_similarity_B",
        "delta_shape_tanimoto_similarity_B",
        "before_shape_protrude_distance_B",
        "after_shape_protrude_distance_B",
        "delta_shape_protrude_distance_B",
        "before_interaction_count",
        "after_interaction_count",
        "interaction_jaccard",
    ]

    compact = dataframe[
        compact_columns
    ]

    compact.to_csv(
        compact_tsv,
        sep="\t",
        index=False,
    )

    print(
        compact.to_string(
            index=False
        )
    )

    print()
    print(
        f"completed={len(dataframe)}"
    )
    print(
        f"expected={args.expected}"
    )
    print(
        f"compact_tsv={compact_tsv}"
    )

    if len(dataframe) != args.expected:
        print(
            "WARNING: missing array results."
        )

    print(
        "POSTMIN_AGGREGATION_STATUS=OK"
    )


if __name__ == "__main__":
    main()
