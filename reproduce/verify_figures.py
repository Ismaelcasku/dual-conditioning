#!/usr/bin/env python3
"""
Level-A figure check: regenerate all manuscript figures from the derived TSVs
and confirm each expected output file is produced. Run after verify_numbers.py.

Assumes the repository layout:
  figures/            <- figure scripts + figstyle.py
  data/derived/       <- exp0, exp1, exp2 TSVs

The figure scripts resolve their project root as the PARENT of the scripts dir,
so we invoke them with --project-root / explicit paths pointing at data/derived.

Run:
  python reproduce/verify_figures.py
Exit 0 if all five figures render, 1 otherwise.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGDIR = REPO / "figures"
DATA = REPO / "data/derived"

# (script, [expected output stems])
JOBS = [
    ("make_figure1_task_protocol_connectivity.py", ["figure1_task_protocol_connectivity"]),
    ("make_figure2_single_step_expansion.py", ["figure2_single_step_expansion"]),
    ("make_figure3_fixed_increment_summary.py", ["figure3_fixed_increment_summary"]),
    ("make_figure4_violin_distributions.py", ["figure4_violin_distributions"]),
    ("make_figure5_AB_plane.py", ["figure5_AB_plane"]),
]


def main() -> int:
    # figure scripts expect exp0/exp1/exp2 under the project root; the release
    # keeps them under data/derived, so we pass explicit paths where the script
    # supports it and otherwise symlink data/derived contents next to figures.
    # Simplest robust approach: run each script with cwd=figures and override
    # the project root via the documented --project-root / --* flags.
    failures = []
    for script, stems in JOBS:
        cmd = [sys.executable, str(FIGDIR / script)]
        if script.startswith("make_figure1"):
            cmd += ["--data-root", str(DATA / "exp0"),
                    "--out-prefix", str(FIGDIR / "figure1_task_protocol_connectivity")]
        elif script.startswith("make_figure2"):
            cmd += ["--grid", str(DATA / "exp1/data/stage1_summary_by_grid.tsv"),
                    "--per-molecule", str(DATA / "exp1/data/stage1_audit_per_molecule.tsv"),
                    "--out-prefix", str(FIGDIR / "figure2_single_step_expansion")]
        elif script.startswith("make_figure3"):
            cmd += ["--summary", str(DATA / "exp2/data/trajectory_summary.tsv"),
                    "--out-prefix", str(FIGDIR / "figure3_fixed_increment_summary")]
        elif script.startswith("make_figure4"):
            cmd += ["--stages", str(DATA / "exp2/data/all_stages_long_shapeAB.tsv"),
                    "--out-prefix", str(FIGDIR / "figure4_violin_distributions")]
        elif script.startswith("make_figure5"):
            cmd += ["--shape", str(DATA / "exp2/data/all_stages_long_shapeAB.tsv"),
                    "--topology", str(DATA / "exp2/data/all_stages_long.tsv"),
                    "--out-prefix", str(FIGDIR / "figure5_AB_plane")]
        r = subprocess.run(cmd, cwd=str(FIGDIR), capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(f"{script} FAILED:\n{r.stderr[-500:]}")
            continue
        for stem in stems:
            produced = any((FIGDIR / f"{stem}.{ext}").exists()
                           for ext in ("png", "pdf", "svg"))
            if not produced:
                failures.append(f"{script}: no output for {stem}")

    print(f"Figure check: {len(JOBS)} scripts, {len(failures)} failures.")
    if failures:
        for f in failures:
            print("  -", f)
        return 1
    print("All figures regenerated from derived TSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
