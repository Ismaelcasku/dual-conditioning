#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/make_figure1_task_protocol_connectivity.py"
python "$HERE/make_figure2_single_step_expansion.py"
python "$HERE/make_figure3_fixed_increment_summary.py"
python "$HERE/make_figure4_violin_distributions.py"
python "$HERE/make_figure5_AB_plane.py"
echo "All figures written to $HERE"
