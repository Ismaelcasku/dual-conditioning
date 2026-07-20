# Phase 4 and Phase 5 repository update

This update integrates the two adaptive staged-growth extensions into the clean
release repository.

## Phase 4 — variable-increment greedy/beam search

The reviewed production implementation is retained under:

- `src/growth/orchestrator_beam.py`
- `src/growth/make_campaign2_manifest.py`
- `jobs/campaign2_beam.slurm`

The SLURM launcher was hardened to resolve the project from
`SLURM_SUBMIT_DIR`, create writable log/tmp directories only after entering the
project, exclude `gorina8`, validate manifest fields, and resume only
non-terminal runs.

Regression tests were added in `tests/test_phase4_beam_logic.py` for the
trade-off gate, absolute-quality ranking, Chamfer diversity, bounded
graduation, parent retention, and checkpoint round trips.

## Phase 5 — adaptive soft-scaffold rigidity

Added production workflows:

- `src/growth/orchestrator_softmask_beam.py`
- `src/growth/make_softmask_manifest.py`
- `jobs/softmask_beam.slurm`

Added an additive DiffSBDD overlay:

- `patches/softmask_overlay/equivariant_diffusion/conditional_model_softmask.py`
- `patches/softmask_overlay/inpaint_softmask.py`
- `patches/softmask_overlay/softmask_atom_order.py`

The overlay implements hard/soft/free masks, stage-wise `rho`, fixed parent
atom features, order-safe Hungarian matching, parent-first renumbering, and
joint ligand/pocket recentering for `rho < 1`. The original hard-inpainting
entry point remains unchanged.

Contract tests were added in:

- `tests/test_softmask_core.py`
- `tests/test_softmask_order.py`

## Reproducibility and documentation

Updated:

- `setup_diffsbdd.sh`
- `reproduce/verify_diffsbdd_patch.py`
- `reproduce/verify_adaptive_search.py`
- `README.md`
- `REPRODUCE.md`
- `MANIFESTS.md`
- `THIRD_PARTY.md`
- `patches/README.md`

Added manuscript-ready Methods modules:

- `docs/methods/variable_increment_beam_search.md`
- `docs/methods/adaptive_soft_scaffold_methods.md`
- `docs/methods/adaptive_soft_scaffold_methods.tex`

Raw generated campaigns, model checkpoints, SLURM logs, and third-party source
checkouts remain excluded from the repository.
