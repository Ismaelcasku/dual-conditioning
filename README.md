# Directed staged growth under shape-guided 3D diffusion

Code and derived data for the manuscript project studying whether a frozen,
pocket-conditioned 3D diffusion model can preserve a local warhead from ligand
A while growing toward the three-dimensional shape of ligand B.

The repository contains the complete analysis lineage from single-shot guidance
through fixed staged growth and two adaptive search extensions:

1. **Variable-increment search:** greedy (`k=1`) and beam (`k=3`) search over
   requested atom increments `+1` to `+5`.
2. **Adaptive soft-scaffold search:** fixed `+4` growth with stage-wise scaffold
   rigidity `rho in {1, 0.75, 0.5, 0.25, 0}`.

The final soft-scaffold implementation keeps the original seven-heavy-atom
warhead hard, allows previously generated parent coordinates to move according
to `rho`, preserves their atom identities, restores the DiffSBDD zero-mean
coordinate invariant, and propagates parent atoms with order-safe matching.

## Repository layout

```text
environment.yml / environment-cpu.yml   GPU and CPU environments
THIRD_PARTY.md                          model, checkpoint, and input-data provenance
config/                                 portable path templates and configs
src/guidance/                           differentiable inference-time guidance
src/generation/                         DiffSBDD wrappers and preflight checks
src/growth/                             fixed and adaptive staged-growth workflows
src/preparation/                        Mpro pair and anchor preparation
src/analysis/                           manuscript analysis pipelines
src/validation/                         restrained-relaxation code
jobs/                                   active SLURM launchers
patches/                                DiffSBDD patches and soft-mask overlay
tests/                                  Phase-4 and Phase-5 contract tests
docs/methods/                           manuscript-ready Methods modules
figures/                                manuscript figure scripts
data/derived/                           data required for CPU-only reproduction
reproduce/                              numerical and integration verification
archive/legacy/                         superseded development scripts
```

## CPU-only reproduction

```bash
conda env create -f environment-cpu.yml
conda activate dual-conditioning-cpu
python reproduce/verify_numbers.py --data-root data/derived
python reproduce/verify_figures.py
python tests/test_phase4_beam_logic.py
```

This reproduces the distributed fixed-increment statistics and manuscript
figures and verifies the pure variable-increment search logic.

## DiffSBDD setup

```bash
cp config/paths.example.env config/paths.env
# Edit config/paths.env
./setup_diffsbdd.sh
python reproduce/verify_diffsbdd_patch.py external/DiffSBDD
```

The setup script clones DiffSBDD at the pinned commit, applies the reviewed
shape-guidance patches, and installs the additive centered soft-mask sampler.
The original `external/DiffSBDD/inpaint.py` remains intact for the earlier
campaigns; Phase 5 uses `external/DiffSBDD/inpaint_softmask.py`.

After setup, the no-checkpoint soft-mask contract tests are:

```bash
python tests/test_softmask_core.py
python tests/test_softmask_order.py
```

## Active adaptive campaigns

Variable increment:

```bash
python src/growth/make_campaign2_manifest.py \
  --out artifacts/phase4_evo_beam/beam_manifest.tsv
sbatch jobs/campaign2_beam.slurm
```

Adaptive soft scaffold:

```bash
python src/growth/make_softmask_manifest.py \
  --out artifacts/phase5_softmask/softmask_manifest.tsv
sbatch jobs/softmask_beam.slurm
```

Both campaigns contain 40 trajectories: two ligand pairs, ten seeds, and greedy
versus beam search.

## Release status

**Available:** derived-data reproduction, figure generation, active campaign
code, manifests, SLURM launchers, variable-increment tests, centered soft-mask
overlay, order-recovery tests, and Methods text for both adaptive phases.

**Still required for complete Level-B regeneration:** the exact checkpoint URL
and hash, definitive Mpro input-data provenance, and finalized external
environment instructions for restrained physical relaxation.

See `REPRODUCE.md`, `MANIFESTS.md`, `THIRD_PARTY.md`, and `docs/methods/`.
