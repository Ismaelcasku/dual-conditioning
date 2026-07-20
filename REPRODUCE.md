# Reproducing the manuscript project

## Level A — figures and reported statistics from derived data

```bash
conda env create -f environment-cpu.yml
conda activate dual-conditioning-cpu
python reproduce/verify_numbers.py --data-root data/derived
python reproduce/verify_figures.py
python tests/test_phase4_beam_logic.py
```

`verify_numbers.py` recomputes the distributed fixed-increment values from
`data/derived/exp2`. `verify_figures.py` regenerates Figures 1–5 as PNG, PDF,
and SVG files.

| figure | script | data |
|---|---|---|
| 1 task/protocol/connectivity | `figures/make_figure1_task_protocol_connectivity.py` | `data/derived/exp0` |
| 2 single-step expansion | `figures/make_figure2_single_step_expansion.py` | `data/derived/exp1` |
| 3 fixed-increment summary | `figures/make_figure3_fixed_increment_summary.py` | `data/derived/exp2` |
| 4 trajectory distributions | `figures/make_figure4_violin_distributions.py` | `data/derived/exp2` |
| 5 A–B distance plane | `figures/make_figure5_AB_plane.py` | `data/derived/exp2` |

Final Phase-4 and Phase-5 summary tables can be added under `data/derived/`
after their statistical analyses are frozen. Raw generated SDF campaigns are
not distributed.

## Level B — regeneration from DiffSBDD

### 1. Configure paths

```bash
cp config/paths.example.env config/paths.env
# Edit SANDBOX and any local checkpoint/data paths.
```

### 2. Install the pinned DiffSBDD integration

```bash
./setup_diffsbdd.sh
python reproduce/verify_diffsbdd_patch.py external/DiffSBDD
```

The setup script clones DiffSBDD at commit
`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`, applies the shape-guidance
patches, and installs the additive centered soft-mask overlay. It verifies exact
hashes for all modified and added files.

The original hard-inpainting entry point remains:

```text
external/DiffSBDD/inpaint.py
```

The adaptive-rigidity entry point is:

```text
external/DiffSBDD/inpaint_softmask.py
```

### 3. Run integration tests

Inside an environment containing the DiffSBDD dependencies:

```bash
python tests/test_softmask_core.py
python tests/test_softmask_order.py
```

These tests do not load the checkpoint and do not require a GPU. They verify:

- hard/soft/free mask semantics;
- exact `rho=1` equivalence to hard inpainting;
- release of soft coordinates while preserving atom features;
- centered ligand states with unchanged ligand–pocket geometry;
- atom-type-constrained parent matching;
- parent-first point-cloud and RDKit renumbering.

### 4. Generate campaign manifests

Fixed-increment campaign:

```bash
python src/growth/make_campaign1_manifest.py ...
sbatch jobs/campaign1_fixed_increment.slurm
```

Variable-increment greedy/beam search:

```bash
python src/growth/make_campaign2_manifest.py \
  --out artifacts/phase4_evo_beam/beam_manifest.tsv
sbatch jobs/campaign2_beam.slurm
```

Adaptive soft-scaffold search:

```bash
python src/growth/make_softmask_manifest.py \
  --out artifacts/phase5_softmask/softmask_manifest.tsv
sbatch jobs/softmask_beam.slurm
```

A smoke manifest can be produced with `--pilot 1`; the complete adaptive
manifest must be generated without `--pilot` and contains rows 0–39.

### 5. Remaining external requirements

Complete generative reproduction still requires:

1. the pretrained checkpoint URL and SHA256 recorded in `THIRD_PARTY.md`;
2. Mpro input-data download and preparation instructions;
3. a working CUDA/Singularity environment in `config/paths.env`;
4. the finalized OpenMM/OpenFF/PoseCheck environment for the restrained
   relaxation campaign.

## Methods provenance

Manuscript-ready descriptions of the two adaptive phases are in:

```text
docs/methods/variable_increment_beam_search.md
docs/methods/adaptive_soft_scaffold_methods.md
docs/methods/adaptive_soft_scaffold_methods.tex
```
