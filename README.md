# Hard local constraints dominate global shape guidance in 3D molecular diffusion

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21206298.svg)](https://doi.org/10.5281/zenodo.21206298)

Code, frozen experiment metadata, connectivity-aware re-audit tools, and figure
source material for the study:

> **Hard local constraints dominate global shape guidance and induce fragmentation in pretrained 3D molecular diffusion**

The project tests whether a frozen pocket-conditioned 3D diffusion model can
preserve a seven-heavy-atom anchor from ligand A while steering the remaining
atoms toward the global shape of a distinct ligand B.

## Repository structure

```text
configs/                         Frozen and corrected evaluation settings
src/dual_conditioning/          Portable guidance and evaluation package
scripts/analysis/               Corrected per-record and campaign audits
scripts/figures/                One canonical script per manuscript figure
reproducibility/exp06_frozen/   Immutable code and metadata used for Exp06
reproducibility/legacy_*        Original analysis and relaxation scripts
reproducibility/pymol/          Structure-rendering scripts
results/source_data/            Current manuscript figure source tables
results/source_renderings/      PyMOL renderings used by Figures 1 and 3
data/structural_examples/       Selected pre/post-minimization structures
results/figures/                Current manuscript figure exports
third_party/                    Third-party notices and licenses
```

The exact Exp06 implementation is preserved separately from the corrected,
maintainable package. Historical files are not silently rewritten.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate dual-conditioning
make test
```

Using an existing Python environment:

```bash
python -m pip install -e '.[guidance,figures,test]'
python -m pytest
```

## Frozen campaign

The 75-condition campaign used:

- directed pairs: `x0434→x1093`, `x0874→x1093`, `x0434→x2193`;
- seeds: `1101, 2202, 3303, 4404, 5505`;
- guidance strengths: `0, 20, 50, 100, 200`;
- ten requested samples per condition;
- 50 diffusion steps and five resampling iterations;
- `x0` Gaussian shape guidance with `alpha=0.3`;
- a per-atom guidance clip of `1.0 Å`;
- seven fixed heavy atoms.

The readable configuration is in `configs/exp06_generation.yaml`. The exact
scripts, manifests, anchor definitions, and SLURM jobs are in
`reproducibility/exp06_frozen/`.

Verify the frozen snapshot with:

```bash
make frozen-hashes
```

## Tests

```bash
make test
```

The tests cover guidance behavior, element-constrained Hungarian anchor
matching, component analysis, exhaustive interfragment classification, shape
metric naming, configuration validation, and the strict-dual definition.

## Corrected campaign re-audit

The original fragmentation classifier assigned each record from one closest
normalized-distance atom pair. The corrected audit examines every heavy-atom
pair across distinct components. It also uses one authoritative strict-dual
definition and one anchor-matching implementation.

After mounting the archived campaign at the expected relative layout:

```bash
make audit-campaign PROJECT_ROOT=/path/to/dual_conditioning_archive
```

The corrected output is written to:

```text
results/source_data/fragment_audit_per_molecule_v2.tsv
```

The current manuscript tables and rendered quantitative figures remain the
frozen pre-re-audit versions until this command is run on the complete SDF
archive. See `docs/AUDIT_DECISIONS.md`.

## Figure generation

Figures 1–4 and the pair-compatibility supplement can be regenerated from the
tracked source tables and renderings:

```bash
make figures
```

Figure 5 additionally requires standalone PyMOL renderings:

```bash
make pymol-figure5-renderings
make figure5
```

The archived final Figure 5 and all selected SDF/PDB inputs are included. Exact
PyMOL pixels may vary with version, renderer, and camera state. See
`docs/FIGURE_REPRODUCTION.md`.

## External data

Large generated SDF collections, the pretrained checkpoint, and complete
OpenMM outputs are intentionally excluded from GitHub. They should be deposited
in the archival data record while preserving the paths described in
`docs/DATA_LAYOUT.md`.

## Reproducibility scope

This repository supports two levels:

1. **Frozen reproduction:** inspect or rerun the exact Exp06 scripts and legacy
   analysis against the archived data.
2. **Corrected re-analysis:** rerun connectivity, component-aware shape,
   anchor matching, and strict-dual evaluation with the cleaned package.

Regenerating the full 75-condition GPU campaign is not required to reproduce
the paper's analysis once the generated SDF archive is available.

## License and third-party code

Original code in this repository is released under the MIT License. Modified
DiffSBDD files are preserved only for reproducibility and remain subject to the
upstream MIT license. See `third_party/NOTICE.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. Replace the placeholder
repository URL and add the final software DOI before the tagged public release.

<!-- campaign-data-start -->
## Campaign data and Figure 3

**Associated dataset:** [10.5281/zenodo.21204994](https://doi.org/10.5281/zenodo.21204994)

Associated dataset: [10.5281/zenodo.21204994](https://doi.org/10.5281/zenodo.21204994)

The generation campaign comprises 691 pair-specific evaluations from 649
unique generated structures. Pair-specific records are used for shape
measurements because these depend on the selected global reference ligand B.
Connectivity and fragmentation are calculated from unique structures, so
shared unguided baselines are counted only once.

Of the 648 unique structures that could be chemically sanitized, 555 were
fragmented. Their interfragment geometries were classified as:

- 5 potential missing-bond configurations;
- 299 bond-distance but valence-limited;
- 176 close but nonbonded;
- 75 geometrically separated.

One additional fragmented structure failed kekulization and is reported
separately. No guided structure satisfied the connected strict-dual criterion
under either Shape Protrude convention.

Figure 3 requires two tables from the associated dataset:

- a pair-specific table for shape-dependent measurements;
- a unique-structure table for connectivity and interfragment classification.

The figure can be regenerated with:

```bash
export DC_FIG3_PAIR_EVAL_TSV=/path/to/per_record_directional.tsv
export DC_FIG3_UNIQUE_TSV=/path/to/figure3_unique_structures.tsv

python scripts/figures/figure3_fragmentation.py
```

The complete generated structures, audit tables, and final figure files are
distributed with the associated dataset rather than duplicated in this
software repository.
<!-- campaign-data-end -->
