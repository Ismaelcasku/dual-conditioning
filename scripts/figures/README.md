# Figure generation

Each manuscript output has one canonical Python script:

| Output | Script | Tracked inputs |
|---|---|---|
| Figure 1 | `figure1_protocol.py` | `results/source_renderings/figure1_*.png` |
| Figure 2 | `figure2_generation_outcomes.py` | campaign summary TSV files |
| Figure 3 | `figure3_fragmentation.py` | per-molecule audit TSV and PyMOL renderings |
| Figure 4 | `figure4_relaxation_outcome_map.py` | `candidate_shape_anchor.tsv` |
| Figure 5 | `figure5_structural_examples.py` | standalone PyMOL relaxation renderings |
| Pair supplement | `supp_pair_compatibility.py` | `pair_reference_shape.tsv` |

Run Figures 1–4 and the supplement with:

```bash
make figures
```

Figure 5 requires the standalone structural renderings first:

```bash
make pymol-figure5-renderings
make figure5
```

The current quantitative source tables reproduce the frozen manuscript analysis.
They must be replaced or versioned after the corrected campaign re-audit.
