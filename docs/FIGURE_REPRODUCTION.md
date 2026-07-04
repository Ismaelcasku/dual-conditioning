# Figure reproduction

## Mapping

| Manuscript item | Canonical script | Main inputs | Output basename |
|---|---|---|---|
| Figure 1 | `scripts/figures/figure1_protocol.py` | two tracked PyMOL PNGs | `figure1_protocol` |
| Figure 2 | `scripts/figures/figure2_generation_outcomes.py` | condition and across-seed TSVs | `figure2_generation_outcomes` |
| Figure 3 | `scripts/figures/figure3_fragmentation.py` | per-molecule audit and two tracked PyMOL PNGs | `figure3_fragmentation` |
| Figure 4 | `scripts/figures/figure4_relaxation_outcome_map.py` | selected minimization TSV | `figure4_relaxation_outcome_map` |
| Figure 5 | `scripts/figures/figure5_structural_examples.py` | four standalone PyMOL PNGs | `figure5_structural_examples` |
| Pair supplement | `scripts/figures/supp_pair_compatibility.py` | pair-reference TSV | `supp_pair_compatibility` |

## Quantitative and assembled figures

```bash
make figures
```

This command does not run the corrected campaign audit. It reproduces the
current manuscript figures from their frozen source tables.

## PyMOL renderings

Figure 1 and Figure 3 standalone renderings are tracked. Their PyMOL scripts are
also included under `reproducibility/pymol/`.

Figure 5 uses four selected structural examples. Generate fresh standalone
renderings from the tracked SDF/PDB inputs with:

```bash
make pymol-figure5-renderings
make figure5
```

The supplied Figure 5 export reflects the camera states used during manuscript
preparation. Fresh PyMOL output may differ slightly in camera framing, lighting,
and antialiasing, while preserving the molecular content.

## Re-audit consequence

Figures 2 and 3 must be regenerated if the corrected exhaustive interfragment
classification, unified anchor matching, or strict-dual definition changes the
source tables.
