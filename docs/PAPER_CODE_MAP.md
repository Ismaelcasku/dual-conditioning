# Paper-to-code map

| Manuscript output | Primary implementation | Required input |
|---|---|---|
| Table 1: pair compatibility | `src/dual_conditioning/evaluation/shape.py`; frozen pair-selection script | reference A/B SDFs |
| Figure 1 | `scripts/figures/figure1_protocol.py` | tracked PyMOL renderings |
| Figure 2A–C | `scripts/analysis/audit_campaign.py`; clean evaluation modules | archived 75-condition SDF campaign |
| Figure 2D | `dual_conditioning.evaluation.strict_dual.evaluate_strict_dual` | per-record local, shape, and connectivity metrics |
| Figure 3A–B | component extraction and tracked structural renderings | selected fragmented SDF record |
| Figure 3C–E | exhaustive interfragment audit | archived generated SDF campaign |
| Figure 4 | `scripts/figures/figure4_relaxation_outcome_map.py` | selected minimization summary |
| Figure 5 | `scripts/figures/figure5_structural_examples.py`; PyMOL scripts | selected pre/post-minimization structures |
