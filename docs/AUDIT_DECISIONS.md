# Audit decisions

## Frozen versus corrected analysis

The manuscript's current fragmentation counts were produced by the legacy script in `reproducibility/legacy_analysis/audit_exp06_fragmentation_and_shape.py`. That implementation selects the single atom pair with the minimum distance divided by covalent-radius sum and assigns the record-level class from that pair.

The corrected implementation in `src/dual_conditioning/evaluation/connectivity.py` examines **all** heavy-atom pairs belonging to different components. A record is a potential missing bond when at least one bond-distance pair has valence headroom on both atoms. This avoids classifying a record as valence-limited solely because its closest normalized pair is saturated.

Both implementations are retained. The corrected counts must be calculated from the frozen generated SDFs before changing the manuscript.

## Fixed-atom assignment

Every analysis now uses an element-constrained Hungarian assignment and returns original RDKit atom indices. The legacy fragmentation audit used a greedy assignment, while the official evaluator used Hungarian assignment.

## Strict-dual criterion

`connected_strict_dual` has one authoritative definition:

```text
one heavy-atom component
AND all seven fixed atoms matched within 0.2 Å with element identity
AND B closer than A by RDKit Shape Tanimoto distance
AND B closer than A by RDKit Shape Protrude distance
```

Component-only geometric comparisons are reported with names that do not imply full dual-conditioning success.

## RDKit protrusion ordering

The frozen evaluator relied on RDKit's default `allowReordering=True`. Some later physical-relaxation scripts explicitly used `False`. The manuscript calls Shape Protrude directional, so the corrected re-audit uses `False`. Both choices are encoded explicitly in separate YAML files.
