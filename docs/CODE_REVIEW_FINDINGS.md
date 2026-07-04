# Code review findings

## Verified

- The code-review archive passed its supplied SHA-256 verification.
- The canonical Exp06 copies of the guidance modules, evaluator, aggregation scripts, sampler modifications, and SLURM workflow matched the active development copies.
- The frozen snapshot in this repository is byte-identical to the supplied `canonical_snapshot/` directory.
- The cleaned package installs as an editable Python package.
- The current clean suite contains 27 passing tests.

## Refactored

- Guidance imports now use the package namespace `dual_conditioning`.
- Fixed-atom matching is centralized and uses element-constrained Hungarian assignment.
- Shape Tanimoto distance and similarity have separate field names.
- Strict-dual success is defined once and requires connectivity, local retention, and both global-shape comparisons.
- Full-record, largest-component, and anchor-component metrics have unambiguous names.
- Interfragment classes use an exhaustive search over all pairs across distinct heavy-atom components.
- The minimum absolute-distance pair and minimum covalent-radius-ratio pair are reported separately.
- Exp06 generation settings are encoded in a validated YAML configuration rather than inferred from environment variables.

## Requires campaign data

The review package did not contain the generated SDF campaign. Consequently, this draft cannot yet determine whether the corrected audit changes:

- the `3 / 306 / 177 / 76` fragmentation-class counts;
- full, parent, anchor, or connected strict-dual counts;
- component-aware shape correlations;
- Figures 2 and 3 source tables.

These should be recalculated from the immutable generated archive, not by regenerating the 75-condition campaign.

## Reporting decision still required

The frozen evaluator relied on `ShapeProtrudeDist(..., allowReordering=True)`, while a later minimization recalculation explicitly used `False`. The manuscript describes a directional protrusion metric. The release must state which configuration supplies each reported result and avoid combining the two silently.
