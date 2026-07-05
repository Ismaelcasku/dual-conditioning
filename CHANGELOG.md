# Changelog


## Unreleased

### Changed

- Updated Figure 3 generation to distinguish pair-specific evaluations from unique structures.
- Used pair-specific records for shape-dependent measurements and unique structures for connectivity and fragmentation.
- Added support for external pair-specific and unique-structure audit tables when regenerating Figure 3.
- Documented the final campaign totals of 691 pair-specific evaluations, 649 unique structures, and 555 sanitizable fragmented structures.
- Updated the interfragment classes to 5 potential missing-bond configurations, 299 bond-distance but valence-limited, 176 close but nonbonded, and 75 geometrically separated.

## 0.2.0 — GitHub-ready repository

- Added one canonical, repository-relative script per manuscript figure.
- Added final figure exports, figure source tables, PyMOL renderings, and selected structural examples.
- Added Make targets for analysis, tests, figures, and PyMOL rendering.
- Removed development-only figure versions and editable-install metadata.
- Added MIT licensing for original project code.
- Preserved the immutable Exp06 snapshot and legacy analysis path.
- Retained the corrected exhaustive interfragment audit and unified strict-dual definition.

## 0.1.0 — repository draft

- Preserved the canonical Exp06 snapshot and legacy analysis scripts.
- Added a portable `src/` package.
- Added validated Exp06 configuration.
- Added element-constrained Hungarian fixed-atom matching.
- Added exhaustive interfragment classification.
- Added explicit RDKit protrusion-ordering configurations.
- Added 27 unit and smoke tests.
