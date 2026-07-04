# Reproducibility snapshots

`exp06_frozen/` is an immutable copy of the canonical 12 June 2026 release material included in the code-review package. `legacy_analysis/` and `legacy_validation/` preserve scripts required to reproduce the manuscript's current values before the corrected re-audit.

New analyses must not overwrite these files. Write revised outputs to `results/source_data/` with an explicit version suffix.
