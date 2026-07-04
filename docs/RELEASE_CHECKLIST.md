# Release checklist

## Required before the manuscript-linked v1.0 release

- [ ] Run the corrected audit against all frozen generated SDF records.
- [ ] Compare revised fragmentation classes with the current `3 / 306 / 177 / 76` counts.
- [ ] Recalculate full, parent, anchor, and connected strict-dual counts with one definition.
- [ ] Decide whether the final paper reports frozen protrusion (`allowReordering=true`) or fixed-direction protrusion (`false`).
- [ ] Update manuscript numbers, source tables, and Figures 2–3 if the re-audit changes them.
- [ ] Record the exact upstream DiffSBDD repository commit used for the vendor baseline.
- [ ] Generate an exact Linux environment lock on the cluster.
- [ ] Add checkpoint and external-data SHA-256 hashes.
- [ ] Replace the placeholder GitHub URL in `CITATION.cff`.
- [ ] Add the archival data/software DOI.
- [ ] Test installation, unit tests, and figure generation in a clean environment.
- [ ] Create a tagged GitHub release and archive it in Zenodo.

## Already completed in this repository

- [x] Canonical figure scripts and source tables incorporated.
- [x] Development-only figure versions removed.
- [x] Repository-relative paths used in public scripts.
- [x] Exp06 frozen snapshot retained with hashes.
- [x] Corrected portable evaluation package and tests included.
- [x] Original project code licensed under MIT.
