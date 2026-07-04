Dual-conditioning canonical release
====================================

Release date: 2026-06-12

Canonical experiment:
- Three directed A-to-B pairs
- Five random seeds
- lambda_global = 0, 20, 50, 100, 200
- 75 experimental conditions
- Canonical shared x0434 lambda=0 baseline

Primary manifest:
manifests/experimental_design_release_75.tsv

Generated molecules:
generated/<pair>/seed_<seed>/lambda_<lambda>/

Official evaluation:
reports/

Frozen local anchors:
fixed_atoms/

Core scripts:
scripts/

Checkpoint:
checkpoints/crossdocked_fullatom_cond.ckpt

Important:
The files under this release should not be modified in place.
Subsequent chemical and pocket-validation results should be written to:
artifacts/validation/
