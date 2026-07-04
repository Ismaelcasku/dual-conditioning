# External data layout

GitHub should contain code, small fixtures, configurations, frozen manifests, and figure source tables. Large files should be deposited in an external archive.

Expected project-relative layout for full reproduction:

```text
data/mpro/prepared/silvr_xchem_hits/<structure>/..._ligand.sdf
data/mpro/prepared/silvr_xchem_hits/<structure>/..._complex.pdb
data/mpro/manifests/*_fixed_atoms.json
artifacts/phase0_exp04_seed_replicates/.../*.sdf
artifacts/phase0_exp06_full_lambda_grid/.../*.sdf
artifacts/validation/fragmentation_audit/
artifacts/validation/restrained_minimization/
```

The Zenodo archive should include a manifest with SHA-256 hashes and preserve these relative paths so `make audit-campaign` can operate without path rewriting.
