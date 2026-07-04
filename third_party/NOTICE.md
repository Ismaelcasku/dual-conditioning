# Third-party software

The frozen Exp06 sampler modifies selected files from DiffSBDD:

- upstream repository: `https://github.com/arneschneuing/DiffSBDD`
- upstream license: MIT
- associated publication DOI: `10.1038/s43588-024-00737-x`

The supplied DiffSBDD license is preserved under `third_party/diffsbdd/LICENSE`.
Modified files are stored under `reproducibility/exp06_frozen/scripts/vendor/`
for experiment reproduction and must not be presented as original project code.

The exact upstream commit used in the HPC workspace was not recorded in the
exported material and must be recovered or documented as unavailable before the
manuscript-linked v1.0 release.

SILVR and OpenEye docking utilities from the development workspace are excluded
because they are not required by the final Exp06 generation or analysis path.
