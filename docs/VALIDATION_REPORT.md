# Repository validation report

Validation performed on 4 July 2026 after assembling the public repository.

## Passed

- 27 Python tests passed.
- `configs/exp06_generation.yaml` passed schema validation.
- All hashes in `reproducibility/exp06_frozen/SHA256SUMS.txt` verified.
- All public Python files compiled without syntax errors.
- Canonical scripts for Figures 1, 2, 3, 4, and the pair-compatibility
  supplement executed successfully from repository-relative paths.
- No personal absolute paths remained outside immutable legacy/frozen material.

## Preserved rather than regenerated

The manuscript figure exports included in `results/figures/` are the original
local exports. Test regeneration used a different font/rendering environment and
therefore did not replace the archived manuscript images.

Figure 5 was not freshly rendered in the validation environment because PyMOL
was not available and the original standalone four-panel PNG inputs were not
part of the staging export. The final Figure 5, all selected SDF/PDB structures,
and a repository-relative PyMOL rendering script are included.

## Scientific release blocker

The corrected campaign-wide audit cannot be executed without the complete
frozen generated SDF archive. The manuscript-linked v1.0 release should follow
the decisions in `docs/RELEASE_CHECKLIST.md`.
