# Third-party dependencies not included in this repository

## DiffSBDD generative model

This project uses DiffSBDD as the pretrained pocket-conditioned 3D diffusion
model. DiffSBDD is open source but is not redistributed here. It is obtained
from the official repository at a fixed commit.

- Repository: https://github.com/arneschneuing/DiffSBDD.git
- Pinned commit: `5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`
- License: see the upstream repository.

`setup_diffsbdd.sh` performs three operations:

1. clones the pinned upstream commit;
2. applies the two reviewed unified patches under `patches/`;
3. copies the additive centered soft-mask overlay from
   `patches/softmask_overlay/`.

The overlay adds new files and leaves the original hard-inpainting entry point
available, allowing both the earlier campaigns and the adaptive-rigidity
campaign to be reproduced from the same checkout. File hashes are verified by
both `setup_diffsbdd.sh` and `reproduce/verify_diffsbdd_patch.py`.

## DiffSBDD pretrained checkpoint

The pinned commit fixes the code, not the model weights. The checkpoint must be
downloaded separately.

- Expected project location:
  `artifacts/checkpoints/crossdocked_fullatom_cond.ckpt`
- Source: `<TODO: add the official checkpoint URL>`
- SHA256: `<TODO: record the exact checkpoint hash>`

Without a pinned checkpoint hash, the generative campaigns are not fully
reproducible at Level B.

## Mpro fragment hits

The campaigns use crystallographic Mpro fragment hits prepared under
`data/mpro/prepared/silvr_xchem_hits/`.

- Source: `<TODO: add the XChem/Fragalysis or archived dataset URL>`
- Redistribution: `<TODO: state whether the prepared structures can be bundled>`

The active A/B pairs are x0434-to-x2193 and x0874-to-x1093.

## SILVR and restrained-relaxation dependencies

SILVR utilities in historical working directories are not required by the
shape-guided growth pipeline. Restrained-relaxation workflows additionally
require the OpenMM/OpenFF/PoseCheck environment described by the validation
scripts and are not installed by `setup_diffsbdd.sh`.
