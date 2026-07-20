# Validation report

Validation performed during repository assembly:

- active Python syntax compilation: 60 files;
- shell and active SLURM syntax: passed;
- Phase-4 logic tests: 6/6 passed;
- soft-mask core contract tests: 7/7 passed;
- soft-mask order-recovery tests: 4/4 passed;
- soft-mask overlay hashes synchronized across `setup_diffsbdd.sh`,
  `reproduce/verify_diffsbdd_patch.py`, and `patches/README.md`.

The soft-mask unit tests were executed against the distributed overlay using a
minimal local dependency shim for `torch_scatter` and the DiffSBDD class import.
They validate pure mask, recentering, runtime-installation, and atom-order logic;
they are not a replacement for a checkpoint-backed GPU integration run.

The checkpoint-backed centered sampler and full search workflow were previously
validated by the project smoke campaign. The repository package itself does not
redistribute that checkpoint, its generated SDF outputs, or the third-party
DiffSBDD checkout.
