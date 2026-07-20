# Tests

`test_phase4_beam_logic.py` exercises the pure search logic and can run in the
project CPU environment.

The soft-mask tests require the DiffSBDD checkout created by
`./setup_diffsbdd.sh` and an environment containing PyTorch, torch-scatter,
RDKit, SciPy, and the DiffSBDD dependencies:

```bash
python tests/test_softmask_core.py
python tests/test_softmask_order.py
```

These tests do not load the pretrained checkpoint and do not require a GPU.
