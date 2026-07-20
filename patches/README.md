# DiffSBDD integration patches and overlays

`setup_diffsbdd.sh` clones the official DiffSBDD repository at commit
`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`, applies the unified patches in
lexical order, and then installs the additive soft-mask overlay.

## Unified patches

- `0001-add-inference-time-shape-guidance.patch`
  - adds the reference-B shape input to `inpaint.py`;
  - wires inference-time guidance into the inpainting sampler;
  - preserves the historical x0-space Gaussian-shape and z-space protrusion
    code paths used during development;
  - adds the optional raw fixed-block diagnostic.
- `0002-add-inference-compatibility-fallbacks.patch`
  - makes Weights & Biases optional for inference;
  - provides compatibility with Biopython versions lacking `three_to_one`;
  - allows checkpoint inference when optional visualization imports are absent.

## Additive soft-mask overlay

`softmask_overlay/` contains three new files that do not replace the original
DiffSBDD hard-inpainting entry point:

- `equivariant_diffusion/conditional_model_softmask.py`
  - installs a runtime inpainting method with hard, soft, and free atom masks;
  - keeps the seven-atom warhead hard;
  - interpolates soft-parent coordinates with stage action `rho`;
  - preserves parent atom-feature channels;
  - restores the zero-ligand-mean invariant for `rho < 1` by translating ligand
    and pocket together.
- `inpaint_softmask.py`
  - exposes the centered soft-mask sampler as a separate command-line entry
    point;
  - preserves requested PDB atom order;
  - canonicalizes generated point clouds and RDKit molecules before writing the
    scaffold propagated to the next stage.
- `softmask_atom_order.py`
  - performs atom-type-constrained Hungarian assignment and parent-first
    renumbering.

The overlay is copied into the pinned DiffSBDD checkout by
`setup_diffsbdd.sh`. The original `inpaint.py` remains available for all earlier
campaigns.

## Verified hashes

Applying both unified patches and installing the overlay produces:

```text
d8fcced3e17352357f18a1a4cd737f1b5f2caa48117d2d3ba66e1bbb5e20e441  equivariant_diffusion/conditional_model.py
659c328fde62ad59fe913938e66f40179857a17267c8d9158c8da6f1d3f305d6  inpaint.py
ae521cc7fae48c27582719b8881f83a094f815308236b0c38cd9e46bdb47c0b9  lightning_modules.py
e8cd642f4fb1314540781c7eca960a22afcf393d634c3b0744dfbfcce99f5059  equivariant_diffusion/conditional_model_softmask.py
61b5a4731395e9a724ad58a2f9b390d0c03d88129e501ea38cd175a6407cfc98  inpaint_softmask.py
f9a5aac8d6b25038c95a59be872a93e0d5b08453eb32fe22dc6d43c8fc34a69f  softmask_atom_order.py
```

Commands that run DiffSBDD must include the repository `src/` directory, the
`src/growth/` directory, and the DiffSBDD checkout in `PYTHONPATH`.
