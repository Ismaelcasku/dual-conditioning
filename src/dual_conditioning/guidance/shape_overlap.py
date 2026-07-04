"""
b_shape_overlap.py — Experiment 03 global guide field: differentiable Gaussian
shape-overlap (Shape Tanimoto) toward B.

Motivation
----------
Experiment 02 used a one-sided B-volume *protrusion* field (pull each free atom
toward its nearest B atom). That objective is misaligned with volumetric shape
similarity: it has no anti-clustering term, so it bunches atoms up and the grid
Shape Tanimoto to B got WORSE (0.76 vs baseline 0.52).

This field instead guides on the gradient of the Gaussian Shape Tanimoto
(Grant & Pickup; ROCS-style; SQUID Eq.1):

    T = V_GB / (V_GG + V_BB - V_GB)
    V_XY = sum_{x in X, y in Y} exp(-0.5 * alpha * ||x - y||^2)

The denominator's V_GG term (self-overlap of the generated molecule) enters with
a negative sign in dT/dx, producing an *anti-clustering* force: maximizing T both
fills B's volume (V_GB up) AND keeps generated atoms from collapsing (V_GG down).
This is the two-sided objective the protrusion lacked, and it is the smooth
surrogate of the frozen evaluator's grid Shape Tanimoto.

Design choices (match the frozen evaluator so guide and judge agree)
--------------------------------------------------------------------
- IN-FRAME, NO ALIGNMENT. The evaluator calls rdShapeHelpers.ShapeTanimotoDist
  on conformers as-is; A/B/generated share the pocket frame. We do the same.
- HEAVY ATOMS ONLY. RDKit ShapeTanimotoDist uses ignoreHs=True by default; the
  caller passes heavy-atom coordinates.
- alpha (Gaussian width) is a parameter. alpha=0.81 ~ ROCS (SQUID). Calibrate to
  RDKit's vdW grid (vdwScale~0.8) if needed; it is the surrogate, not the judge.
- WHOLE-MOLECULE shape. Fixed (inpainted) atoms are part of the molecule's
  volume, so they enter V_GG and V_GB; pass them via `fixed_coords`. The gradient
  is taken w.r.t. FREE atoms only (fixed atoms are reimposed by the hard mask).
- COM-neutrality and the lambda*step scaling are applied by the caller (the
  sampler), exactly as for the Exp02 field. This module returns the (clipped)
  ascent direction +dT/dx for the free atoms.

Coordinate space: internal sampler units. With norm_values[0]=1 this is Angstrom,
so `clip` is in Angstrom.
"""

from __future__ import annotations

import torch

DEFAULT_ALPHA = 0.81   # ROCS/SQUID Gaussian width; surrogate for RDKit grid metric
DEFAULT_CLIP = 1.0     # max per-atom guide norm (Angstrom), bounds the step
_EPS = 1e-8


def _pairwise_overlap(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Sum of 2-body Gaussian volume overlaps between point sets X and Y.

    Squared distances are computed directly (no sqrt) to stay finite and
    differentiable at coincident points (self-overlap d=0 -> exp(0)=1).
    """
    # X: [n,3], Y: [m,3] -> d2: [n,m]
    d2 = (X[:, None, :] - Y[None, :, :]).pow(2).sum(dim=-1)
    return torch.exp(-0.5 * alpha * d2).sum()


def shape_tanimoto_overlap(
    free_coords: torch.Tensor,
    b_coords: torch.Tensor,
    fixed_coords: torch.Tensor | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Differentiable Gaussian Shape Tanimoto between the generated molecule
    (free + optional fixed atoms) and B. Returns a scalar tensor in [0, 1].
    """
    if fixed_coords is not None and fixed_coords.numel() > 0:
        G = torch.cat([free_coords, fixed_coords], dim=0)
    else:
        G = free_coords

    v_gg = _pairwise_overlap(G, G, alpha)
    v_bb = _pairwise_overlap(b_coords, b_coords, alpha)
    v_gb = _pairwise_overlap(G, b_coords, alpha)
    denom = v_gg + v_bb - v_gb
    return v_gb / (denom + _EPS)


def compute_shape_overlap_guide(
    free_coords: torch.Tensor,
    b_coords: torch.Tensor,
    fixed_coords: torch.Tensor | None = None,
    alpha: float = DEFAULT_ALPHA,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """Ascent direction +d(ShapeTanimoto)/d(free_coords), clipped per-atom.

    Args:
        free_coords:  [n_free, 3] predicted clean coords of NON-fixed heavy atoms.
        b_coords:     [m, 3] heavy-atom coords of B in the SAME (per-sample) frame.
        fixed_coords: [n_fixed, 3] heavy-atom coords of fixed atoms, or None.
                      Included in the molecule volume but NOT moved.
        alpha:        Gaussian width.
        clip:         max per-atom guide norm (Angstrom); direction preserved.

    Returns:
        [n_free, 3] guide vectors. Empty input -> empty tensor. The caller applies
        COM-neutralization and the lambda*step scaling (as in Exp02).
    """
    if free_coords.numel() == 0 or b_coords.numel() == 0:
        return torch.zeros_like(free_coords)

    # Isolate from the model graph: leaf with grad, B/fixed are constants.
    x = free_coords.detach().clone().requires_grad_(True)
    fixed = None if fixed_coords is None else fixed_coords.detach()
    b = b_coords.detach()

    with torch.enable_grad():
        t = shape_tanimoto_overlap(x, b, fixed, alpha)
        (grad,) = torch.autograd.grad(t, x, create_graph=False, retain_graph=False)

    grad = grad.detach()
    if not torch.isfinite(grad).all():
        return torch.zeros_like(free_coords)

    # Ascent on T. Clip per-atom NORM (preserve direction).
    norms = grad.norm(dim=1, keepdim=True)
    scale = torch.clamp(clip / (norms + _EPS), max=1.0)
    return grad * scale
