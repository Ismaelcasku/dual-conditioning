#!/usr/bin/env python
"""
B-volume protrusion guide (option 2a) for dual-conditioning Phase 0.

Pushes non-fixed ligand nodes that protrude OUTSIDE the volume occupied by
reference molecule B back toward B, while leaving nodes already inside B
untouched. Count-invariant: each node is evaluated against the whole B cloud
by nearest-neighbor; no node<->B matching.

All coordinates here are expected in the model's INTERNAL frame
(i.e. already divided by norm_values[0] and centered on mean_known).
This module does not normalize or center; the caller does that once, outside
the loop, so units are explicit and auditable.

This is NOT a gaussian shape-overlap term. It minimizes protrusion against a
union of spheres of radius r centered on B's heavy atoms. Deliberately the
cheap, falsifiable first prototype.
"""

import torch


def compute_b_volume_protrusion_guide(
    x_unknown_internal,   # [N_unknown, 3] non-fixed node coords, internal frame
    b_coords_internal,    # [M, 3] B heavy-atom coords, internal frame
    r_internal,           # effective B atom radius, internal units
    clip_internal,        # max guide magnitude per call, internal units
):
    """
    Returns guide [N_unknown, 3], internal units.

    For each non-fixed node p:
        d, b_nearest = nearest B atom and its distance
        if d <= r:  guide = 0
        if d  > r:  guide = unit(b_nearest - p) * min(d - r, clip)
    """
    if x_unknown_internal.numel() == 0 or b_coords_internal.numel() == 0:
        return torch.zeros_like(x_unknown_internal)

    # Pairwise distances: [N_unknown, M]
    dmat = torch.cdist(x_unknown_internal, b_coords_internal)

    d_min, idx_nearest = dmat.min(dim=1)          # [N_unknown]
    b_nearest = b_coords_internal[idx_nearest]    # [N_unknown, 3]

    direction = b_nearest - x_unknown_internal    # toward nearest B atom
    # Safe unit vector: avoid divide-by-zero when a node sits exactly on B.
    norm = d_min.clamp_min(1e-8).unsqueeze(1)
    unit = direction / norm

    protrusion = (d_min - r_internal).clamp_min(0.0)        # 0 if inside B
    magnitude = torch.minimum(
        protrusion, torch.full_like(protrusion, clip_internal)
    )

    guide = unit * magnitude.unsqueeze(1)
    # Nodes inside B (protrusion == 0) already get zero magnitude -> zero guide.
    return guide


def clipped_fraction(x_unknown_internal, b_coords_internal, r_internal, clip_internal):
    """Diagnostic: fraction of non-fixed nodes whose protrusion exceeds clip
    (i.e. nodes being clipped). High early-step values mean clip is acting as a
    blade, not a safety cap -> raise clip."""
    if x_unknown_internal.numel() == 0 or b_coords_internal.numel() == 0:
        return 0.0
    dmat = torch.cdist(x_unknown_internal, b_coords_internal)
    d_min, _ = dmat.min(dim=1)
    protrusion = (d_min - r_internal).clamp_min(0.0)
    n_clipped = (protrusion > clip_internal).sum().item()
    return n_clipped / max(1, protrusion.numel())
