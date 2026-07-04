#!/usr/bin/env python
import torch
from dual_conditioning.guidance.volume_protrusion import compute_b_volume_protrusion_guide


R = 1.7
CLIP = 1.0


def test_node_inside_B_gets_zero_guide():
    # Node sits 0.5 inside the radius of a B atom -> inside volume -> zero.
    b = torch.tensor([[0.0, 0.0, 0.0]])
    p = torch.tensor([[1.2, 0.0, 0.0]])  # d = 1.2 < r = 1.7
    g = compute_b_volume_protrusion_guide(p, b, R, CLIP)
    assert torch.allclose(g, torch.zeros_like(g)), g


def test_node_outside_B_points_toward_nearest():
    b = torch.tensor([[0.0, 0.0, 0.0]])
    p = torch.tensor([[5.0, 0.0, 0.0]])  # d = 5.0 > r
    g = compute_b_volume_protrusion_guide(p, b, R, CLIP)
    # Direction must be toward origin: negative x, zero y/z.
    assert g[0, 0] < 0
    assert torch.allclose(g[0, 1:], torch.zeros(2), atol=1e-6)


def test_magnitude_never_exceeds_clip():
    b = torch.tensor([[0.0, 0.0, 0.0]])
    # Node very far -> protrusion huge -> must be clipped to CLIP.
    p = torch.tensor([[50.0, 0.0, 0.0]])
    g = compute_b_volume_protrusion_guide(p, b, R, CLIP)
    assert torch.allclose(g.norm(dim=1), torch.tensor([CLIP]), atol=1e-5), g.norm(dim=1)


def test_nearest_among_multiple_B_atoms():
    # Node at x=15 is OUTSIDE both B atoms (nearest is x=10, d=5 > r).
    # Must be pushed toward the nearer atom at x=10, i.e. in -x.
    b = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    p = torch.tensor([[15.0, 0.0, 0.0]])
    g = compute_b_volume_protrusion_guide(p, b, R, CLIP)
    assert g[0, 0] < 0  # pushed toward x=10, i.e. -x


def test_empty_inputs_return_zero():
    b = torch.zeros((0, 3))
    p = torch.zeros((0, 3))
    g = compute_b_volume_protrusion_guide(p, b, R, CLIP)
    assert g.shape == (0, 3)


def test_node_exactly_on_B_no_nan():
    b = torch.tensor([[0.0, 0.0, 0.0]])
    p = torch.tensor([[0.0, 0.0, 0.0]])  # d = 0, inside r -> zero, no nan
    g = compute_b_volume_protrusion_guide(p, b, R, CLIP)
    assert not torch.isnan(g).any()
    assert torch.allclose(g, torch.zeros_like(g))
