"""Cold tests for codes/guidance/b_shape_overlap.py (Experiment 03 shape-overlap guide).

Run inside the sandbox:
  PYTHONPATH=codes:codes/vendor/diffsbdd python -m pytest tests/guidance/test_b_shape_overlap.py -q
"""
import torch

from dual_conditioning.guidance.shape_overlap import (
    compute_shape_overlap_guide as guide,
    shape_tanimoto_overlap as T,
)


def _proper_rotation(seed=0):
    torch.manual_seed(seed)
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return R


def test_gradient_ascends_shape_tanimoto():
    torch.manual_seed(0)
    free = torch.randn(8, 3)
    B = torch.randn(12, 3) * 1.5 + torch.tensor([1.0, 0.0, 0.0])
    g = guide(free, B, clip=10.0)
    t0 = T(free, B).item()
    t1 = T(free + 0.05 * g, B).item()
    assert t1 > t0


def test_anticlustering_separates_coincident_atoms():
    # The V_GG self-overlap term must push near-coincident free atoms APART.
    clump = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    B = torch.randn(15, 3) * 2.0
    g = guide(clump, B, clip=10.0)
    rel = g[1] - g[0]
    sep_dir = clump[1] - clump[0]
    sep_dir = sep_dir / sep_dir.norm()
    assert (rel @ sep_dir).item() > 0


def test_far_atom_pulled_toward_B():
    far = torch.tensor([[10.0, 0.0, 0.0]])
    B = torch.zeros(5, 3) + torch.randn(5, 3) * 0.3
    g = guide(far, B, clip=10.0)
    assert (g[0] @ (B.mean(0) - far[0])).item() > 0


def test_se3_equivariance():
    torch.manual_seed(1)
    free = torch.randn(8, 3)
    B = torch.randn(12, 3) * 1.5
    R = _proper_rotation(2)
    gA = guide(free, B, clip=10.0)
    gB = guide(free @ R.T, B @ R.T, clip=10.0)
    assert torch.allclose(gB, gA @ R.T, atol=1e-4)


def test_empty_inputs():
    B = torch.randn(5, 3)
    assert guide(torch.empty(0, 3), B).shape == (0, 3)
    free = torch.randn(4, 3)
    assert torch.allclose(guide(free, torch.empty(0, 3)), torch.zeros_like(free))


def test_perfect_overlap_tiny_guide():
    B = torch.randn(10, 3)
    g = guide(B.clone(), B, clip=10.0)
    assert g.norm() < 1e-2
    assert abs(T(B.clone(), B).item() - 1.0) < 1e-3


def test_autograd_matches_numerical_gradient():
    torch.manual_seed(3)
    x = torch.randn(5, 3, dtype=torch.float64)
    B = torch.randn(7, 3, dtype=torch.float64)
    xg = x.clone().requires_grad_(True)
    (an,) = torch.autograd.grad(T(xg, B), xg)
    num = torch.zeros_like(x)
    h = 1e-5
    for i in range(x.shape[0]):
        for j in range(3):
            xp = x.clone(); xp[i, j] += h
            xm = x.clone(); xm[i, j] -= h
            num[i, j] = (T(xp, B) - T(xm, B)) / (2 * h)
    assert torch.allclose(an, num, atol=1e-5)


def test_clip_bounds_per_atom_norm():
    torch.manual_seed(4)
    free = torch.randn(8, 3) * 5
    B = torch.randn(12, 3)
    g = guide(free, B, clip=0.3)
    assert g.norm(dim=1).max().item() <= 0.3 + 1e-6


def test_fixed_atoms_enter_volume():
    torch.manual_seed(5)
    free = torch.randn(6, 3)
    B = torch.randn(9, 3)
    fixed = torch.randn(7, 3)
    assert T(free, B).item() != T(free, B, fixed_coords=fixed).item()
