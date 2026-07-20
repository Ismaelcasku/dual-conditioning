#!/usr/bin/env python3
"""Contract tests for the final centered soft-scaffold sampler overlay."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
DIFFSBDD = PROJECT / "external" / "DiffSBDD"
if not DIFFSBDD.is_dir():
    raise SystemExit(
        "DiffSBDD checkout missing. Run ./setup_diffsbdd.sh before this test."
    )
sys.path.insert(0, str(DIFFSBDD))

from torch_scatter import scatter_mean
from equivariant_diffusion.conditional_model_softmask import (
    blend_noisy_states,
    install_softmask_patch,
    recenter_ligand_and_pocket,
    validate_soft_masks,
)


def assert_close(first, second, atol=1e-6):
    if not torch.allclose(first, second, atol=atol, rtol=0):
        raise AssertionError(f"\nfirst={first}\nsecond={second}")


def toy_states():
    known = torch.tensor(
        [
            [10.0, 10.0, 10.0, 1.0, 0.0],
            [20.0, 20.0, 20.0, 0.0, 1.0],
            [30.0, 30.0, 30.0, 1.0, 0.0],
            [40.0, 40.0, 40.0, 0.0, 1.0],
        ]
    )
    unknown = torch.tensor(
        [
            [1.0, 1.0, 1.0, 9.0, 9.0],
            [2.0, 2.0, 2.0, 8.0, 8.0],
            [3.0, 3.0, 3.0, 7.0, 7.0],
            [4.0, 4.0, 4.0, 6.0, 6.0],
        ]
    )
    fixed = torch.tensor([1, 1, 1, 0])
    hard = torch.tensor([1, 0, 0, 0])
    return known, unknown, fixed, hard


def test_mask_partition():
    fixed = torch.tensor([1, 1, 1, 0, 0])
    hard = torch.tensor([1, 0, 0, 0, 0])
    h, s, f = validate_soft_masks(fixed, hard)
    assert_close(h.view(-1), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert_close(s.view(-1), torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]))
    assert_close(f.view(-1), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_rho_one_equals_hard_inpainting():
    known, unknown, fixed, hard = toy_states()
    result = blend_noisy_states(known, unknown, fixed, hard, rho=1.0, n_dims=3)
    expected = known * fixed[:, None] + unknown * (1 - fixed[:, None])
    assert_close(result, expected, atol=0.0)


def test_rho_zero_releases_soft_coordinates_but_not_features():
    known, unknown, fixed, hard = toy_states()
    result = blend_noisy_states(known, unknown, fixed, hard, rho=0.0, n_dims=3)
    assert_close(result[0], known[0], atol=0.0)
    assert_close(result[1:3, :3], unknown[1:3, :3], atol=0.0)
    assert_close(result[1:3, 3:], known[1:3, 3:], atol=0.0)
    assert_close(result[3], unknown[3], atol=0.0)


def test_intermediate_rho():
    known, unknown, fixed, hard = toy_states()
    result = blend_noisy_states(known, unknown, fixed, hard, rho=0.25, n_dims=3)
    expected = 0.25 * known[1:3, :3] + 0.75 * unknown[1:3, :3]
    assert_close(result[1:3, :3], expected, atol=0.0)


def test_invalid_masks_rejected():
    try:
        validate_soft_masks(torch.tensor([1, 0]), torch.tensor([1, 1]))
    except ValueError:
        return
    raise AssertionError("Hard atoms outside the fixed mask were accepted")


def test_recenter_sets_zero_mean_and_preserves_geometry():
    z_lig = torch.tensor(
        [
            [2.0, 1.0, 0.0, 1.0],
            [5.0, 1.0, 0.0, 2.0],
        ]
    )
    pocket = torch.tensor(
        [
            [8.0, 4.0, 0.0, 3.0],
            [9.0, 7.0, 0.0, 4.0],
        ]
    )
    lig_mask = torch.zeros(2, dtype=torch.long)
    pocket_mask = torch.zeros(2, dtype=torch.long)

    z_centered, p_centered, _ = recenter_ligand_and_pocket(
        z_lig, pocket, lig_mask, pocket_mask, n_dims=3
    )
    post = scatter_mean(z_centered[:, :3], lig_mask, dim=0)
    assert_close(post, torch.zeros_like(post))
    assert_close(torch.cdist(z_centered[:, :3], z_centered[:, :3]),
                 torch.cdist(z_lig[:, :3], z_lig[:, :3]))
    assert_close(torch.cdist(p_centered[:, :3], p_centered[:, :3]),
                 torch.cdist(pocket[:, :3], pocket[:, :3]))
    assert_close(torch.cdist(z_centered[:, :3], p_centered[:, :3]),
                 torch.cdist(z_lig[:, :3], pocket[:, :3]))
    assert_close(z_centered[:, 3:], z_lig[:, 3:])
    assert_close(p_centered[:, 3:], pocket[:, 3:])


def test_centered_runtime_patch_installation():
    from equivariant_diffusion import conditional_model as base

    original = base.ConditionalDDPM.inpaint
    try:
        install_softmask_patch()
        installed = base.ConditionalDDPM.inpaint
        assert getattr(installed, "_dc_softmask_centered", False)
        if hasattr(base, "SimpleConditionalDDPM"):
            assert base.SimpleConditionalDDPM.inpaint is installed
    finally:
        base.ConditionalDDPM.inpaint = original
        if hasattr(base, "SimpleConditionalDDPM"):
            base.SimpleConditionalDDPM.inpaint = original


def main() -> None:
    tests = [
        test_mask_partition,
        test_rho_one_equals_hard_inpainting,
        test_rho_zero_releases_soft_coordinates_but_not_features,
        test_intermediate_rho,
        test_invalid_masks_rejected,
        test_recenter_sets_zero_mean_and_preserves_geometry,
        test_centered_runtime_patch_installation,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\nAll {len(tests)} soft-mask core tests passed.")


if __name__ == "__main__":
    main()
