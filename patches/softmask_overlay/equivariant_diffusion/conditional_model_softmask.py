#!/usr/bin/env python3
"""Centered stage-wise soft-scaffold inpainting for the vendored DiffSBDD sampler.

This module leaves ``equivariant_diffusion/conditional_model.py`` untouched.
``inpaint_softmask.py`` imports :func:`install_softmask_patch` before importing
``lightning_modules``; the function then replaces only ``ConditionalDDPM.inpaint``
at runtime.

Mask semantics
--------------
``lig_hard``
    Warhead atoms. Coordinates and feature channels are re-injected exactly as
    in hard inpainting at every reverse step.
``lig_fixed - lig_hard``
    Previously generated parent scaffold. Feature channels remain hard. The
    coordinate channels are mixed at the current noisy timestep with scalar
    ``soft_scaffold_rho``.
``1 - lig_fixed``
    Newly generated atoms.

For rho=1 the method follows the original hard-inpainting path exactly. For
rho<1, the mixed ligand state is translated after every blend so its
per-sample coordinate mean is zero; the pocket receives the same translation,
preserving all ligand-pocket relative geometry.
"""

from __future__ import annotations

import inspect
import os
from typing import Tuple

import torch
from torch_scatter import scatter_mean


def _column_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != 2 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape [N] or [N,1], got {tuple(mask.shape)}")
    return mask


def validate_soft_masks(
    lig_fixed: torch.Tensor,
    lig_hard: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard, soft and free masks after validating their partition."""
    fixed = _column_mask(lig_fixed, name="lig_fixed").float()
    hard = _column_mask(lig_hard, name="lig_hard").float()

    if fixed.shape != hard.shape:
        raise ValueError(
            f"lig_fixed and lig_hard must have equal shapes: "
            f"{tuple(fixed.shape)} != {tuple(hard.shape)}"
        )
    if torch.any((fixed < 0) | (fixed > 1)):
        raise ValueError("lig_fixed must be binary")
    if torch.any((hard < 0) | (hard > 1)):
        raise ValueError("lig_hard must be binary")
    if torch.any(hard > fixed):
        raise ValueError("Every hard atom must also be part of lig_fixed")

    soft = fixed - hard
    free = 1.0 - fixed
    return hard, soft, free


def blend_noisy_states(
    z_known: torch.Tensor,
    z_unknown: torch.Tensor,
    lig_fixed: torch.Tensor,
    lig_hard: torch.Tensor,
    rho: float,
    n_dims: int,
) -> torch.Tensor:
    """Combine known and model states at the same diffusion timestep.

    Coordinate channels of the soft parent are interpolated. Feature channels
    of the whole parent remain hard-inpainted. This function is deliberately
    pure so it can be unit-tested without loading a checkpoint.
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"soft_scaffold_rho must be in [0,1], got {rho}")
    if z_known.shape != z_unknown.shape:
        raise ValueError("z_known and z_unknown must have identical shapes")

    hard, soft, free = validate_soft_masks(lig_fixed, lig_hard)
    hard = hard.to(device=z_known.device, dtype=z_known.dtype)
    soft = soft.to(device=z_known.device, dtype=z_known.dtype)
    free = free.to(device=z_known.device, dtype=z_known.dtype)

    x_known = z_known[:, :n_dims]
    x_unknown = z_unknown[:, :n_dims]
    h_known = z_known[:, n_dims:]
    h_unknown = z_unknown[:, n_dims:]

    x_out = (
        hard * x_known
        + soft * (rho * x_known + (1.0 - rho) * x_unknown)
        + free * x_unknown
    )
    h_out = (hard + soft) * h_known + free * h_unknown
    return torch.cat((x_out, h_out), dim=1)


def _supports_guidance_kwargs(method) -> bool:
    params = inspect.signature(method).parameters
    return "lig_fixed" in params and "b_internal_uncentered" in params


def recenter_ligand_and_pocket(
    z_lig: torch.Tensor,
    xh_pocket: torch.Tensor,
    ligand_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    n_dims: int,
):
    """Restore the zero-COM invariant without changing relative geometry.

    The mixed soft-mask state can acquire a non-zero coordinate mean because
    hard, soft and free subsets are taken from different noisy states. DiffSBDD
    requires each ligand sample to have zero coordinate mean before the next
    reverse-diffusion call.

    We therefore subtract the ligand mean from every ligand atom and apply the
    same translation to its pocket atoms. Internal ligand distances, internal
    pocket distances and ligand-pocket relative coordinates are unchanged.
    """

    if z_lig.ndim != 2 or xh_pocket.ndim != 2:
        raise ValueError("z_lig and xh_pocket must be rank-2 tensors")
    if n_dims < 1:
        raise ValueError("n_dims must be positive")
    if z_lig.shape[1] < n_dims or xh_pocket.shape[1] < n_dims:
        raise ValueError("Coordinate dimension exceeds tensor width")

    ligand_mask = ligand_mask.long().view(-1)
    pocket_mask = pocket_mask.long().view(-1)

    if len(ligand_mask) != len(z_lig):
        raise ValueError("ligand_mask length does not match z_lig")
    if len(pocket_mask) != len(xh_pocket):
        raise ValueError("pocket_mask length does not match xh_pocket")

    ligand_com = scatter_mean(
        z_lig[:, :n_dims],
        ligand_mask,
        dim=0,
    )

    z_centered = z_lig.clone()
    pocket_centered = xh_pocket.clone()

    z_centered[:, :n_dims] = (
        z_centered[:, :n_dims] - ligand_com[ligand_mask]
    )
    pocket_centered[:, :n_dims] = (
        pocket_centered[:, :n_dims] - ligand_com[pocket_mask]
    )

    return z_centered, pocket_centered, ligand_com


@torch.no_grad()
def softmask_inpaint(
    self,
    ligand,
    pocket,
    lig_fixed,
    resamplings=1,
    return_frames=1,
    timesteps=None,
    center="ligand",
    b_cloud_raw=None,
    lambda_global=0.0,
    guide_r_raw=1.7,
    guide_clip_raw=1.0,
    *,
    lig_hard=None,
    soft_scaffold_rho=1.0,
):
    """RePaint inpainting with a hard warhead and a stage-wise soft parent."""
    rho = float(soft_scaffold_rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"soft_scaffold_rho must be in [0,1], got {rho}")

    timesteps = self.T if timesteps is None else timesteps
    assert 0 < return_frames <= timesteps
    assert timesteps % return_frames == 0

    lig_fixed = _column_mask(lig_fixed, name="lig_fixed")
    if lig_hard is None:
        lig_hard = lig_fixed.clone()
    lig_hard = _column_mask(lig_hard, name="lig_hard")
    hard_mask, soft_mask, _ = validate_soft_masks(lig_fixed, lig_hard)

    lig_fixed = lig_fixed.to(pocket["x"].device)
    lig_hard = lig_hard.to(pocket["x"].device)
    hard_mask = hard_mask.to(pocket["x"].device)
    soft_mask = soft_mask.to(pocket["x"].device)

    n_samples = len(ligand["size"])
    device = pocket["x"].device

    # Exact legacy path at rho=1. For rho<1, the hard warhead defines the frame
    # and the parent scaffold is treated as mobile by the guidance field.
    legacy_hard_path = rho == 1.0
    center_mask = lig_fixed if legacy_hard_path else lig_hard
    guide_fixed_mask = lig_fixed if legacy_hard_path else lig_hard

    ligand, pocket = self.normalize(ligand, pocket)
    xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
    com_pocket_0 = scatter_mean(pocket["x"], pocket["mask"], dim=0)
    xh0_ligand = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
    xh_ligand = xh0_ligand.clone()

    if center == "ligand":
        center_sel = center_mask.bool().view(-1)
        if int(center_sel.sum()) == 0:
            raise ValueError("The hard/centering mask contains no atoms")
        mean_known = scatter_mean(
            ligand["x"][center_sel],
            ligand["mask"][center_sel],
            dim=0,
        )
    elif center == "pocket":
        mean_known = scatter_mean(pocket["x"], pocket["mask"], dim=0)
    else:
        raise NotImplementedError(f"Centering option {center} not implemented")

    mu_lig_x = mean_known
    mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
    mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[ligand["mask"]]
    sigma = torch.ones_like(pocket["size"]).unsqueeze(1)

    use_global_guide = b_cloud_raw is not None and lambda_global != 0.0
    guide_space = os.environ.get("DC_GUIDE_SPACE", "x0")
    b_internal_uncentered = None
    guide_r_internal = guide_r_raw
    guide_clip_internal = guide_clip_raw
    mobile_node_mask = None

    if use_global_guide:
        coord_scale = self.norm_values[0]
        b_internal_uncentered = b_cloud_raw.to(device) / coord_scale
        guide_r_internal = guide_r_raw / coord_scale
        guide_clip_internal = guide_clip_raw / coord_scale
        mobile_node_mask = (1 - guide_fixed_mask).bool().view(-1)

        if not _supports_guidance_kwargs(self.sample_p_zs_given_zt):
            raise RuntimeError(
                "The active conditional_model.py does not expose the guidance "
                "arguments required by this project. Use the already patched "
                "vendor version before installing the soft-mask runtime patch."
            )

    z_lig, xh_pocket = self.sample_normal_zero_com(
        mu_lig, xh0_pocket, sigma, ligand["mask"], pocket["mask"]
    )

    out_lig = torch.zeros((return_frames,) + z_lig.size(), device=z_lig.device)
    out_pocket = torch.zeros((return_frames,) + xh_pocket.size(), device=device)

    for s in reversed(range(0, timesteps)):
        for u in range(resamplings):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            gamma_t = self.gamma(t_array)
            gamma_s = self.gamma(s_array)

            kwargs = {}
            if use_global_guide and guide_space == "x0":
                kwargs = {
                    "lig_fixed": guide_fixed_mask,
                    "b_internal_uncentered": b_internal_uncentered,
                    "mean_known": mean_known,
                    "lambda_global": lambda_global,
                    "guide_r_internal": guide_r_internal,
                    "guide_clip_internal": guide_clip_internal,
                    "guide_space": guide_space,
                    "dbg_s": s,
                    "dbg_T": timesteps,
                }

            z_lig_unknown, xh_pocket = self.sample_p_zs_given_zt(
                s_array,
                t_array,
                z_lig,
                xh_pocket,
                ligand["mask"],
                pocket["mask"],
                **kwargs,
            )

            com_pocket = scatter_mean(
                xh_pocket[:, : self.n_dims], pocket["mask"], dim=0
            )
            xh_ligand[:, : self.n_dims] = (
                ligand["x"] + (com_pocket - com_pocket_0)[ligand["mask"]]
            )
            z_lig_known, xh_pocket, _ = self.noised_representation(
                xh_ligand,
                xh_pocket,
                ligand["mask"],
                pocket["mask"],
                gamma_s,
            )

            if use_global_guide and guide_space == "z" and u == resamplings - 1:
                from guidance.b_volume_protrusion import (
                    compute_b_volume_protrusion_guide,
                )

                debug = os.environ.get("DC_GUIDE_DEBUG")
                for sample_idx in range(n_samples):
                    sel = (ligand["mask"] == sample_idx) & mobile_node_mask
                    if int(sel.sum()) == 0:
                        continue
                    x_mobile = z_lig_unknown[sel][:, : self.n_dims]
                    b_i = b_internal_uncentered - mean_known[sample_idx]
                    guide = compute_b_volume_protrusion_guide(
                        x_mobile, b_i, guide_r_internal, guide_clip_internal
                    )
                    if guide.shape[0] > 0:
                        guide = guide - guide.mean(dim=0, keepdim=True)
                    idx = sel.nonzero(as_tuple=True)[0]
                    z_lig_unknown[idx, : self.n_dims] = (
                        x_mobile + lambda_global * guide
                    )
                    if debug and sample_idx == 0 and s in (
                        timesteps - 1,
                        timesteps // 2,
                        0,
                    ):
                        print(
                            f"[SOFTMASK_GUIDE_DBG] s={s} rho={rho:.3f} "
                            f"mobile={int(sel.sum())}",
                            flush=True,
                        )

            # Preserve the exact original centering operation at rho=1. At
            # rho<1 only the warhead anchors the coordinate frame.
            align_sel = center_mask.bool().view(-1)
            com_noised = scatter_mean(
                z_lig_known[align_sel][:, : self.n_dims],
                ligand["mask"][align_sel],
                dim=0,
            )
            com_denoised = scatter_mean(
                z_lig_unknown[align_sel][:, : self.n_dims],
                ligand["mask"][align_sel],
                dim=0,
            )
            dx = com_denoised - com_noised
            z_lig_known[:, : self.n_dims] = (
                z_lig_known[:, : self.n_dims] + dx[ligand["mask"]]
            )
            xh_pocket[:, : self.n_dims] = (
                xh_pocket[:, : self.n_dims] + dx[pocket["mask"]]
            )

            z_lig = blend_noisy_states(
                z_known=z_lig_known,
                z_unknown=z_lig_unknown,
                lig_fixed=lig_fixed,
                lig_hard=lig_hard,
                rho=rho,
                n_dims=self.n_dims,
            )

            # The rho=1 branch remains byte-for-byte equivalent to the legacy
            # hard-inpainting path. A subset-wise blend at rho<1 can break the
            # zero-COM invariant required by sample_p_zs_given_zt, so restore it
            # by translating ligand and pocket together.
            if not legacy_hard_path:
                z_lig, xh_pocket, residual_com = recenter_ligand_and_pocket(
                    z_lig,
                    xh_pocket,
                    ligand["mask"],
                    pocket["mask"],
                    self.n_dims,
                )
                if (
                    os.environ.get("DC_SOFTMASK_CENTER_DEBUG") == "1"
                    and u == resamplings - 1
                    and s in (timesteps - 1, timesteps // 2, 0)
                ):
                    max_residual = float(
                        residual_com.norm(dim=1).max().detach().cpu()
                    )
                    post_com = scatter_mean(
                        z_lig[:, : self.n_dims],
                        ligand["mask"],
                        dim=0,
                    )
                    max_post = float(
                        post_com.norm(dim=1).max().detach().cpu()
                    )
                    print(
                        f"[SOFTMASK_CENTER_DBG] s={s} rho={rho:.3f} "
                        f"pre={max_residual:.6e} post={max_post:.6e}",
                        flush=True,
                    )

            if u < resamplings - 1:
                z_lig, xh_pocket = self.sample_p_zt_given_zs(
                    z_lig,
                    xh_pocket,
                    ligand["mask"],
                    pocket["mask"],
                    gamma_t,
                    gamma_s,
                )

            if u == resamplings - 1 and (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = self.unnormalize_z(
                    z_lig, xh_pocket
                )

    x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
        z_lig,
        xh_pocket,
        ligand["mask"],
        pocket["mask"],
        n_samples,
    )
    out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
    out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

    return (
        out_lig.squeeze(0),
        out_pocket.squeeze(0),
        ligand["mask"],
        pocket["mask"],
    )


def install_softmask_patch() -> None:
    """Install the centered alternate inpaint method at runtime."""
    from equivariant_diffusion import conditional_model as base

    if getattr(
        base.ConditionalDDPM.inpaint,
        "_dc_softmask_centered",
        False,
    ):
        return

    softmask_inpaint._dc_softmask = True
    softmask_inpaint._dc_softmask_centered = True
    base.ConditionalDDPM.inpaint = softmask_inpaint

    if hasattr(base, "SimpleConditionalDDPM"):
        base.SimpleConditionalDDPM.inpaint = softmask_inpaint

