#!/usr/bin/env python3
"""Order-recovery utilities for stage-wise soft-mask inpainting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


class AtomOrderError(RuntimeError):
    """Raised when a parent-to-child atom correspondence cannot be recovered."""


@dataclass(frozen=True)
class AtomMatch:
    """Parent-to-child assignment in parent order."""

    candidate_indices: tuple[int, ...]
    distances: tuple[float, ...]

    @property
    def rmsd(self) -> float:
        if not self.distances:
            return 0.0
        values = np.asarray(self.distances, dtype=float)
        return float(np.sqrt(np.mean(values * values)))

    @property
    def max_distance(self) -> float:
        return max(self.distances, default=0.0)


def _as_xyz(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise AtomOrderError(f"{name} must have shape [N,3], got {array.shape}")
    if not np.isfinite(array).all():
        raise AtomOrderError(f"{name} contains non-finite coordinates")
    return array


def _as_labels(values, *, name: str) -> np.ndarray:
    return np.asarray(values, dtype=int).reshape(-1)


def match_reference_atoms(
    reference_xyz,
    reference_labels,
    candidate_xyz,
    candidate_labels,
    *,
    hard_count: int = 0,
    hard_tolerance: float = 0.20,
    soft_max_distance: float | None = None,
) -> AtomMatch:
    """Match every reference atom to one candidate atom.

    Matching is one-to-one and constrained to equal integer labels. Labels may
    be atomic numbers or model atom-type indices.
    """

    ref_xyz = _as_xyz(reference_xyz, name="reference_xyz")
    cand_xyz = _as_xyz(candidate_xyz, name="candidate_xyz")
    ref_labels = _as_labels(reference_labels, name="reference_labels")
    cand_labels = _as_labels(candidate_labels, name="candidate_labels")

    n_ref = len(ref_xyz)
    n_cand = len(cand_xyz)

    if len(ref_labels) != n_ref:
        raise AtomOrderError("reference coordinates and labels differ in length")
    if len(cand_labels) != n_cand:
        raise AtomOrderError("candidate coordinates and labels differ in length")
    if n_ref == 0:
        return AtomMatch((), ())
    if n_cand < n_ref:
        raise AtomOrderError(
            f"candidate has fewer atoms than parent: {n_cand} < {n_ref}"
        )
    if not 0 <= int(hard_count) <= n_ref:
        raise AtomOrderError(
            f"hard_count must be in [0,{n_ref}], got {hard_count}"
        )

    distances = np.linalg.norm(
        ref_xyz[:, None, :] - cand_xyz[None, :, :],
        axis=2,
    )
    type_mismatch = ref_labels[:, None] != cand_labels[None, :]

    penalty = 1.0e6
    cost = distances.copy()
    cost[type_mismatch] = penalty

    rows, columns = linear_sum_assignment(cost)
    if len(rows) != n_ref or not np.array_equal(rows, np.arange(n_ref)):
        raise AtomOrderError("Hungarian assignment did not cover every parent atom")
    if np.any(cost[rows, columns] >= penalty):
        missing = []
        for label in np.unique(ref_labels):
            available = int(np.sum(cand_labels == label))
            needed = int(np.sum(ref_labels == label))
            if available < needed:
                missing.append(
                    f"label {int(label)}: candidate {available}, parent {needed}"
                )
        detail = "; ".join(missing) if missing else "no complete type-compatible assignment"
        raise AtomOrderError(detail)

    matched_distances = distances[rows, columns]

    hard_count = int(hard_count)
    if hard_count:
        hard_max = float(matched_distances[:hard_count].max())
        if hard_max > float(hard_tolerance):
            raise AtomOrderError(
                "hard warhead exceeds tolerance: "
                f"{hard_max:.6f} A > {float(hard_tolerance):.6f} A"
            )

    if soft_max_distance is not None and n_ref > hard_count:
        soft_max = float(matched_distances[hard_count:].max())
        if soft_max > float(soft_max_distance):
            raise AtomOrderError(
                "soft parent matching exceeds tolerance: "
                f"{soft_max:.6f} A > {float(soft_max_distance):.6f} A"
            )

    return AtomMatch(
        tuple(int(value) for value in columns.tolist()),
        tuple(float(value) for value in matched_distances.tolist()),
    )


def parent_first_order(
    n_candidate_atoms: int,
    parent_candidate_indices: Sequence[int],
) -> list[int]:
    parent = [int(value) for value in parent_candidate_indices]
    if len(set(parent)) != len(parent):
        raise AtomOrderError("parent assignment contains duplicate candidate indices")
    if any(value < 0 or value >= n_candidate_atoms for value in parent):
        raise AtomOrderError("parent assignment contains an out-of-range index")

    used = set(parent)
    return parent + [
        idx for idx in range(int(n_candidate_atoms)) if idx not in used
    ]


def canonicalize_point_cloud(
    sample_xyz: torch.Tensor,
    sample_labels: torch.Tensor,
    parent_xyz: torch.Tensor,
    parent_labels: torch.Tensor,
    *,
    hard_count: int,
    hard_tolerance: float = 0.20,
    soft_max_distance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, AtomMatch]:
    """Put parent nodes first in a generated tensor point cloud."""

    if sample_xyz.ndim != 2 or sample_xyz.shape[1] != 3:
        raise AtomOrderError(
            f"sample_xyz must have shape [N,3], got {tuple(sample_xyz.shape)}"
        )
    if sample_labels.ndim != 1 or len(sample_labels) != len(sample_xyz):
        raise AtomOrderError("sample labels do not match sample coordinates")
    if parent_xyz.ndim != 2 or parent_xyz.shape[1] != 3:
        raise AtomOrderError("parent_xyz must have shape [N,3]")
    if parent_labels.ndim != 1 or len(parent_labels) != len(parent_xyz):
        raise AtomOrderError("parent labels do not match parent coordinates")

    match = match_reference_atoms(
        parent_xyz.detach().cpu().numpy(),
        parent_labels.detach().cpu().numpy(),
        sample_xyz.detach().cpu().numpy(),
        sample_labels.detach().cpu().numpy(),
        hard_count=hard_count,
        hard_tolerance=hard_tolerance,
        soft_max_distance=soft_max_distance,
    )
    order = parent_first_order(len(sample_xyz), match.candidate_indices)
    index = torch.tensor(order, dtype=torch.long, device=sample_xyz.device)

    reordered_xyz = sample_xyz.index_select(0, index)
    reordered_labels = sample_labels.index_select(0, index).clone()

    n_parent = len(parent_labels)
    reordered_labels[:n_parent] = parent_labels.to(
        device=reordered_labels.device,
        dtype=reordered_labels.dtype,
    )
    return reordered_xyz, reordered_labels, match


def rdkit_arrays(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    if mol is None or mol.GetNumConformers() == 0:
        raise AtomOrderError("RDKit molecule has no conformer")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    labels = np.asarray(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=int,
    )
    return coords, labels


def canonicalize_rdkit_mol(
    mol: Chem.Mol,
    parent_xyz,
    parent_atomic_numbers,
    *,
    hard_count: int,
    hard_tolerance: float = 0.20,
    soft_max_distance: float | None = None,
) -> tuple[Chem.Mol, AtomMatch]:
    child_xyz, child_atomic_numbers = rdkit_arrays(mol)
    match = match_reference_atoms(
        parent_xyz,
        parent_atomic_numbers,
        child_xyz,
        child_atomic_numbers,
        hard_count=hard_count,
        hard_tolerance=hard_tolerance,
        soft_max_distance=soft_max_distance,
    )
    order = parent_first_order(mol.GetNumAtoms(), match.candidate_indices)
    return Chem.RenumberAtoms(mol, order), match


def atom_type_indices_to_atomic_numbers(
    atom_type_indices,
    atom_decoder: Sequence[str],
) -> np.ndarray:
    periodic = Chem.GetPeriodicTable()
    result = []
    for raw_index in np.asarray(atom_type_indices, dtype=int).reshape(-1):
        try:
            symbol = str(atom_decoder[int(raw_index)])
        except (IndexError, TypeError) as exc:
            raise AtomOrderError(
                f"invalid model atom-type index: {int(raw_index)}"
            ) from exc
        atomic_number = int(periodic.GetAtomicNumber(symbol))
        if atomic_number <= 0:
            raise AtomOrderError(f"unknown element symbol in decoder: {symbol}")
        result.append(atomic_number)
    return np.asarray(result, dtype=int)
