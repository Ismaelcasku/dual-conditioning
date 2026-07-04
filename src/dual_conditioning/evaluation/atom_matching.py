"""Element-constrained fixed-atom matching using the Hungarian algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class FixedAtomMatch:
    assignment_ok: bool
    element_match_ok: bool
    reference_atom_indices: tuple[int, ...]
    generated_atom_indices: tuple[int, ...]
    distances_angstrom: tuple[float, ...]
    rmsd_angstrom: float | None
    mean_distance_angstrom: float | None
    max_distance_angstrom: float | None
    message: str = ""

    def all_within(self, threshold_angstrom: float) -> bool:
        return (
            self.assignment_ok
            and self.element_match_ok
            and len(self.distances_angstrom) == len(self.reference_atom_indices)
            and all(distance <= threshold_angstrom for distance in self.distances_angstrom)
        )


def _coords(mol: Chem.Mol, atom_indices: Sequence[int]) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = mol.GetConformer()
    return np.asarray(
        [
            [
                conf.GetAtomPosition(int(index)).x,
                conf.GetAtomPosition(int(index)).y,
                conf.GetAtomPosition(int(index)).z,
            ]
            for index in atom_indices
        ],
        dtype=float,
    )


def match_fixed_atoms(
    reference: Chem.Mol,
    generated: Chem.Mol,
    fixed_reference_indices: Sequence[int],
    *,
    heavy_only: bool = True,
) -> FixedAtomMatch:
    """Match fixed reference atoms to unique generated atoms.

    Element identity is a hard constraint. Returned generated indices are the
    original RDKit atom indices, so they can be mapped directly to components.
    """
    reference_indices = tuple(int(index) for index in fixed_reference_indices)
    generated_indices = tuple(
        atom.GetIdx()
        for atom in generated.GetAtoms()
        if (not heavy_only or atom.GetAtomicNum() > 1)
    )

    if len(generated_indices) < len(reference_indices):
        return FixedAtomMatch(
            False,
            False,
            reference_indices,
            (),
            (),
            None,
            None,
            None,
            f"generated molecule has {len(generated_indices)} eligible atoms; "
            f"{len(reference_indices)} fixed atoms are required",
        )

    ref_coords = _coords(reference, reference_indices)
    gen_coords = _coords(generated, generated_indices)
    ref_symbols = [reference.GetAtomWithIdx(index).GetSymbol() for index in reference_indices]
    gen_symbols = [generated.GetAtomWithIdx(index).GetSymbol() for index in generated_indices]

    distances = np.linalg.norm(ref_coords[:, None, :] - gen_coords[None, :, :], axis=2)
    costs = distances.copy()
    for i, ref_symbol in enumerate(ref_symbols):
        for j, gen_symbol in enumerate(gen_symbols):
            if ref_symbol != gen_symbol:
                costs[i, j] = np.inf

    if not np.all(np.isfinite(costs).any(axis=1)):
        return FixedAtomMatch(
            False,
            False,
            reference_indices,
            (),
            (),
            None,
            None,
            None,
            "at least one fixed atom has no generated atom with a matching element",
        )

    finite_costs = np.where(np.isfinite(costs), costs, 1e9)
    row_indices, col_indices = linear_sum_assignment(finite_costs)
    assignment = {int(row): int(col) for row, col in zip(row_indices, col_indices)}
    if len(assignment) != len(reference_indices):
        return FixedAtomMatch(
            False,
            False,
            reference_indices,
            (),
            (),
            None,
            None,
            None,
            "incomplete fixed-atom assignment",
        )

    selected_positions = [assignment[index] for index in range(len(reference_indices))]
    matched_indices = tuple(generated_indices[position] for position in selected_positions)
    matched_symbols = [generated.GetAtomWithIdx(index).GetSymbol() for index in matched_indices]
    element_ok = all(a == b for a, b in zip(ref_symbols, matched_symbols))
    matched_distances = tuple(
        float(distances[index, selected_positions[index]])
        for index in range(len(reference_indices))
    )
    rmsd = float(np.sqrt(np.mean(np.square(matched_distances))))

    return FixedAtomMatch(
        assignment_ok=element_ok,
        element_match_ok=element_ok,
        reference_atom_indices=reference_indices,
        generated_atom_indices=matched_indices,
        distances_angstrom=matched_distances,
        rmsd_angstrom=rmsd,
        mean_distance_angstrom=float(np.mean(matched_distances)),
        max_distance_angstrom=float(np.max(matched_distances)),
        message="" if element_ok else "element mismatch after assignment",
    )
