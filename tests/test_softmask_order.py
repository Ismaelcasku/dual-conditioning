#!/usr/bin/env python3
"""Order-recovery tests for soft-scaffold propagation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

PROJECT = Path(__file__).resolve().parents[1]
DIFFSBDD = PROJECT / "external" / "DiffSBDD"
if not DIFFSBDD.is_dir():
    raise SystemExit(
        "DiffSBDD checkout missing. Run ./setup_diffsbdd.sh before this test."
    )
sys.path.insert(0, str(DIFFSBDD))

from softmask_atom_order import (
    AtomOrderError,
    canonicalize_point_cloud,
    canonicalize_rdkit_mol,
    match_reference_atoms,
)


def make_mol(atomic_numbers, coords):
    editable = Chem.RWMol()
    for atomic_number in atomic_numbers:
        editable.AddAtom(Chem.Atom(int(atomic_number)))
    mol = editable.GetMol()
    conf = Chem.Conformer(len(atomic_numbers))
    for idx, xyz in enumerate(np.asarray(coords, dtype=float)):
        conf.SetAtomPosition(idx, tuple(float(value) for value in xyz))
    mol.AddConformer(conf)
    return mol


def test_type_constrained_matching():
    reference_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    reference_types = np.array([6, 7, 8])
    candidate_xyz = np.array(
        [[2.02, 0.0, 0.0], [8.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.03, 0.0, 0.0]]
    )
    candidate_types = np.array([8, 6, 6, 7])
    match = match_reference_atoms(
        reference_xyz,
        reference_types,
        candidate_xyz,
        candidate_types,
        hard_count=3,
        hard_tolerance=0.10,
    )
    assert match.candidate_indices == (2, 3, 0)
    assert match.max_distance < 0.04


def test_point_cloud_parent_first():
    parent_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    parent_types = torch.tensor([0, 1, 2])
    sample_xyz = torch.tensor(
        [[9.0, 0.0, 0.0], [2.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.03, 0.0, 0.0]]
    )
    sample_types = torch.tensor([3, 2, 0, 1])
    ordered_xyz, ordered_types, match = canonicalize_point_cloud(
        sample_xyz,
        sample_types,
        parent_xyz,
        parent_types,
        hard_count=3,
        hard_tolerance=0.10,
    )
    assert match.candidate_indices == (2, 3, 1)
    assert torch.equal(ordered_types[:3], parent_types)
    assert torch.equal(ordered_types[3:], torch.tensor([3]))
    assert torch.allclose(
        ordered_xyz[:3],
        torch.tensor([[0.02, 0.0, 0.0], [1.03, 0.0, 0.0], [2.01, 0.0, 0.0]]),
    )


def test_hard_tolerance_rejected():
    try:
        match_reference_atoms(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([6]),
            np.array([[0.30, 0.0, 0.0]]),
            np.array([6]),
            hard_count=1,
            hard_tolerance=0.20,
        )
    except AtomOrderError:
        return
    raise AssertionError("Hard displacement above tolerance was accepted")


def test_rdkit_renumbering():
    parent_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    parent_types = np.array([6, 7, 8])
    child_xyz = np.array(
        [[2.01, 0.0, 0.0], [8.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.02, 0.0, 0.0]]
    )
    child = make_mol([8, 6, 6, 7], child_xyz)
    ordered, match = canonicalize_rdkit_mol(
        child,
        parent_xyz,
        parent_types,
        hard_count=3,
        hard_tolerance=0.10,
    )
    assert match.candidate_indices == (2, 3, 0)
    assert [ordered.GetAtomWithIdx(i).GetAtomicNum() for i in range(3)] == [6, 7, 8]


def main() -> None:
    tests = [
        test_type_constrained_matching,
        test_point_cloud_parent_first,
        test_hard_tolerance_rejected,
        test_rdkit_renumbering,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\nAll {len(tests)} soft-mask order tests passed.")


if __name__ == "__main__":
    main()
