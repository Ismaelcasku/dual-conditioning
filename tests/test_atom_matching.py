from __future__ import annotations

from rdkit import Chem
from rdkit.Geometry import Point3D

from dual_conditioning.evaluation.atom_matching import match_fixed_atoms


def make_carbon_cloud(xs):
    molecule = Chem.MolFromSmiles(".".join("[C]" for _ in xs))
    conformer = Chem.Conformer(len(xs))
    for index, x in enumerate(xs):
        conformer.SetAtomPosition(index, Point3D(float(x), 0.0, 0.0))
    molecule.AddConformer(conformer)
    return molecule


def test_hungarian_assignment_finds_global_minimum():
    reference = make_carbon_cloud([0.0, 1.0, 3.0])
    generated = make_carbon_cloud([1.0, 2.0, 4.0])
    result = match_fixed_atoms(reference, generated, [0, 1, 2])
    assert result.assignment_ok
    assert sum(result.distances_angstrom) == 3.0


def test_element_identity_is_hard_constraint():
    reference = Chem.MolFromSmiles("[C].[N]")
    generated = Chem.MolFromSmiles("[C].[C]")
    for molecule in (reference, generated):
        conformer = Chem.Conformer(2)
        conformer.SetAtomPosition(0, Point3D(0, 0, 0))
        conformer.SetAtomPosition(1, Point3D(1, 0, 0))
        molecule.AddConformer(conformer)
    result = match_fixed_atoms(reference, generated, [0, 1])
    assert not result.assignment_ok
    assert not result.element_match_ok
