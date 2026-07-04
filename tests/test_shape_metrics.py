from __future__ import annotations

from rdkit import Chem
from rdkit.Geometry import Point3D

from dual_conditioning.evaluation.shape import ShapeMetricConfig, shape_metrics


def molecule() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, Point3D(0, 0, 0))
    conformer.SetAtomPosition(1, Point3D(1.5, 0, 0))
    mol.AddConformer(conformer)
    return mol


def test_distance_to_similarity_conversion():
    mol = molecule()
    metrics = shape_metrics(mol, mol, ShapeMetricConfig())
    assert abs(metrics.tanimoto_distance) < 1e-12
    assert abs(metrics.tanimoto_similarity - 1.0) < 1e-12
