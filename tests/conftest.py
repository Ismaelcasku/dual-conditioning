from __future__ import annotations

from rdkit import Chem
from rdkit.Geometry import Point3D


def molecule_with_coordinates(smiles: str, coordinates: list[tuple[float, float, float]]) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    assert molecule is not None
    assert molecule.GetNumAtoms() == len(coordinates)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    return molecule
