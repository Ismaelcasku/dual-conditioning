from __future__ import annotations

from rdkit import Chem
from rdkit.Geometry import Point3D

from dual_conditioning.evaluation.connectivity import audit_interfragment_geometry, fragment_summary


def make(smiles: str, xs: list[float]) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None and molecule.GetNumAtoms() == len(xs)
    conformer = Chem.Conformer(len(xs))
    for index, x in enumerate(xs):
        conformer.SetAtomPosition(index, Point3D(float(x), 0.0, 0.0))
    molecule.AddConformer(conformer)
    return molecule


def test_connected_molecule():
    molecule = make("CC", [0.0, 1.54])
    summary = fragment_summary(molecule)
    audit = audit_interfragment_geometry(molecule, summary)
    assert summary.connected
    assert audit.classification == "single_heavy_component"


def test_potential_missing_bond():
    molecule = make("[CH3].[CH3]", [0.0, 1.5])
    audit = audit_interfragment_geometry(molecule)
    assert audit.classification == "potential_missing_bond"
    assert audit.n_bond_distance_pairs_with_headroom == 1


def test_bond_distance_valence_limited():
    molecule = make("[CH4].[CH4]", [0.0, 1.5])
    audit = audit_interfragment_geometry(molecule)
    assert audit.classification == "bond_distance_valence_limited"


def test_close_nonbonded():
    molecule = make("[CH3].[CH3]", [0.0, 2.4])
    audit = audit_interfragment_geometry(molecule)
    assert audit.classification == "close_nonbonded"


def test_geometrically_separated():
    molecule = make("[CH3].[CH3]", [0.0, 4.0])
    audit = audit_interfragment_geometry(molecule)
    assert audit.classification == "geometrically_separated"


def test_exhaustive_search_finds_nonclosest_viable_pair():
    # Closest normalized pair is saturated-saturated; a slightly more distant
    # methyl-methyl pair is still bond-compatible and has valence headroom.
    molecule = make("[CH4].[CH4].[CH3].[CH3]", [0.0, 1.40, 10.0, 11.50])
    audit = audit_interfragment_geometry(molecule)
    assert audit.minimum_ratio_pair is not None
    assert not audit.minimum_ratio_pair.both_have_valence_headroom
    assert audit.n_bond_distance_pairs_with_headroom >= 1
    assert audit.classification == "potential_missing_bond"
