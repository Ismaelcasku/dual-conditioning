"""Molecular component analysis and exhaustive interfragment geometry audit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from rdkit import Chem


@dataclass(frozen=True)
class FragmentSummary:
    atom_components: tuple[tuple[int, ...], ...]
    heavy_component_ids: tuple[int, ...]
    heavy_counts: tuple[int, ...]
    parent_component_id: int | None
    atom_to_component: dict[int, int]

    @property
    def n_heavy_components(self) -> int:
        return len(self.heavy_component_ids)

    @property
    def connected(self) -> bool:
        return self.n_heavy_components == 1

    @property
    def total_heavy_atoms(self) -> int:
        return sum(self.heavy_counts[index] for index in self.heavy_component_ids)

    @property
    def parent_heavy_atoms(self) -> int:
        if self.parent_component_id is None:
            return 0
        return self.heavy_counts[self.parent_component_id]

    @property
    def parent_heavy_fraction(self) -> float | None:
        total = self.total_heavy_atoms
        return None if total == 0 else self.parent_heavy_atoms / total


@dataclass(frozen=True)
class AtomPairGeometry:
    atom_index_1: int
    atom_index_2: int
    symbol_1: str
    symbol_2: str
    distance_angstrom: float
    covalent_radius_ratio: float
    atom_1_has_valence_headroom: bool
    atom_2_has_valence_headroom: bool

    @property
    def both_have_valence_headroom(self) -> bool:
        return self.atom_1_has_valence_headroom and self.atom_2_has_valence_headroom


@dataclass(frozen=True)
class InterfragmentAudit:
    classification: str
    minimum_distance_pair: AtomPairGeometry | None
    minimum_ratio_pair: AtomPairGeometry | None
    n_pairs_examined: int
    n_bond_distance_pairs: int
    n_bond_distance_pairs_with_headroom: int
    n_close_pairs: int
    bond_ratio_threshold: float
    close_ratio_threshold: float


def fragment_summary(mol: Chem.Mol) -> FragmentSummary:
    components = tuple(
        tuple(int(index) for index in component)
        for component in Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    )
    heavy_counts = tuple(
        sum(mol.GetAtomWithIdx(index).GetAtomicNum() > 1 for index in component)
        for component in components
    )
    heavy_component_ids = tuple(
        index for index, heavy_count in enumerate(heavy_counts) if heavy_count > 0
    )
    parent_component_id = (
        None
        if not heavy_component_ids
        else max(
            heavy_component_ids,
            key=lambda index: (heavy_counts[index], len(components[index]), -index),
        )
    )
    atom_to_component = {
        atom_index: component_id
        for component_id, component in enumerate(components)
        for atom_index in component
    }
    return FragmentSummary(
        atom_components=components,
        heavy_component_ids=heavy_component_ids,
        heavy_counts=heavy_counts,
        parent_component_id=parent_component_id,
        atom_to_component=atom_to_component,
    )


def component_molecule(mol: Chem.Mol, component_id: int) -> Chem.Mol:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return Chem.Mol(fragments[component_id])


def _atom_xyz(mol: Chem.Mol, atom_index: int) -> tuple[float, float, float]:
    point = mol.GetConformer().GetAtomPosition(atom_index)
    return float(point.x), float(point.y), float(point.z)


def _distance(first: Iterable[float], second: Iterable[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def valence_headroom(atom: Chem.Atom, required_order: int = 1) -> bool:
    """Return whether at least one standard valence permits another bond.

    The calculation deliberately follows the legacy audit's conservative
    integer-valence rule so that only the pair search, not the chemistry model,
    changes in the corrected analysis.
    """
    periodic_table = Chem.GetPeriodicTable()
    try:
        current_valence = float(atom.GetTotalValence())
    except Exception:
        current_valence = float(atom.GetExplicitValence())
    allowed = [
        int(value)
        for value in periodic_table.GetValenceList(atom.GetAtomicNum())
        if int(value) >= 0
    ]
    return any(value >= current_valence + required_order for value in allowed)


def _pair_geometry(mol: Chem.Mol, first_index: int, second_index: int) -> AtomPairGeometry:
    first_atom = mol.GetAtomWithIdx(first_index)
    second_atom = mol.GetAtomWithIdx(second_index)
    distance = _distance(_atom_xyz(mol, first_index), _atom_xyz(mol, second_index))
    periodic_table = Chem.GetPeriodicTable()
    covalent_sum = (
        periodic_table.GetRcovalent(first_atom.GetAtomicNum())
        + periodic_table.GetRcovalent(second_atom.GetAtomicNum())
    )
    ratio = distance / covalent_sum if covalent_sum > 0 else float("inf")
    return AtomPairGeometry(
        atom_index_1=first_index,
        atom_index_2=second_index,
        symbol_1=first_atom.GetSymbol(),
        symbol_2=second_atom.GetSymbol(),
        distance_angstrom=distance,
        covalent_radius_ratio=ratio,
        atom_1_has_valence_headroom=valence_headroom(first_atom),
        atom_2_has_valence_headroom=valence_headroom(second_atom),
    )


def audit_interfragment_geometry(
    mol: Chem.Mol,
    fragments: FragmentSummary | None = None,
    *,
    bond_ratio_threshold: float = 1.25,
    close_ratio_threshold: float = 1.75,
) -> InterfragmentAudit:
    """Classify a fragmented record after examining every intercomponent pair.

    Hierarchy:
    1. potential_missing_bond if any bond-distance pair has valence headroom;
    2. bond_distance_valence_limited if bond-distance pairs exist but none can bond;
    3. close_nonbonded if a close pair exists;
    4. geometrically_separated otherwise.

    This corrects the legacy implementation, which classified only from the
    single minimum normalized-distance pair.
    """
    fragments = fragment_summary(mol) if fragments is None else fragments
    if fragments.n_heavy_components <= 1:
        return InterfragmentAudit(
            "single_heavy_component", None, None, 0, 0, 0, 0,
            bond_ratio_threshold, close_ratio_threshold,
        )

    pairs: list[AtomPairGeometry] = []
    ids = fragments.heavy_component_ids
    for first_position, first_component_id in enumerate(ids):
        first_component = fragments.atom_components[first_component_id]
        for second_component_id in ids[first_position + 1:]:
            second_component = fragments.atom_components[second_component_id]
            for first_index in first_component:
                if mol.GetAtomWithIdx(first_index).GetAtomicNum() <= 1:
                    continue
                for second_index in second_component:
                    if mol.GetAtomWithIdx(second_index).GetAtomicNum() <= 1:
                        continue
                    pairs.append(_pair_geometry(mol, first_index, second_index))

    if not pairs:
        return InterfragmentAudit(
            "geometrically_separated", None, None, 0, 0, 0, 0,
            bond_ratio_threshold, close_ratio_threshold,
        )

    minimum_distance_pair = min(pairs, key=lambda pair: pair.distance_angstrom)
    minimum_ratio_pair = min(pairs, key=lambda pair: pair.covalent_radius_ratio)
    bond_pairs = [
        pair for pair in pairs if pair.covalent_radius_ratio <= bond_ratio_threshold
    ]
    headroom_pairs = [pair for pair in bond_pairs if pair.both_have_valence_headroom]
    close_pairs = [
        pair for pair in pairs if pair.covalent_radius_ratio <= close_ratio_threshold
    ]

    if headroom_pairs:
        classification = "potential_missing_bond"
    elif bond_pairs:
        classification = "bond_distance_valence_limited"
    elif close_pairs:
        classification = "close_nonbonded"
    else:
        classification = "geometrically_separated"

    return InterfragmentAudit(
        classification=classification,
        minimum_distance_pair=minimum_distance_pair,
        minimum_ratio_pair=minimum_ratio_pair,
        n_pairs_examined=len(pairs),
        n_bond_distance_pairs=len(bond_pairs),
        n_bond_distance_pairs_with_headroom=len(headroom_pairs),
        n_close_pairs=len(close_pairs),
        bond_ratio_threshold=bond_ratio_threshold,
        close_ratio_threshold=close_ratio_threshold,
    )
