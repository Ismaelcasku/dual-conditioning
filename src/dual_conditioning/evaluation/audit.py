"""Per-record component-aware audit used by the command-line scripts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from rdkit import Chem

from .atom_matching import match_fixed_atoms
from .connectivity import audit_interfragment_geometry, component_molecule, fragment_summary
from .shape import ShapeMetricConfig, compare_shape_to_references
from .strict_dual import evaluate_strict_dual


def _shape_fields(prefix: str, comparison: Any) -> dict[str, Any]:
    if comparison is None:
        return {
            f"{prefix}_shape_ok": False,
            f"{prefix}_tanimoto_distance_to_a": "",
            f"{prefix}_tanimoto_distance_to_b": "",
            f"{prefix}_tanimoto_similarity_to_a": "",
            f"{prefix}_tanimoto_similarity_to_b": "",
            f"{prefix}_protrude_distance_to_a": "",
            f"{prefix}_protrude_distance_to_b": "",
            f"{prefix}_b_closer_by_both_shape_metrics": False,
        }
    return {
        f"{prefix}_shape_ok": True,
        f"{prefix}_tanimoto_distance_to_a": comparison.to_a.tanimoto_distance,
        f"{prefix}_tanimoto_distance_to_b": comparison.to_b.tanimoto_distance,
        f"{prefix}_tanimoto_similarity_to_a": comparison.to_a.tanimoto_similarity,
        f"{prefix}_tanimoto_similarity_to_b": comparison.to_b.tanimoto_similarity,
        f"{prefix}_protrude_distance_to_a": comparison.to_a.protrude_distance,
        f"{prefix}_protrude_distance_to_b": comparison.to_b.protrude_distance,
        f"{prefix}_b_closer_by_both_shape_metrics": comparison.b_closer_by_both_metrics,
    }


def audit_record(
    molecule: Chem.Mol,
    reference_a: Chem.Mol,
    reference_b: Chem.Mol,
    fixed_reference_indices: Sequence[int],
    *,
    metric_config: ShapeMetricConfig | None = None,
    local_threshold_angstrom: float = 0.2,
    bond_ratio_threshold: float = 1.25,
    close_ratio_threshold: float = 1.75,
) -> dict[str, Any]:
    fragments = fragment_summary(molecule)
    fixed_match = match_fixed_atoms(reference_a, molecule, fixed_reference_indices)
    interfragment = audit_interfragment_geometry(
        molecule,
        fragments,
        bond_ratio_threshold=bond_ratio_threshold,
        close_ratio_threshold=close_ratio_threshold,
    )

    parent_molecule = (
        None
        if fragments.parent_component_id is None
        else component_molecule(molecule, fragments.parent_component_id)
    )
    matched_components = {
        fragments.atom_to_component[index]
        for index in fixed_match.generated_atom_indices
        if index in fragments.atom_to_component
    }
    anchor_component_id = (
        next(iter(matched_components))
        if len(matched_components) == 1 and len(fixed_match.generated_atom_indices) > 0
        else None
    )
    anchor_molecule = (
        None if anchor_component_id is None else component_molecule(molecule, anchor_component_id)
    )

    def compare(candidate: Chem.Mol | None):
        if candidate is None or candidate.GetNumConformers() == 0:
            return None
        try:
            return compare_shape_to_references(candidate, reference_a, reference_b, metric_config)
        except Exception:
            return None

    full_shape = compare(molecule)
    parent_shape = compare(parent_molecule)
    anchor_shape = compare(anchor_molecule)

    local_pass = fixed_match.all_within(local_threshold_angstrom)
    full_global_pass = bool(full_shape and full_shape.b_closer_by_both_metrics)
    parent_global_pass = bool(parent_shape and parent_shape.b_closer_by_both_metrics)
    anchor_global_pass = bool(anchor_shape and anchor_shape.b_closer_by_both_metrics)
    connected_strict = (
        evaluate_strict_dual(
            connected=fragments.connected,
            fixed_atom_match=fixed_match,
            shape_comparison=full_shape,
            local_threshold_angstrom=local_threshold_angstrom,
        )
        if full_shape is not None
        else None
    )

    row: dict[str, Any] = {
        "n_heavy_components": fragments.n_heavy_components,
        "connected": fragments.connected,
        "total_heavy_atoms": fragments.total_heavy_atoms,
        "parent_heavy_atoms": fragments.parent_heavy_atoms,
        "parent_heavy_fraction": fragments.parent_heavy_fraction,
        "anchor_component_id": "" if anchor_component_id is None else anchor_component_id,
        "all_fixed_atoms_in_one_component": len(matched_components) == 1,
        "fixed_assignment_ok": fixed_match.assignment_ok,
        "fixed_element_match_ok": fixed_match.element_match_ok,
        "fixed_atom_rmsd_angstrom": fixed_match.rmsd_angstrom,
        "fixed_atom_max_distance_angstrom": fixed_match.max_distance_angstrom,
        "fixed_atom_distances_angstrom": ",".join(
            f"{distance:.6f}" for distance in fixed_match.distances_angstrom
        ),
        "local_retention_all_atoms": local_pass,
        "interfragment_class": interfragment.classification,
        "n_interfragment_pairs_examined": interfragment.n_pairs_examined,
        "n_bond_distance_pairs": interfragment.n_bond_distance_pairs,
        "n_bond_distance_pairs_with_headroom": interfragment.n_bond_distance_pairs_with_headroom,
        "n_close_interfragment_pairs": interfragment.n_close_pairs,
        "minimum_interfragment_distance_angstrom": (
            "" if interfragment.minimum_distance_pair is None
            else interfragment.minimum_distance_pair.distance_angstrom
        ),
        "minimum_ratio_distance_angstrom": (
            "" if interfragment.minimum_ratio_pair is None
            else interfragment.minimum_ratio_pair.distance_angstrom
        ),
        "minimum_covalent_radius_ratio": (
            "" if interfragment.minimum_ratio_pair is None
            else interfragment.minimum_ratio_pair.covalent_radius_ratio
        ),
        "full_dual_geometry_with_local_retention": local_pass and full_global_pass,
        "parent_dual_geometry_with_local_retention": local_pass and parent_global_pass,
        "anchor_dual_geometry_with_local_retention": (
            local_pass and anchor_component_id is not None and anchor_global_pass
        ),
        "connected_strict_dual": bool(connected_strict and connected_strict.strict_dual),
    }
    row.update(_shape_fields("full", full_shape))
    row.update(_shape_fields("parent", parent_shape))
    row.update(_shape_fields("anchor", anchor_shape))
    if full_shape is not None and parent_shape is not None:
        row["shape_similarity_inflation"] = (
            full_shape.to_b.tanimoto_similarity - parent_shape.to_b.tanimoto_similarity
        )
    else:
        row["shape_similarity_inflation"] = ""
    return row
