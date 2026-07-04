"""Single authoritative definition of dual-conditioning success."""

from __future__ import annotations

from dataclasses import dataclass

from .atom_matching import FixedAtomMatch
from .shape import ReferenceShapeComparison


@dataclass(frozen=True)
class StrictDualResult:
    connected: bool
    local_retention: bool
    global_shape: bool
    strict_dual: bool


def evaluate_strict_dual(
    *,
    connected: bool,
    fixed_atom_match: FixedAtomMatch,
    shape_comparison: ReferenceShapeComparison,
    local_threshold_angstrom: float = 0.2,
) -> StrictDualResult:
    local_retention = fixed_atom_match.all_within(local_threshold_angstrom)
    global_shape = shape_comparison.b_closer_by_both_metrics
    return StrictDualResult(
        connected=bool(connected),
        local_retention=local_retention,
        global_shape=global_shape,
        strict_dual=bool(connected and local_retention and global_shape),
    )
