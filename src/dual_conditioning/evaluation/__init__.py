"""Connectivity-, anchor-, and shape-aware molecular evaluation."""

from .atom_matching import FixedAtomMatch, match_fixed_atoms
from .connectivity import FragmentSummary, InterfragmentAudit, audit_interfragment_geometry, fragment_summary
from .shape import ShapeMetricConfig, ShapeMetrics, compare_shape_to_references, shape_metrics
from .strict_dual import StrictDualResult, evaluate_strict_dual

__all__ = [
    "FixedAtomMatch",
    "match_fixed_atoms",
    "FragmentSummary",
    "InterfragmentAudit",
    "audit_interfragment_geometry",
    "fragment_summary",
    "ShapeMetricConfig",
    "ShapeMetrics",
    "shape_metrics",
    "compare_shape_to_references",
    "StrictDualResult",
    "evaluate_strict_dual",
]
