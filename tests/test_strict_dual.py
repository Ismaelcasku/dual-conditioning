from __future__ import annotations

from dual_conditioning.evaluation.atom_matching import FixedAtomMatch
from dual_conditioning.evaluation.shape import ReferenceShapeComparison, ShapeMetrics
from dual_conditioning.evaluation.strict_dual import evaluate_strict_dual


def match(distance=0.1):
    return FixedAtomMatch(True, True, (0,), (0,), (distance,), distance, distance, distance)


def comparison(b_wins=True):
    if b_wins:
        return ReferenceShapeComparison(
            ShapeMetrics(0.5, 0.5, 0.5),
            ShapeMetrics(0.3, 0.7, 0.3),
        )
    return ReferenceShapeComparison(
        ShapeMetrics(0.3, 0.7, 0.3),
        ShapeMetrics(0.5, 0.5, 0.5),
    )


def test_all_three_conditions_are_required():
    assert evaluate_strict_dual(connected=True, fixed_atom_match=match(), shape_comparison=comparison()).strict_dual
    assert not evaluate_strict_dual(connected=False, fixed_atom_match=match(), shape_comparison=comparison()).strict_dual
    assert not evaluate_strict_dual(connected=True, fixed_atom_match=match(0.3), shape_comparison=comparison()).strict_dual
    assert not evaluate_strict_dual(connected=True, fixed_atom_match=match(), shape_comparison=comparison(False)).strict_dual
