"""Explicit RDKit shape-distance and similarity calculations."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import rdShapeHelpers


@dataclass(frozen=True)
class ShapeMetricConfig:
    grid_spacing: float = 0.5
    bits_per_point: DataStructs.DiscreteValueType = DataStructs.DiscreteValueType.TWOBITVALUE
    vdw_scale: float = 0.8
    step_size: float = 0.25
    max_layers: int = -1
    ignore_hydrogens: bool = True
    allow_protrude_reordering: bool = True


@dataclass(frozen=True)
class ShapeMetrics:
    tanimoto_distance: float
    tanimoto_similarity: float
    protrude_distance: float


@dataclass(frozen=True)
class ReferenceShapeComparison:
    to_a: ShapeMetrics
    to_b: ShapeMetrics

    @property
    def b_closer_tanimoto(self) -> bool:
        return self.to_b.tanimoto_distance < self.to_a.tanimoto_distance

    @property
    def b_closer_protrude(self) -> bool:
        return self.to_b.protrude_distance < self.to_a.protrude_distance

    @property
    def b_closer_by_both_metrics(self) -> bool:
        return self.b_closer_tanimoto and self.b_closer_protrude


def shape_metrics(
    molecule: Chem.Mol,
    reference: Chem.Mol,
    config: ShapeMetricConfig | None = None,
) -> ShapeMetrics:
    config = ShapeMetricConfig() if config is None else config
    tanimoto_distance = float(
        rdShapeHelpers.ShapeTanimotoDist(
            molecule,
            reference,
            confId1=-1,
            confId2=-1,
            gridSpacing=config.grid_spacing,
            bitsPerPoint=config.bits_per_point,
            vdwScale=config.vdw_scale,
            stepSize=config.step_size,
            maxLayers=config.max_layers,
            ignoreHs=config.ignore_hydrogens,
        )
    )
    protrude_distance = float(
        rdShapeHelpers.ShapeProtrudeDist(
            molecule,
            reference,
            confId1=-1,
            confId2=-1,
            gridSpacing=config.grid_spacing,
            bitsPerPoint=config.bits_per_point,
            vdwScale=config.vdw_scale,
            stepSize=config.step_size,
            maxLayers=config.max_layers,
            ignoreHs=config.ignore_hydrogens,
            allowReordering=config.allow_protrude_reordering,
        )
    )
    return ShapeMetrics(
        tanimoto_distance=tanimoto_distance,
        tanimoto_similarity=1.0 - tanimoto_distance,
        protrude_distance=protrude_distance,
    )


def compare_shape_to_references(
    molecule: Chem.Mol,
    reference_a: Chem.Mol,
    reference_b: Chem.Mol,
    config: ShapeMetricConfig | None = None,
) -> ReferenceShapeComparison:
    return ReferenceShapeComparison(
        to_a=shape_metrics(molecule, reference_a, config),
        to_b=shape_metrics(molecule, reference_b, config),
    )
