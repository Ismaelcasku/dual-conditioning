#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdShapeHelpers


GRID_SPACING = 0.5
BITS_PER_POINT = DataStructs.DiscreteValueType.TWOBITVALUE
VDW_SCALE = 0.8
STEP_SIZE = 0.25
MAX_LAYERS = -1
IGNORE_HS = True
ALLOW_PROTRUDE_REORDERING = False


def resolve_path(
    value: str | Path,
    project: Path,
) -> Path:
    value = str(value)

    if value.startswith("/work/"):
        return (
            project
            / value.removeprefix("/work/")
        ).resolve()

    path = Path(value)

    if path.is_absolute():
        return path.resolve()

    return (project / path).resolve()


def load_molecule(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=True,
    )

    molecule = next(
        (
            mol
            for mol in supplier
            if mol is not None
        ),
        None,
    )

    if molecule is None:
        raise RuntimeError(
            f"No readable molecule in {path}"
        )

    if molecule.GetNumConformers() == 0:
        raise RuntimeError(
            f"No conformer in {path}"
        )

    return molecule


def find_reference(
    project: Path,
    ligand_id: str,
) -> Path:
    root = (
        project
        / "data/mpro/prepared/silvr_xchem_hits"
    )

    matches = sorted(
        root.glob(
            f"{ligand_id}__*/"
            "*_ligand.sdf"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one reference for "
            f"{ligand_id}; found {matches}"
        )

    return matches[0].resolve()


def shape_tanimoto_similarity(
    molecule: Chem.Mol,
    reference: Chem.Mol,
) -> float:
    distance = float(
        rdShapeHelpers.ShapeTanimotoDist(
            molecule,
            reference,
            confId1=-1,
            confId2=-1,
            gridSpacing=GRID_SPACING,
            bitsPerPoint=BITS_PER_POINT,
            vdwScale=VDW_SCALE,
            stepSize=STEP_SIZE,
            maxLayers=MAX_LAYERS,
            ignoreHs=IGNORE_HS,
        )
    )

    return 1.0 - distance


def shape_protrude_distance(
    molecule: Chem.Mol,
    reference: Chem.Mol,
) -> float:
    return float(
        rdShapeHelpers.ShapeProtrudeDist(
            molecule,
            reference,
            confId1=-1,
            confId2=-1,
            gridSpacing=GRID_SPACING,
            bitsPerPoint=BITS_PER_POINT,
            vdwScale=VDW_SCALE,
            stepSize=STEP_SIZE,
            maxLayers=MAX_LAYERS,
            ignoreHs=IGNORE_HS,
            allowReordering=ALLOW_PROTRUDE_REORDERING,
        )
    )


def mapped_rmsd(
    candidate: Chem.Mol,
    reference: Chem.Mol,
    candidate_indices: list[int],
    reference_indices: list[int],
) -> tuple[float, float]:
    if len(candidate_indices) != len(reference_indices):
        raise RuntimeError(
            "Candidate and reference index lists "
            "have different lengths."
        )

    candidate_conf = candidate.GetConformer()
    reference_conf = reference.GetConformer()

    squared_distances = []

    for candidate_index, reference_index in zip(
        candidate_indices,
        reference_indices,
    ):
        candidate_atom = candidate.GetAtomWithIdx(
            int(candidate_index)
        )

        reference_atom = reference.GetAtomWithIdx(
            int(reference_index)
        )

        if (
            candidate_atom.GetAtomicNum()
            != reference_atom.GetAtomicNum()
        ):
            raise RuntimeError(
                "Element mismatch in fixed-atom mapping: "
                f"candidate {candidate_index} "
                f"{candidate_atom.GetSymbol()} versus "
                f"reference {reference_index} "
                f"{reference_atom.GetSymbol()}"
            )

        candidate_position = np.asarray(
            candidate_conf.GetAtomPosition(
                int(candidate_index)
            ),
            dtype=float,
        )

        reference_position = np.asarray(
            reference_conf.GetAtomPosition(
                int(reference_index)
            ),
            dtype=float,
        )

        squared_distances.append(
            float(
                np.sum(
                    (
                        candidate_position
                        - reference_position
                    )
                    ** 2
                )
            )
        )

    rmsd = math.sqrt(
        float(
            np.mean(
                squared_distances
            )
        )
    )

    maximum_displacement = math.sqrt(
        max(
            squared_distances
        )
    )

    return rmsd, maximum_displacement


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--postroot",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    project = Path(
        args.project
    ).resolve()

    manifest_path = Path(
        args.manifest
    ).resolve()

    postroot = Path(
        args.postroot
    ).resolve()

    outdir = Path(
        args.outdir
    ).resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype=str,
    )

    pair_cache: dict[
        tuple[str, str],
        dict[str, float | str],
    ] = {}

    candidate_rows = []

    for row in manifest.to_dict(
        orient="records"
    ):
        candidate_id = row["candidate_id"]
        group = row["group"]

        match = re.match(
            r"^(x[0-9]+)_(x[0-9]+)_",
            candidate_id,
        )

        if match is None:
            raise RuntimeError(
                f"Invalid candidate ID: "
                f"{candidate_id}"
            )

        a_local = match.group(1)
        b_global = match.group(2)

        fixed_json_path = resolve_path(
            row["fixed_json"],
            project,
        )

        generated_path = resolve_path(
            row["ligand_sdf"],
            project,
        )

        prepared_path = (
            postroot
            / candidate_id
            / "prepared_ligand.sdf"
        )

        minimized_path = (
            postroot
            / candidate_id
            / "minimized_ligand.sdf"
        )

        for required in (
            fixed_json_path,
            generated_path,
            prepared_path,
            minimized_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    required
                )

        fixed = json.loads(
            fixed_json_path.read_text()
        )

        reference_a_path = resolve_path(
            fixed["reference_ligand"],
            project,
        )

        reference_b_path = find_reference(
            project,
            b_global,
        )

        reference_a = load_molecule(
            reference_a_path
        )

        reference_b = load_molecule(
            reference_b_path
        )

        generated = load_molecule(
            generated_path
        )

        prepared = load_molecule(
            prepared_path
        )

        minimized = load_molecule(
            minimized_path
        )

        pair_key = (
            a_local,
            b_global,
        )

        if pair_key not in pair_cache:
            pair_cache[pair_key] = {
                "pair_id":
                    f"{a_local}_{b_global}",
                "A_local":
                    a_local,
                "B_global":
                    b_global,
                "A_B_shape_tanimoto_similarity":
                    shape_tanimoto_similarity(
                        reference_a,
                        reference_b,
                    ),
                "A_to_B_shape_protrude_distance":
                    shape_protrude_distance(
                        reference_a,
                        reference_b,
                    ),
                "reference_A":
                    str(reference_a_path),
                "reference_B":
                    str(reference_b_path),
            }

        candidate_indices = [
            int(index)
            for index in fixed[
                "fixed_candidate_indices"
            ]
        ]

        reference_indices = [
            int(index)
            for index in fixed[
                "fixed_reference_indices"
            ]
        ]

        generated_rmsd, generated_max = (
            mapped_rmsd(
                generated,
                reference_a,
                candidate_indices,
                reference_indices,
            )
        )

        prepared_rmsd, prepared_max = (
            mapped_rmsd(
                prepared,
                reference_a,
                candidate_indices,
                reference_indices,
            )
        )

        minimized_rmsd, minimized_max = (
            mapped_rmsd(
                minimized,
                reference_a,
                candidate_indices,
                reference_indices,
            )
        )

        generated_similarity = (
            shape_tanimoto_similarity(
                generated,
                reference_b,
            )
        )

        prepared_similarity = (
            shape_tanimoto_similarity(
                prepared,
                reference_b,
            )
        )

        minimized_similarity = (
            shape_tanimoto_similarity(
                minimized,
                reference_b,
            )
        )

        generated_protrude = (
            shape_protrude_distance(
                generated,
                reference_b,
            )
        )

        prepared_protrude = (
            shape_protrude_distance(
                prepared,
                reference_b,
            )
        )

        minimized_protrude = (
            shape_protrude_distance(
                minimized,
                reference_b,
            )
        )

        pair_metrics = pair_cache[
            pair_key
        ]

        candidate_rows.append(
            {
                "candidate_id":
                    candidate_id,
                "group":
                    group,
                "pair_id":
                    pair_metrics["pair_id"],
                "A_local":
                    a_local,
                "B_global":
                    b_global,
                "A_B_shape_tanimoto_similarity":
                    pair_metrics[
                        "A_B_shape_tanimoto_similarity"
                    ],
                "A_to_B_shape_protrude_distance":
                    pair_metrics[
                        "A_to_B_shape_protrude_distance"
                    ],
                "generated_shape_similarity_B":
                    generated_similarity,
                "prepared_shape_similarity_B":
                    prepared_similarity,
                "minimized_shape_similarity_B":
                    minimized_similarity,
                "delta_shape_generated_to_minimized":
                    (
                        minimized_similarity
                        - generated_similarity
                    ),
                "delta_shape_prepared_to_minimized":
                    (
                        minimized_similarity
                        - prepared_similarity
                    ),
                "generated_protrude_B":
                    generated_protrude,
                "prepared_protrude_B":
                    prepared_protrude,
                "minimized_protrude_B":
                    minimized_protrude,
                "delta_protrude_generated_to_minimized":
                    (
                        minimized_protrude
                        - generated_protrude
                    ),
                "delta_protrude_prepared_to_minimized":
                    (
                        minimized_protrude
                        - prepared_protrude
                    ),
                "generated_warhead_rmsd_A":
                    generated_rmsd,
                "prepared_warhead_rmsd_A":
                    prepared_rmsd,
                "minimized_warhead_rmsd_A":
                    minimized_rmsd,
                "warhead_drift_generation_to_minimized":
                    (
                        minimized_rmsd
                        - generated_rmsd
                    ),
                "warhead_drift_prepared_to_minimized":
                    (
                        minimized_rmsd
                        - prepared_rmsd
                    ),
                "generated_warhead_max_displacement_A":
                    generated_max,
                "prepared_warhead_max_displacement_A":
                    prepared_max,
                "minimized_warhead_max_displacement_A":
                    minimized_max,
                "warhead_pass_generated_0p2A":
                    generated_rmsd <= 0.2,
                "warhead_pass_minimized_0p2A":
                    minimized_rmsd <= 0.2,
            }
        )

    pair_dataframe = pd.DataFrame(
        pair_cache.values()
    ).sort_values(
        "pair_id"
    )

    candidate_dataframe = pd.DataFrame(
        candidate_rows
    ).sort_values(
        [
            "pair_id",
            "candidate_id",
        ]
    )

    pair_output = (
        outdir
        / "pair_reference_shape.tsv"
    )

    candidate_output = (
        outdir
        / "candidate_shape_anchor.tsv"
    )

    pair_dataframe.to_csv(
        pair_output,
        sep="\t",
        index=False,
    )

    candidate_dataframe.to_csv(
        candidate_output,
        sep="\t",
        index=False,
    )

    configuration = {
        "shape_metric":
            "RDKit grid-based Shape Tanimoto",
        "shape_similarity_definition":
            "1 - ShapeTanimotoDist",
        "protrude_direction":
            "candidate_or_A relative to B",
        "grid_spacing_angstrom":
            GRID_SPACING,
        "bits_per_point":
            BITS_PER_POINT,
        "vdw_scale":
            VDW_SCALE,
        "step_size":
            STEP_SIZE,
        "max_layers":
            MAX_LAYERS,
        "ignore_hydrogens":
            IGNORE_HS,
        "allow_protrude_reordering":
            ALLOW_PROTRUDE_REORDERING,
        "alignment":
            "none; original crystallographic frame",
        "warhead_rmsd_alignment":
            "none; mapped absolute coordinates",
    }

    (
        outdir
        / "metric_configuration.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        )
    )

    print(
        "=== DIRECT REFERENCE A-B METRICS ==="
    )

    print(
        pair_dataframe[
            [
                "pair_id",
                "A_B_shape_tanimoto_similarity",
                "A_to_B_shape_protrude_distance",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=== CANDIDATE METRICS ==="
    )

    compact_columns = [
        "candidate_id",
        "pair_id",
        "A_B_shape_tanimoto_similarity",
        "generated_shape_similarity_B",
        "minimized_shape_similarity_B",
        "delta_shape_generated_to_minimized",
        "generated_protrude_B",
        "minimized_protrude_B",
        "delta_protrude_generated_to_minimized",
        "generated_warhead_rmsd_A",
        "minimized_warhead_rmsd_A",
        "warhead_drift_generation_to_minimized",
        "warhead_pass_minimized_0p2A",
    ]

    print(
        candidate_dataframe[
            compact_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"pair_output={pair_output}"
    )

    print(
        f"candidate_output={candidate_output}"
    )

    print(
        "HOMOGENEOUS_RECALCULATION_STATUS=OK"
    )


if __name__ == "__main__":
    main()
