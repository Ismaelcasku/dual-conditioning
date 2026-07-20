#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from posecheck import PoseCheck
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers
from rdkit.Geometry import Point3D


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Ignoring unrecognized record",
)


def load_molecule(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=True,
    )

    molecule = next(
        (
            molecule
            for molecule in supplier
            if molecule is not None
        ),
        None,
    )

    if molecule is None:
        raise RuntimeError(
            f"No readable molecule in {path}"
        )

    molecule = Chem.RemoveHs(
        molecule,
        sanitize=True,
    )

    if molecule.GetNumConformers() == 0:
        raise RuntimeError(
            f"No conformer in {path}"
        )

    return molecule


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


def locate_complex(
    result_json: Path,
    result_data: dict[str, Any],
    project: Path,
    state: str,
) -> Path:
    if state == "prepared":
        keys = (
            "prepared_complex",
            "prepared_complex_pdb",
            "initial_complex",
            "initial_complex_pdb",
        )

        patterns = (
            "*prepared*complex*.pdb",
            "*prepared*.pdb",
            "*initial*complex*.pdb",
        )

    elif state == "minimized":
        keys = (
            "minimized_complex",
            "minimized_complex_pdb",
            "final_complex",
            "final_complex_pdb",
        )

        patterns = (
            "*minimized*complex*.pdb",
            "*minimized*.pdb",
            "*final*complex*.pdb",
        )

    else:
        raise ValueError(
            f"Unsupported state: {state}"
        )

    for key in keys:
        value = result_data.get(key)

        if not value:
            continue

        candidate = resolve_path(
            value,
            project,
        )

        if candidate.is_file():
            return candidate

    matches: list[Path] = []

    for pattern in patterns:
        matches.extend(
            result_json.parent.glob(pattern)
        )

    matches = sorted(
        {
            path.resolve()
            for path in matches
            if path.is_file()
        }
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not identify {state} complex. "
            f"Matches: {matches}"
        )

    return matches[0]


def protein_atom_count(
    result_data: dict[str, Any],
) -> int:
    local_indices = result_data.get(
        "fixed_ligand_indices"
    )

    global_indices = result_data.get(
        "fixed_global_indices"
    )

    if not local_indices or not global_indices:
        raise RuntimeError(
            "Missing fixed ligand/global indices "
            "in minimization_results.json"
        )

    if len(local_indices) != len(global_indices):
        raise RuntimeError(
            "Inconsistent fixed-index arrays."
        )

    offsets = {
        int(global_index) - int(local_index)
        for global_index, local_index in zip(
            global_indices,
            local_indices,
        )
    }

    if len(offsets) != 1:
        raise RuntimeError(
            f"Inconsistent protein offsets: {offsets}"
        )

    return offsets.pop()


def pdb_element(line: str) -> str:
    element = line[76:78].strip()

    if element:
        return element.title()

    name = line[12:16].strip().upper()

    if name.startswith("CL"):
        return "Cl"

    if name.startswith("BR"):
        return "Br"

    letters = re.sub(
        r"[^A-Z]",
        "",
        name,
    )

    return letters[:1].title()


def read_pdb_atoms(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open() as handle:
        for line in handle:
            if not line.startswith(
                ("ATOM  ", "HETATM")
            ):
                continue

            records.append(
                {
                    "line": line,
                    "element": pdb_element(line),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            )

    if not records:
        raise RuntimeError(
            f"No atoms found in {path}"
        )

    return records


def split_complex(
    complex_path: Path,
    source_ligand: Path,
    result_data: dict[str, Any],
    protein_output: Path,
    ligand_output: Path,
) -> Chem.Mol:
    molecule = load_molecule(
        source_ligand
    )

    records = read_pdb_atoms(
        complex_path
    )

    offset = protein_atom_count(
        result_data
    )

    if offset <= 0 or offset >= len(records):
        raise RuntimeError(
            "Invalid protein/ligand partition: "
            f"offset={offset}, atoms={len(records)}"
        )

    protein_records = records[:offset]
    ligand_records = records[offset:]

    ligand_heavy_records = [
        record
        for record in ligand_records
        if record["element"] not in ("H", "D")
    ]

    if (
        len(ligand_heavy_records)
        != molecule.GetNumAtoms()
    ):
        raise RuntimeError(
            "Ligand heavy-atom count mismatch: "
            f"PDB={len(ligand_heavy_records)}, "
            f"SDF={molecule.GetNumAtoms()}"
        )

    expected_elements = [
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    ]

    observed_elements = [
        record["element"]
        for record in ligand_heavy_records
    ]

    if expected_elements != observed_elements:
        raise RuntimeError(
            "Ligand atom-order mismatch.\n"
            f"Expected: {expected_elements}\n"
            f"Observed: {observed_elements}"
        )

    conformer = molecule.GetConformer()

    for index, record in enumerate(
        ligand_heavy_records
    ):
        conformer.SetAtomPosition(
            index,
            Point3D(
                record["x"],
                record["y"],
                record["z"],
            ),
        )

    with protein_output.open("w") as handle:
        for record in protein_records:
            if record["element"] in ("H", "D"):
                continue

            handle.write(record["line"])

        handle.write("TER\n")
        handle.write("END\n")

    writer = Chem.SDWriter(
        str(ligand_output)
    )

    writer.write(molecule)
    writer.close()

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


def shape_metrics(
    molecule: Chem.Mol,
    reference: Chem.Mol,
) -> dict[str, float]:
    tanimoto_distance = float(
        rdShapeHelpers.ShapeTanimotoDist(
            molecule,
            reference,
            ignoreHs=True,
        )
    )

    protrude_distance = float(
        rdShapeHelpers.ShapeProtrudeDist(
            molecule,
            reference,
            ignoreHs=True,
        )
    )

    return {
        "tanimoto_distance":
            tanimoto_distance,
        "tanimoto_similarity":
            1.0 - tanimoto_distance,
        "protrude_distance":
            protrude_distance,
    }


def active_interactions(
    dataframe: pd.DataFrame | None,
) -> set[str]:
    if dataframe is None or dataframe.empty:
        return set()

    result: set[str] = set()

    for column, value in dataframe.iloc[0].items():
        try:
            if pd.isna(value) or not bool(value):
                continue
        except Exception:
            continue

        if isinstance(column, tuple):
            name = "|".join(
                str(part)
                for part in column
                if str(part) not in (
                    "",
                    "nan",
                    "None",
                )
            )
        else:
            name = str(column)

        result.add(name)

    return result


def calculate_posecheck(
    protein_path: Path,
    ligand_path: Path,
    reduce_path: Path,
    interactions_csv: Path,
) -> dict[str, Any]:
    posecheck = PoseCheck(
        reduce_path=str(reduce_path),
    )

    posecheck.load_protein_from_pdb(
        str(protein_path)
    )

    posecheck.load_ligands_from_sdf(
        str(ligand_path)
    )

    clashes = int(
        posecheck.calculate_clashes()[0]
    )

    strain = math.nan
    strain_error = ""

    try:
        strain = float(
            posecheck.calculate_strain_energy()[0]
        )
    except Exception as exception:
        strain_error = (
            f"{type(exception).__name__}: "
            f"{exception}"
        )

    interactions_dataframe = None
    interaction_error = ""

    try:
        interactions_dataframe = (
            posecheck.calculate_interactions()
        )

        interactions_dataframe.to_csv(
            interactions_csv
        )

    except Exception as exception:
        interaction_error = (
            f"{type(exception).__name__}: "
            f"{exception}"
        )

    return {
        "clashes":
            clashes,
        "strain":
            strain,
        "strain_error":
            strain_error,
        "interaction_error":
            interaction_error,
        "interactions":
            active_interactions(
                interactions_dataframe
            ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True,
    )

    parser.add_argument(
        "--candidate-id",
        required=True,
    )

    parser.add_argument(
        "--group",
        required=True,
    )

    parser.add_argument(
        "--result-json",
        required=True,
    )

    parser.add_argument(
        "--ligand-sdf",
        required=True,
    )

    parser.add_argument(
        "--fixed-json",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--reduce",
        required=True,
    )

    args = parser.parse_args()

    project = Path(
        args.project
    ).resolve()

    result_json = Path(
        args.result_json
    ).resolve()

    ligand_sdf = Path(
        args.ligand_sdf
    ).resolve()

    fixed_json = Path(
        args.fixed_json
    ).resolve()

    output = Path(
        args.output
    ).resolve()

    reduce_path = Path(
        args.reduce
    ).resolve()

    for required in (
        result_json,
        ligand_sdf,
        fixed_json,
        reduce_path,
    ):
        if not required.exists():
            raise FileNotFoundError(
                required
            )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_data = json.loads(
        result_json.read_text()
    )

    fixed_data = json.loads(
        fixed_json.read_text()
    )

    match = re.match(
        r"^(x[0-9]+)_(x[0-9]+)_",
        args.candidate_id,
    )

    if match is None:
        raise RuntimeError(
            f"Invalid candidate ID: "
            f"{args.candidate_id}"
        )

    a_local = match.group(1)
    b_global = match.group(2)

    if fixed_data.get("A_local") != a_local:
        raise RuntimeError(
            "A_local mismatch between ID "
            "and fixed-atom JSON."
        )

    prepared_complex = locate_complex(
        result_json,
        result_data,
        project,
        "prepared",
    )

    minimized_complex = locate_complex(
        result_json,
        result_data,
        project,
        "minimized",
    )

    prepared_protein = (
        output
        / "prepared_protein_noH.pdb"
    )

    minimized_protein = (
        output
        / "minimized_protein_noH.pdb"
    )

    prepared_ligand = (
        output
        / "prepared_ligand.sdf"
    )

    minimized_ligand = (
        output
        / "minimized_ligand.sdf"
    )

    before_molecule = split_complex(
        prepared_complex,
        ligand_sdf,
        result_data,
        prepared_protein,
        prepared_ligand,
    )

    after_molecule = split_complex(
        minimized_complex,
        ligand_sdf,
        result_data,
        minimized_protein,
        minimized_ligand,
    )

    reference_b_path = find_reference(
        project,
        b_global,
    )

    reference_b = load_molecule(
        reference_b_path
    )

    before_shape = shape_metrics(
        before_molecule,
        reference_b,
    )

    after_shape = shape_metrics(
        after_molecule,
        reference_b,
    )

    before_posecheck = calculate_posecheck(
        prepared_protein,
        prepared_ligand,
        reduce_path,
        output
        / "prepared_interactions.csv",
    )

    after_posecheck = calculate_posecheck(
        minimized_protein,
        minimized_ligand,
        reduce_path,
        output
        / "minimized_interactions.csv",
    )

    before_interactions = (
        before_posecheck["interactions"]
    )

    after_interactions = (
        after_posecheck["interactions"]
    )

    retained = (
        before_interactions
        & after_interactions
    )

    lost = (
        before_interactions
        - after_interactions
    )

    gained = (
        after_interactions
        - before_interactions
    )

    union = (
        before_interactions
        | after_interactions
    )

    jaccard = (
        len(retained) / len(union)
        if union
        else 1.0
    )

    warhead_rmsd = float(
        result_data.get(
            "warhead_rmsd_A",
            math.nan,
        )
    )

    ligand_rmsd = float(
        result_data.get(
            "ligand_heavy_rmsd_A",
            math.nan,
        )
    )

    summary = {
        "candidate_id":
            args.candidate_id,
        "group":
            args.group,
        "A_local":
            a_local,
        "B_global":
            b_global,
        "warhead_rmsd_A":
            warhead_rmsd,
        "warhead_pass_0p2A":
            warhead_rmsd <= 0.2,
        "ligand_heavy_rmsd_A":
            ligand_rmsd,
        "before_clashes":
            before_posecheck["clashes"],
        "after_clashes":
            after_posecheck["clashes"],
        "delta_clashes":
            (
                after_posecheck["clashes"]
                - before_posecheck["clashes"]
            ),
        "before_strain":
            before_posecheck["strain"],
        "after_strain":
            after_posecheck["strain"],
        "delta_strain":
            (
                after_posecheck["strain"]
                - before_posecheck["strain"]
            ),
        "before_shape_tanimoto_similarity_B":
            before_shape[
                "tanimoto_similarity"
            ],
        "after_shape_tanimoto_similarity_B":
            after_shape[
                "tanimoto_similarity"
            ],
        "delta_shape_tanimoto_similarity_B":
            (
                after_shape[
                    "tanimoto_similarity"
                ]
                - before_shape[
                    "tanimoto_similarity"
                ]
            ),
        "before_shape_tanimoto_distance_B":
            before_shape[
                "tanimoto_distance"
            ],
        "after_shape_tanimoto_distance_B":
            after_shape[
                "tanimoto_distance"
            ],
        "before_shape_protrude_distance_B":
            before_shape[
                "protrude_distance"
            ],
        "after_shape_protrude_distance_B":
            after_shape[
                "protrude_distance"
            ],
        "delta_shape_protrude_distance_B":
            (
                after_shape[
                    "protrude_distance"
                ]
                - before_shape[
                    "protrude_distance"
                ]
            ),
        "before_interaction_count":
            len(before_interactions),
        "after_interaction_count":
            len(after_interactions),
        "retained_interaction_count":
            len(retained),
        "lost_interaction_count":
            len(lost),
        "gained_interaction_count":
            len(gained),
        "interaction_jaccard":
            jaccard,
        "before_strain_error":
            before_posecheck[
                "strain_error"
            ],
        "after_strain_error":
            after_posecheck[
                "strain_error"
            ],
        "before_interaction_error":
            before_posecheck[
                "interaction_error"
            ],
        "after_interaction_error":
            after_posecheck[
                "interaction_error"
            ],
        "prepared_complex":
            str(prepared_complex),
        "minimized_complex":
            str(minimized_complex),
        "reference_B":
            str(reference_b_path),
    }

    details = {
        "summary":
            summary,
        "prepared_interactions":
            sorted(before_interactions),
        "minimized_interactions":
            sorted(after_interactions),
        "retained_interactions":
            sorted(retained),
        "lost_interactions":
            sorted(lost),
        "gained_interactions":
            sorted(gained),
    }

    with (
        output
        / "validation_results.json"
    ).open("w") as handle:
        json.dump(
            details,
            handle,
            indent=2,
            allow_nan=True,
        )

    summary_path = (
        output
        / "summary.tsv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary),
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerow(summary)

    print(
        f"candidate_id={args.candidate_id}"
    )

    print(
        f"before_clashes="
        f"{before_posecheck['clashes']}"
    )

    print(
        f"after_clashes="
        f"{after_posecheck['clashes']}"
    )

    print(
        f"before_strain="
        f"{before_posecheck['strain']}"
    )

    print(
        f"after_strain="
        f"{after_posecheck['strain']}"
    )

    print(
        "before_shape_tanimoto_similarity_B="
        f"{before_shape['tanimoto_similarity']:.6f}"
    )

    print(
        "after_shape_tanimoto_similarity_B="
        f"{after_shape['tanimoto_similarity']:.6f}"
    )

    print(
        "delta_shape_tanimoto_similarity_B="
        f"{summary['delta_shape_tanimoto_similarity_B']:.6f}"
    )

    print(
        "before_shape_protrude_distance_B="
        f"{before_shape['protrude_distance']:.6f}"
    )

    print(
        "after_shape_protrude_distance_B="
        f"{after_shape['protrude_distance']:.6f}"
    )

    print(
        "delta_shape_protrude_distance_B="
        f"{summary['delta_shape_protrude_distance_B']:.6f}"
    )

    print(
        f"summary={summary_path}"
    )

    print(
        "POSTMIN_VALIDATION_STATUS=OK"
    )


if __name__ == "__main__":
    main()
