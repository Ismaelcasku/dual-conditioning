#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from posecheck import PoseCheck


PROTEINS = {
    "5R83_x0434": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
        "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_protein.pdb"
    ),
    "5REZ_x0874": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
        "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_protein.pdb"
    ),
}

REFERENCE_LIGANDS = {
    "5R83_x0434": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0434__5R83_Mpro-x0434_AAR-POS-d2a4d1df-11/"
        "5R83_Mpro-x0434_AAR-POS-d2a4d1df-11_ligand.sdf"
    ),
    "5REZ_x0874": Path(
        "data/mpro/prepared/silvr_xchem_hits/"
        "x0874__5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14/"
        "5REZ_Mpro-x0874_AAR-POS-d2a4d1df-14_ligand.sdf"
    ),
}

COMMON_INTERACTION_TYPES = [
    "VdWContact",
    "Hydrophobic",
    "HBAcceptor",
    "HBDonor",
    "PiStacking",
    "PiCation",
    "CationPi",
    "Cationic",
    "Anionic",
    "SaltBridge",
    "XBAcceptor",
    "XBDonor",
    "MetalAcceptor",
    "MetalDonor",
]


def read_tsv(path):
    with Path(path).open() as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


def write_tsv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:
        fields = list(rows[0].keys())

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def first_scalar(value):
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        value = value.to_numpy().ravel()[0]

    elif isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]

    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]

    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    if isinstance(value, np.generic):
        value = value.item()

    try:
        return float(value)
    except Exception:
        return None


def parse_interactions(table):
    interactions = []

    if table is None:
        return interactions

    if not isinstance(table, pd.DataFrame):
        return interactions

    if table.empty:
        return interactions

    row = table.iloc[0]

    for column, value in row.items():
        try:
            active = bool(value)
        except Exception:
            active = False

        if not active:
            continue

        if isinstance(column, tuple):
            parts = [str(part) for part in column]
        else:
            parts = [str(column)]

        if len(parts) >= 3:
            ligand_id = parts[0]
            residue = parts[1]
            interaction_type = parts[2]
        else:
            ligand_id = ""
            residue = ""
            interaction_type = str(column)

        interactions.append({
            "ligand_id": ligand_id,
            "residue": residue,
            "interaction_type": interaction_type,
        })

    return interactions


def interaction_signature(interactions, include_vdw=True):
    signature = set()

    for interaction in interactions:
        interaction_type = interaction[
            "interaction_type"
        ]

        if (
            not include_vdw
            and interaction_type == "VdWContact"
        ):
            continue

        signature.add((
            interaction["residue"],
            interaction_type,
        ))

    return signature


def residue_signature(interactions):
    return {
        interaction["residue"]
        for interaction in interactions
    }


def run_posecheck(protein, ligand):
    result = {
        "load_status": "NOT_RUN",
        "clashes_status": "NOT_RUN",
        "strain_status": "NOT_RUN",
        "interactions_status": "NOT_RUN",
        "clashes": None,
        "strain_energy": None,
        "interactions": [],
        "error_messages": [],
    }

    pc = PoseCheck()

    try:
        pc.load_protein_from_pdb(str(protein))
        pc.load_ligands_from_sdf(str(ligand))
        result["load_status"] = "OK"

    except Exception as error:
        result["load_status"] = "FAILED"
        result["error_messages"].append(
            "load:{}:{}".format(
                type(error).__name__,
                str(error),
            )
        )
        return result

    try:
        result["clashes"] = first_scalar(
            pc.calculate_clashes()
        )
        result["clashes_status"] = (
            "OK"
            if result["clashes"] is not None
            else "EMPTY"
        )

    except Exception as error:
        result["clashes_status"] = "FAILED"
        result["error_messages"].append(
            "clashes:{}:{}".format(
                type(error).__name__,
                str(error),
            )
        )

    try:
        result["strain_energy"] = first_scalar(
            pc.calculate_strain_energy()
        )
        result["strain_status"] = (
            "OK"
            if result["strain_energy"] is not None
            else "EMPTY"
        )

    except Exception as error:
        result["strain_status"] = "FAILED"
        result["error_messages"].append(
            "strain:{}:{}".format(
                type(error).__name__,
                str(error),
            )
        )

    try:
        table = pc.calculate_interactions()
        result["interactions"] = parse_interactions(
            table
        )
        result["interactions_status"] = "OK"

    except Exception as error:
        result["interactions_status"] = "FAILED"
        result["error_messages"].append(
            "interactions:{}:{}".format(
                type(error).__name__,
                str(error),
            )
        )

    return result


def mean_or_blank(values):
    clean = [
        float(value)
        for value in values
        if value not in ("", None)
        and math.isfinite(float(value))
    ]

    if not clean:
        return ""

    return statistics.mean(clean)


def median_or_blank(values):
    clean = [
        float(value)
        for value in values
        if value not in ("", None)
        and math.isfinite(float(value))
    ]

    if not clean:
        return ""

    return statistics.median(clean)


def fraction_or_blank(rows, predicate):
    if not rows:
        return ""

    return sum(
        bool(predicate(row))
        for row in rows
    ) / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", required=True)
    parser.add_argument("--labelled", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    physical_rows = read_tsv(args.physical)
    labelled_rows = read_tsv(args.labelled)
    out_dir = Path(args.out_dir)

    if len(physical_rows) != 79:
        raise RuntimeError(
            f"Expected 79 physical candidates, found "
            f"{len(physical_rows)}"
        )

    if len(labelled_rows) != 105:
        raise RuntimeError(
            f"Expected 105 labelled entries, found "
            f"{len(labelled_rows)}"
        )

    ###########################################################################
    # Reference calibration
    ###########################################################################

    reference_results = {}
    reference_rows = []
    interactions_long = []

    for receptor_key in sorted(PROTEINS):
        result = run_posecheck(
            PROTEINS[receptor_key],
            REFERENCE_LIGANDS[receptor_key],
        )

        all_signature = interaction_signature(
            result["interactions"],
            include_vdw=True,
        )

        specific_signature = interaction_signature(
            result["interactions"],
            include_vdw=False,
        )

        residues = residue_signature(
            result["interactions"]
        )

        counts = Counter(
            interaction["interaction_type"]
            for interaction in result["interactions"]
        )

        reference_row = {
            "receptor_key": receptor_key,
            "protein_path":
                str(PROTEINS[receptor_key]),
            "reference_ligand_path":
                str(REFERENCE_LIGANDS[receptor_key]),
            "load_status":
                result["load_status"],
            "clashes_status":
                result["clashes_status"],
            "strain_status":
                result["strain_status"],
            "interactions_status":
                result["interactions_status"],
            "posecheck_clashes":
                result["clashes"],
            "posecheck_strain_energy":
                result["strain_energy"],
            "n_interactions_all":
                len(all_signature),
            "n_interactions_specific":
                len(specific_signature),
            "n_contacted_residues":
                len(residues),
            "n_hydrogen_bonds":
                counts["HBAcceptor"]
                + counts["HBDonor"],
            "residues": ",".join(
                sorted(residues)
            ),
            "specific_interactions": ",".join(
                sorted(
                    f"{residue}:{interaction_type}"
                    for residue, interaction_type
                    in specific_signature
                )
            ),
            "error_messages":
                " | ".join(
                    result["error_messages"]
                ),
        }

        for interaction_type in COMMON_INTERACTION_TYPES:
            reference_row[
                f"n_{interaction_type}"
            ] = counts[interaction_type]

        reference_rows.append(reference_row)

        reference_results[receptor_key] = {
            "metrics": reference_row,
            "all_signature": all_signature,
            "specific_signature":
                specific_signature,
            "residues": residues,
        }

        for interaction in result["interactions"]:
            interactions_long.append({
                "record_type": "reference",
                "compatibility_id": receptor_key,
                "receptor_key": receptor_key,
                "pair_id": "",
                "lambda_global": "",
                "seed": "",
                "sample": "",
                "residue":
                    interaction["residue"],
                "interaction_type":
                    interaction[
                        "interaction_type"
                    ],
            })

    ###########################################################################
    # Physical candidates
    ###########################################################################

    posecheck_rows = []

    for number, source_row in enumerate(
        physical_rows,
        start=1,
    ):
        compatibility_id = source_row[
            "compatibility_id"
        ]

        receptor_key = source_row[
            "receptor_key"
        ]

        ligand_path = Path(
            source_row["candidate_sdf"]
        )

        result = run_posecheck(
            PROTEINS[receptor_key],
            ligand_path,
        )

        all_signature = interaction_signature(
            result["interactions"],
            include_vdw=True,
        )

        specific_signature = interaction_signature(
            result["interactions"],
            include_vdw=False,
        )

        residues = residue_signature(
            result["interactions"]
        )

        counts = Counter(
            interaction["interaction_type"]
            for interaction in result["interactions"]
        )

        reference = reference_results[
            receptor_key
        ]

        reference_all = reference[
            "all_signature"
        ]

        reference_specific = reference[
            "specific_signature"
        ]

        reference_residues = reference[
            "residues"
        ]

        exact_all_recovery = (
            len(all_signature & reference_all)
            / len(reference_all)
            if reference_all else ""
        )

        exact_specific_recovery = (
            len(
                specific_signature
                & reference_specific
            )
            / len(reference_specific)
            if reference_specific else ""
        )

        residue_recovery = (
            len(residues & reference_residues)
            / len(reference_residues)
            if reference_residues else ""
        )

        novel_specific = (
            specific_signature
            - reference_specific
        )

        reference_clashes = reference[
            "metrics"
        ]["posecheck_clashes"]

        reference_strain = reference[
            "metrics"
        ]["posecheck_strain_energy"]

        clashes_excess = (
            result["clashes"]
            - float(reference_clashes)
            if (
                result["clashes"] is not None
                and reference_clashes not in (
                    "",
                    None,
                )
            )
            else ""
        )

        strain_excess = (
            result["strain_energy"]
            - float(reference_strain)
            if (
                result["strain_energy"] is not None
                and reference_strain not in (
                    "",
                    None,
                )
            )
            else ""
        )

        strain_ratio = (
            result["strain_energy"]
            / float(reference_strain)
            if (
                result["strain_energy"] is not None
                and reference_strain not in (
                    "",
                    None,
                    0,
                )
            )
            else ""
        )

        output_row = dict(source_row)

        output_row.update({
            "posecheck_load_status":
                result["load_status"],
            "posecheck_clashes_status":
                result["clashes_status"],
            "posecheck_strain_status":
                result["strain_status"],
            "posecheck_interactions_status":
                result[
                    "interactions_status"
                ],
            "posecheck_clashes":
                result["clashes"],
            "reference_clashes":
                reference_clashes,
            "clashes_excess_vs_reference":
                clashes_excess,
            "posecheck_strain_energy":
                result["strain_energy"],
            "reference_strain_energy":
                reference_strain,
            "strain_excess_vs_reference":
                strain_excess,
            "strain_ratio_vs_reference":
                strain_ratio,
            "n_interactions_all":
                len(all_signature),
            "n_interactions_specific":
                len(specific_signature),
            "n_contacted_residues":
                len(residues),
            "n_hydrogen_bonds":
                counts["HBAcceptor"]
                + counts["HBDonor"],
            "reference_interaction_recovery_all":
                exact_all_recovery,
            "reference_interaction_recovery_specific":
                exact_specific_recovery,
            "reference_residue_recovery":
                residue_recovery,
            "n_novel_specific_interactions":
                len(novel_specific),
            "contacted_residues":
                ",".join(
                    sorted(residues)
                ),
            "specific_interactions":
                ",".join(
                    sorted(
                        f"{residue}:"
                        f"{interaction_type}"
                        for residue, interaction_type
                        in specific_signature
                    )
                ),
            "novel_specific_interactions":
                ",".join(
                    sorted(
                        f"{residue}:"
                        f"{interaction_type}"
                        for residue, interaction_type
                        in novel_specific
                    )
                ),
            "posecheck_error_messages":
                " | ".join(
                    result["error_messages"]
                ),
        })

        for interaction_type in COMMON_INTERACTION_TYPES:
            output_row[
                f"n_{interaction_type}"
            ] = counts[interaction_type]

        posecheck_rows.append(output_row)

        for interaction in result["interactions"]:
            interactions_long.append({
                "record_type": "candidate",
                "compatibility_id":
                    compatibility_id,
                "receptor_key": receptor_key,
                "pair_id":
                    source_row["pair_labels"],
                "lambda_global":
                    source_row["lambda_global"],
                "seed": source_row["seed"],
                "sample": source_row["sample"],
                "residue":
                    interaction["residue"],
                "interaction_type":
                    interaction[
                        "interaction_type"
                    ],
            })

        print(
            "[{}/79] {} clashes={} strain={} "
            "specific={} recovery={}".format(
                number,
                compatibility_id,
                result["clashes"],
                result["strain_energy"],
                len(specific_signature),
                (
                    f"{exact_specific_recovery:.3f}"
                    if exact_specific_recovery
                    not in ("", None)
                    else "NA"
                ),
            ),
            flush=True,
        )

    physical_output = (
        out_dir
        / "posecheck_physical_per_candidate.tsv"
    )

    write_tsv(
        physical_output,
        posecheck_rows,
    )

    write_tsv(
        out_dir / "posecheck_interactions_long.tsv",
        interactions_long,
    )

    write_tsv(
        out_dir / "posecheck_reference_calibration.tsv",
        reference_rows,
    )

    ###########################################################################
    # Expand physical metrics back to the 105 experimental labels
    ###########################################################################

    posecheck_index = {
        row["compatibility_id"]: row
        for row in posecheck_rows
    }

    expanded_rows = []

    for labelled in labelled_rows:
        physical = posecheck_index[
            labelled["compatibility_id"]
        ]

        expanded = dict(labelled)

        for key, value in physical.items():
            if key not in expanded:
                expanded[key] = value

        expanded_rows.append(expanded)

    write_tsv(
        out_dir / "posecheck_pair_labelled.tsv",
        expanded_rows,
    )

    ###########################################################################
    # Summary by pair and lambda
    ###########################################################################

    groups = defaultdict(list)

    for row in expanded_rows:
        groups[
            (
                row["pair_id"],
                row["lambda_global"],
            )
        ].append(row)

    summaries = []

    for key in sorted(
        groups,
        key=lambda item: (
            item[0],
            float(item[1]),
        ),
    ):
        rows = groups[key]

        successful = [
            row for row in rows
            if (
                row[
                    "posecheck_clashes_status"
                ] == "OK"
                and row[
                    "posecheck_interactions_status"
                ] == "OK"
            )
        ]

        summaries.append({
            "pair_id": key[0],
            "lambda_global": key[1],
            "n_candidates": len(rows),
            "n_posecheck_successful":
                len(successful),
            "mean_clashes": mean_or_blank([
                row["posecheck_clashes"]
                for row in successful
            ]),
            "median_clashes":
                median_or_blank([
                    row["posecheck_clashes"]
                    for row in successful
                ]),
            "mean_clashes_excess_vs_reference":
                mean_or_blank([
                    row[
                        "clashes_excess_vs_reference"
                    ]
                    for row in successful
                ]),
            "mean_strain_energy":
                mean_or_blank([
                    row[
                        "posecheck_strain_energy"
                    ]
                    for row in successful
                ]),
            "median_strain_energy":
                median_or_blank([
                    row[
                        "posecheck_strain_energy"
                    ]
                    for row in successful
                ]),
            "mean_strain_ratio_vs_reference":
                mean_or_blank([
                    row[
                        "strain_ratio_vs_reference"
                    ]
                    for row in successful
                ]),
            "mean_interactions_all":
                mean_or_blank([
                    row["n_interactions_all"]
                    for row in successful
                ]),
            "mean_interactions_specific":
                mean_or_blank([
                    row[
                        "n_interactions_specific"
                    ]
                    for row in successful
                ]),
            "mean_hydrogen_bonds":
                mean_or_blank([
                    row["n_hydrogen_bonds"]
                    for row in successful
                ]),
            "mean_reference_interaction_recovery_all":
                mean_or_blank([
                    row[
                        "reference_interaction_recovery_all"
                    ]
                    for row in successful
                ]),
            "mean_reference_interaction_recovery_specific":
                mean_or_blank([
                    row[
                        "reference_interaction_recovery_specific"
                    ]
                    for row in successful
                ]),
            "mean_reference_residue_recovery":
                mean_or_blank([
                    row[
                        "reference_residue_recovery"
                    ]
                    for row in successful
                ]),
            "mean_novel_specific_interactions":
                mean_or_blank([
                    row[
                        "n_novel_specific_interactions"
                    ]
                    for row in successful
                ]),
            "fraction_clashes_at_or_below_reference":
                fraction_or_blank(
                    successful,
                    lambda row: (
                        float(
                            row[
                                "clashes_excess_vs_reference"
                            ]
                        ) <= 0
                    ),
                ),
            "fraction_clashes_within_reference_plus_2":
                fraction_or_blank(
                    successful,
                    lambda row: (
                        float(
                            row[
                                "clashes_excess_vs_reference"
                            ]
                        ) <= 2
                    ),
                ),
            "fraction_strain_at_or_below_2x_reference":
                fraction_or_blank(
                    successful,
                    lambda row: (
                        float(
                            row[
                                "strain_ratio_vs_reference"
                            ]
                        ) <= 2
                    ),
                ),
        })

    write_tsv(
        out_dir / "posecheck_summary_by_pair_lambda.tsv",
        summaries,
    )

    ###########################################################################
    # Metadata
    ###########################################################################

    metadata = {
        "schema": "exp06_posecheck_raw_v1",
        "posecheck_version": "1.3.1",
        "n_physical_candidates":
            len(posecheck_rows),
        "n_pair_labelled_entries":
            len(expanded_rows),
        "pose_type": "raw generated parent pose",
        "protein_reference": (
            "A-local crystallographic receptor: "
            "5R83 for x0434 and 5REZ for x0874."
        ),
        "minimization": False,
        "docking": False,
        "interpretation": (
            "PoseCheck metrics are interpreted relative "
            "to the corresponding crystallographic "
            "reference processed by the same software."
        ),
        "specific_interactions_exclude":
            ["VdWContact"],
    }

    (
        out_dir / "posecheck_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    ###########################################################################
    # Console report
    ###########################################################################

    print()
    print("REFERENCE CALIBRATION")
    print(
        "receptor\tclashes\tstrain\tall\tspecific\t"
        "hbond\tresidues"
    )

    for row in reference_rows:
        print(
            "{}\t{}\t{:.4f}\t{}\t{}\t{}\t{}".format(
                row["receptor_key"],
                row["posecheck_clashes"],
                float(
                    row[
                        "posecheck_strain_energy"
                    ]
                ),
                row["n_interactions_all"],
                row[
                    "n_interactions_specific"
                ],
                row["n_hydrogen_bonds"],
                row["n_contacted_residues"],
            )
        )

    print()
    print("POSECHECK SUMMARY BY PAIR AND LAMBDA")
    print(
        "pair\tlambda\tn\tOK\tmeanClashes\t"
        "medianClashes\tmeanStrain\tstrainRatio\t"
        "specific\tHBond\tspecificRecovery\t"
        "residueRecovery\tnovelSpecific"
    )

    def fmt(value):
        if value in ("", None):
            return "NA"
        return f"{float(value):.3f}"

    for row in summaries:
        print(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
            "{}\t{}\t{}\t{}\t{}".format(
                row["pair_id"],
                row["lambda_global"],
                row["n_candidates"],
                row[
                    "n_posecheck_successful"
                ],
                fmt(row["mean_clashes"]),
                fmt(row["median_clashes"]),
                fmt(row["mean_strain_energy"]),
                fmt(
                    row[
                        "mean_strain_ratio_vs_reference"
                    ]
                ),
                fmt(
                    row[
                        "mean_interactions_specific"
                    ]
                ),
                fmt(
                    row[
                        "mean_hydrogen_bonds"
                    ]
                ),
                fmt(
                    row[
                        "mean_reference_interaction_recovery_specific"
                    ]
                ),
                fmt(
                    row[
                        "mean_reference_residue_recovery"
                    ]
                ),
                fmt(
                    row[
                        "mean_novel_specific_interactions"
                    ]
                ),
            )
        )

    print()
    print(f"physical_candidates={len(posecheck_rows)}")
    print(f"pair_labelled_entries={len(expanded_rows)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
