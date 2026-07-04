"""Extract the representative fragmented record and its components."""

from pathlib import Path
import csv

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]

INPUT_SDF = (
    ROOT
    / "results/source_data/fragment_example"
    / "exp06_x0434_x2193_seed_4404_lambda_50.0_n10.sdf"
)

OUTPUT_DIR = (
    ROOT
    / "results/source_data/fragment_example"
    / "sample_2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_SAMPLE = 2
EXPECTED_HEAVY_FRAGMENTS = 4
EXPECTED_PARENT_FRACTION = 0.466667
TOLERANCE = 1e-5


def heavy_atom_count(mol):
    return sum(
        atom.GetAtomicNum() > 1
        for atom in mol.GetAtoms()
    )


def fragment_statistics(mol):
    atom_mappings = []

    fragments = Chem.GetMolFrags(
        mol,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=atom_mappings,
    )

    records = []

    for original_id, (fragment, atom_indices) in enumerate(
        zip(fragments, atom_mappings)
    ):
        heavy_atoms = heavy_atom_count(fragment)

        records.append(
            {
                "original_fragment_id": original_id,
                "mol": fragment,
                "atom_indices": tuple(atom_indices),
                "n_atoms": fragment.GetNumAtoms(),
                "n_heavy_atoms": heavy_atoms,
            }
        )

    heavy_records = [
        record
        for record in records
        if record["n_heavy_atoms"] > 0
    ]

    heavy_records.sort(
        key=lambda record: (
            record["n_heavy_atoms"],
            record["n_atoms"],
        ),
        reverse=True,
    )

    total_heavy = heavy_atom_count(mol)

    parent_heavy = (
        heavy_records[0]["n_heavy_atoms"]
        if heavy_records
        else 0
    )

    parent_fraction = (
        parent_heavy / total_heavy
        if total_heavy
        else float("nan")
    )

    return {
        "all_fragments": records,
        "heavy_fragments": heavy_records,
        "n_heavy_full": total_heavy,
        "n_heavy_fragments": len(heavy_records),
        "parent_heavy_atoms": parent_heavy,
        "parent_heavy_fraction": parent_fraction,
    }


supplier = Chem.SDMolSupplier(
    str(INPUT_SDF),
    sanitize=False,
    removeHs=False,
    strictParsing=False,
)

molecules = [
    mol
    for mol in supplier
]

print(f"N_RECORDS={len(molecules)}")
print()
print(
    "index\tname\theavy_atoms\theavy_fragments\t"
    "parent_heavy_fraction"
)

evaluated = []

for index, mol in enumerate(molecules):
    if mol is None:
        print(f"{index}\tINVALID")
        continue

    stats = fragment_statistics(mol)

    name = (
        mol.GetProp("_Name")
        if mol.HasProp("_Name")
        else ""
    )

    evaluated.append(
        {
            "index": index,
            "mol": mol,
            "stats": stats,
        }
    )

    print(
        f"{index}\t{name}\t"
        f"{stats['n_heavy_full']}\t"
        f"{stats['n_heavy_fragments']}\t"
        f"{stats['parent_heavy_fraction']:.6f}"
    )


def matches_expected(record):
    stats = record["stats"]

    return (
        stats["n_heavy_fragments"]
        == EXPECTED_HEAVY_FRAGMENTS
        and abs(
            stats["parent_heavy_fraction"]
            - EXPECTED_PARENT_FRACTION
        )
        <= TOLERANCE
    )


indexed_candidate = next(
    (
        record
        for record in evaluated
        if record["index"] == EXPECTED_SAMPLE
    ),
    None,
)

if (
    indexed_candidate is not None
    and matches_expected(indexed_candidate)
):
    selected = indexed_candidate
    selection_method = "sample_index_and_audit_match"
else:
    matching_records = [
        record
        for record in evaluated
        if matches_expected(record)
    ]

    if len(matching_records) != 1:
        raise RuntimeError(
            "Could not identify one unique record matching "
            "the audited fragmentation statistics. "
            f"Matches found: {len(matching_records)}"
        )

    selected = matching_records[0]
    selection_method = "audit_statistics_match"


selected_index = selected["index"]
selected_mol = selected["mol"]
selected_stats = selected["stats"]
heavy_fragments = selected_stats["heavy_fragments"]

print()
print(f"SELECTED_RECORD_INDEX={selected_index}")
print(f"SELECTION_METHOD={selection_method}")
print(
    "SELECTED_HEAVY_FRAGMENTS="
    f"{selected_stats['n_heavy_fragments']}"
)
print(
    "SELECTED_PARENT_HEAVY_FRACTION="
    f"{selected_stats['parent_heavy_fraction']:.6f}"
)


# Complete disconnected record
complete_path = OUTPUT_DIR / "fragmented_full_record.sdf"

writer = Chem.SDWriter(str(complete_path))
writer.write(selected_mol)
writer.close()


# Export components ordered from largest to smallest
summary_rows = []

for rank, record in enumerate(
    heavy_fragments,
    start=1,
):
    role = (
        "parent"
        if rank == 1
        else "secondary"
    )

    fragment = record["mol"]

    fragment.SetProp(
        "_Name",
        f"sample2_{role}_fragment_{rank}",
    )

    fragment.SetIntProp(
        "fragment_rank",
        rank,
    )

    fragment.SetProp(
        "fragment_role",
        role,
    )

    fragment.SetIntProp(
        "n_heavy_atoms",
        record["n_heavy_atoms"],
    )

    output_path = (
        OUTPUT_DIR
        / f"fragment_{rank:02d}_{role}.sdf"
    )

    writer = Chem.SDWriter(str(output_path))
    writer.write(fragment)
    writer.close()

    summary_rows.append(
        {
            "fragment_rank": rank,
            "fragment_role": role,
            "original_fragment_id": (
                record["original_fragment_id"]
            ),
            "n_atoms": record["n_atoms"],
            "n_heavy_atoms": record["n_heavy_atoms"],
            "heavy_atom_fraction": (
                record["n_heavy_atoms"]
                / selected_stats["n_heavy_full"]
            ),
            "original_atom_indices": ",".join(
                str(index)
                for index in record["atom_indices"]
            ),
            "sdf_path": str(output_path),
        }
    )


# Parent-only file
parent_path = OUTPUT_DIR / "parent_component.sdf"

writer = Chem.SDWriter(str(parent_path))
writer.write(heavy_fragments[0]["mol"])
writer.close()


# All secondary fragments in one multi-record SDF
secondary_path = OUTPUT_DIR / "secondary_components.sdf"

writer = Chem.SDWriter(str(secondary_path))

for record in heavy_fragments[1:]:
    writer.write(record["mol"])

writer.close()


summary_path = OUTPUT_DIR / "fragment_summary.tsv"

with summary_path.open(
    "w",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=summary_rows[0].keys(),
        delimiter="\t",
    )

    writer.writeheader()
    writer.writerows(summary_rows)


print()
print("=== EXPORTED COMPONENTS ===")

for row in summary_rows:
    print(
        f"rank={row['fragment_rank']} "
        f"role={row['fragment_role']} "
        f"heavy_atoms={row['n_heavy_atoms']} "
        f"fraction={row['heavy_atom_fraction']:.3f}"
    )

print()
print(f"FULL_RECORD={complete_path}")
print(f"PARENT_COMPONENT={parent_path}")
print(f"SECONDARY_COMPONENTS={secondary_path}")
print(f"SUMMARY={summary_path}")
print("FRAGMENTATION_EXAMPLE_EXTRACTION_STATUS=OK")
