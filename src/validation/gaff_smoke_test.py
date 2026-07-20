#!/usr/bin/env python

import argparse
import csv
import math
import os
import traceback
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

import openmm
from openmm import Context, LocalEnergyMinimizer, Platform, VerletIntegrator, unit
from openmm.app import ForceField, NoCutoff, PDBFile
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator


def read_molecule(path):
    suffix = path.suffix.lower()

    if suffix == ".sdf":
        molecules = [
            mol for mol in Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=True
            )
            if mol is not None
        ]
        return molecules[0] if molecules else None

    if suffix == ".mol2":
        return Chem.MolFromMol2File(
            str(path), removeHs=False, sanitize=True
        )

    if suffix == ".mol":
        return Chem.MolFromMolFile(
            str(path), removeHs=False, sanitize=True
        )

    return None


def run_smoke(label, source, outdir):
    mol = read_molecule(source)

    if mol is None:
        raise RuntimeError("RDKit could not read the molecule.")

    Chem.SanitizeMol(mol)

    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 20260623

        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("Conformer generation failed.")
    else:
        mol = Chem.AddHs(mol, addCoords=True)

    offmol = Molecule.from_rdkit(
        mol,
        allow_undefined_stereo=True,
    )
    offmol.name = label
    offmol.generate_unique_atom_names()

    cache = outdir / "cache" / f"{label}_gaff.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    generator = GAFFTemplateGenerator(
        molecules=offmol,
        forcefield="gaff-2.11",
        cache=str(cache),
    )

    forcefield = ForceField()
    forcefield.registerTemplateGenerator(
        generator.generator
    )

    topology = offmol.to_topology().to_openmm()
    positions = offmol.conformers[0].to_openmm()

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
    )

    integrator = VerletIntegrator(
        0.001 * unit.picoseconds
    )

    platform = Platform.getPlatformByName("CPU")

    context = Context(
        system,
        integrator,
        platform,
        {"Threads": "4"},
    )
    context.setPositions(positions)

    initial = context.getState(
        getEnergy=True,
        getPositions=True,
    )

    initial_energy = initial.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

    LocalEnergyMinimizer.minimize(
        context,
        tolerance=10.0,
        maxIterations=200,
    )

    final = context.getState(
        getEnergy=True,
        getPositions=True,
    )

    final_energy = final.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

    if not math.isfinite(initial_energy):
        raise RuntimeError("Initial energy is not finite.")

    if not math.isfinite(final_energy):
        raise RuntimeError("Final energy is not finite.")

    pdb_path = outdir / "minimized" / f"{label}.pdb"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    with pdb_path.open("w") as handle:
        PDBFile.writeFile(
            topology,
            final.getPositions(),
            handle,
            keepIds=True,
        )

    return {
        "label": label,
        "source": str(source),
        "status": "PASS",
        "atoms": mol.GetNumAtoms(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "initial_energy_kj_mol": initial_energy,
        "final_energy_kj_mol": final_energy,
        "energy_change_kj_mol": final_energy - initial_energy,
        "minimized_pdb": str(pdb_path),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))

    results = []

    for entry in entries:
        label = entry["label"]
        source = Path(entry["path"])

        print(f"\n=== {label} ===", flush=True)
        print(f"source={source}", flush=True)

        try:
            result = run_smoke(
                label,
                source,
                outdir,
            )
        except Exception as exc:
            error_dir = outdir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)

            (error_dir / f"{label}.txt").write_text(
                traceback.format_exc()
            )

            result = {
                "label": label,
                "source": str(source),
                "status": "FAIL",
                "atoms": "",
                "heavy_atoms": "",
                "formal_charge": "",
                "initial_energy_kj_mol": "",
                "final_energy_kj_mol": "",
                "energy_change_kj_mol": "",
                "minimized_pdb": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

        results.append(result)
        print(f"status={result['status']}", flush=True)

        if result["error"]:
            print(f"error={result['error']}", flush=True)
        else:
            print(
                f"initial_energy_kj_mol="
                f"{result['initial_energy_kj_mol']:.6f}"
            )
            print(
                f"final_energy_kj_mol="
                f"{result['final_energy_kj_mol']:.6f}"
            )

    result_path = outdir / "gaff_smoke_results.csv"

    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    failures = [
        row["label"]
        for row in results
        if row["status"] != "PASS"
    ]

    print(f"\nresults={result_path}")

    if failures:
        print("failed=" + ",".join(failures))
        print("GAFF_SMOKE_STATUS=FAILED")
        raise SystemExit(1)

    print("GAFF_SMOKE_STATUS=OK")


if __name__ == "__main__":
    main()
