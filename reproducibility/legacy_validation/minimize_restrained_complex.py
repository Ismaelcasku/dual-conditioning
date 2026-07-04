#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from rdkit import Chem

import openmm
from openmm import (
    Context,
    CustomExternalForce,
    LocalEnergyMinimizer,
    Platform,
    VerletIntegrator,
    unit,
)
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    NoCutoff,
    PDBFile,
)
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from pdbfixer import PDBFixer


def read_ligand(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=True,
    )

    for molecule in supplier:
        if molecule is not None:
            return molecule

    raise RuntimeError(f"No readable ligand in {path}")


def _positions_nm_array(positions) -> np.ndarray:
    """Convert OpenMM positions to a plain N×3 NumPy array in nm."""
    if hasattr(positions, "value_in_unit"):
        return np.asarray(
            positions.value_in_unit(unit.nanometer),
            dtype=float,
        )

    return np.asarray(positions, dtype=float)


def add_restraint_force(
    system: openmm.System,
    atom_indices: list[int],
    positions,
    force_constant: float,
    force_group: int,
) -> None:
    force = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )

    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    force.setForceGroup(force_group)

    positions_nm = _positions_nm_array(positions)

    for atom_index in atom_indices:
        x0, y0, z0 = positions_nm[atom_index]

        force.addParticle(
            int(atom_index),
            [
                float(force_constant),
                float(x0),
                float(y0),
                float(z0),
            ],
        )

    system.addForce(force)


def coordinates_nm(
    state,
    indices: list[int],
) -> np.ndarray:
    positions_nm = _positions_nm_array(
        state.getPositions(asNumpy=True)
    )

    return positions_nm[
        np.asarray(indices, dtype=int)
    ]

def rmsd_angstrom(
    coordinates_a: np.ndarray,
    coordinates_b: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (coordinates_a - coordinates_b) ** 2,
                    axis=1,
                )
            )
        )
        * 10.0
    )


def steric_proxy(
    protein_indices: list[int],
    ligand_indices: list[int],
    state,
) -> tuple[int, float]:
    positions_nm = _positions_nm_array(
        state.getPositions(asNumpy=True)
    )

    protein_xyz = positions_nm[
        np.asarray(protein_indices, dtype=int)
    ]

    ligand_xyz = positions_nm[
        np.asarray(ligand_indices, dtype=int)
    ]

    distances = np.linalg.norm(
        protein_xyz[:, None, :]
        - ligand_xyz[None, :, :],
        axis=2,
    )

    severe_contacts = int(
        np.sum(distances < 0.20)
    )

    minimum_distance = float(
        np.min(distances) * 10.0
    )

    return severe_contacts, minimum_distance


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--protein", required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--fixed-json", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument(
        "--protein-k",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--warhead-k",
        type=float,
        default=10000.0,
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3000,
    )

    args = parser.parse_args()

    protein_path = Path(args.protein)
    ligand_path = Path(args.ligand)
    fixed_json_path = Path(args.fixed_json)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    fixed_data = json.loads(
        fixed_json_path.read_text()
    )

    fixed_ligand_indices = [
        int(index)
        for index in fixed_data["fixed_candidate_indices"]
    ]

    expected_fixed_atoms = int(
        fixed_data["expected_fixed_atoms"]
    )

    if len(fixed_ligand_indices) != expected_fixed_atoms:
        raise RuntimeError(
            "Unexpected number of fixed ligand atoms: "
            f"detected={len(fixed_ligand_indices)}, "
            f"expected={expected_fixed_atoms}"
        )

    print("=== Protein preparation ===")

    fixer = PDBFixer(
        filename=str(protein_path)
    )

    fixer.removeHeterogens(
        keepWater=False
    )

    fixer.findNonstandardResidues()

    if fixer.nonstandardResidues:
        raise RuntimeError(
            "Nonstandard protein residues detected: "
            f"{fixer.nonstandardResidues}"
        )

    fixer.findMissingResidues()

    # No reconstruimos loops o residuos ausentes.
    fixer.missingResidues = {}

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    protein_topology = fixer.topology
    protein_positions = fixer.positions

    protein_atom_count = protein_topology.getNumAtoms()

    protein_heavy_indices = [
        atom.index
        for atom in protein_topology.atoms()
        if atom.element is not None
        and atom.element.symbol != "H"
    ]

    print(f"protein_atoms={protein_atom_count}")
    print(
        f"protein_heavy_atoms={len(protein_heavy_indices)}"
    )

    print("\n=== Ligand preparation ===")

    rdkit_molecule = read_ligand(
        ligand_path
    )

    rdkit_molecule = Chem.AddHs(
        rdkit_molecule,
        addCoords=True,
    )

    Chem.SanitizeMol(
        rdkit_molecule
    )

    off_molecule = Molecule.from_rdkit(
        rdkit_molecule,
        allow_undefined_stereo=True,
    )

    off_molecule.name = "LIG"
    off_molecule.generate_unique_atom_names()

    ligand_topology = (
        off_molecule
        .to_topology()
        .to_openmm()
    )

    ligand_positions = (
        off_molecule
        .conformers[0]
        .to_openmm()
    )

    ligand_atom_count = ligand_topology.getNumAtoms()

    print(f"ligand_atoms={ligand_atom_count}")
    print(
        f"fixed_ligand_indices={fixed_ligand_indices}"
    )

    modeller = Modeller(
        protein_topology,
        protein_positions,
    )

    ligand_offset = modeller.topology.getNumAtoms()

    modeller.add(
        ligand_topology,
        ligand_positions,
    )

    ligand_global_indices = list(
        range(
            ligand_offset,
            ligand_offset + ligand_atom_count,
        )
    )

    ligand_heavy_global_indices = [
        ligand_offset + atom.GetIdx()
        for atom in rdkit_molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    fixed_global_indices = [
        ligand_offset + index
        for index in fixed_ligand_indices
    ]

    prepared_path = (
        outdir / "complex_prepared.pdb"
    )

    with prepared_path.open("w") as handle:
        PDBFile.writeFile(
            modeller.topology,
            modeller.positions,
            handle,
            keepIds=True,
        )

    print("\n=== Force-field assignment ===")

    cache_path = (
        outdir / "gaff_template_cache.json"
    )

    gaff_generator = GAFFTemplateGenerator(
        molecules=off_molecule,
        forcefield="gaff-2.11",
        cache=str(cache_path),
    )

    forcefield = ForceField(
        "amber14-all.xml"
    )

    forcefield.registerTemplateGenerator(
        gaff_generator.generator
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=True,
    )

    add_restraint_force(
        system=system,
        atom_indices=protein_heavy_indices,
        positions=modeller.positions,
        force_constant=args.protein_k,
        force_group=30,
    )

    add_restraint_force(
        system=system,
        atom_indices=fixed_global_indices,
        positions=modeller.positions,
        force_constant=args.warhead_k,
        force_group=31,
    )

    integrator = VerletIntegrator(
        0.001 * unit.picoseconds
    )

    platform = Platform.getPlatformByName(
        "CPU"
    )

    context = Context(
        system,
        integrator,
        platform,
        {
            "Threads": os.environ.get(
                "SLURM_CPUS_PER_TASK",
                "4",
            )
        },
    )

    context.setPositions(
        modeller.positions
    )

    initial_state = context.getState(
        getEnergy=True,
        getPositions=True,
    )

    initial_energy = (
        initial_state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    initial_fixed_xyz = coordinates_nm(
        initial_state,
        fixed_global_indices,
    )

    initial_ligand_xyz = coordinates_nm(
        initial_state,
        ligand_heavy_global_indices,
    )

    initial_clashes, initial_min_distance = steric_proxy(
        protein_heavy_indices,
        ligand_heavy_global_indices,
        initial_state,
    )

    print("\n=== Minimization ===")
    print(
        f"initial_energy_kj_mol={initial_energy:.6f}"
    )
    print(
        f"initial_severe_contacts_lt2A={initial_clashes}"
    )
    print(
        f"initial_min_protein_ligand_distance_A="
        f"{initial_min_distance:.4f}"
    )

    LocalEnergyMinimizer.minimize(
        context,
        tolerance=10.0
        * unit.kilojoule_per_mole
        / unit.nanometer,
        maxIterations=args.max_iterations,
    )

    final_state = context.getState(
        getEnergy=True,
        getPositions=True,
    )

    final_energy = (
        final_state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if not math.isfinite(final_energy):
        raise RuntimeError(
            "Final complex energy is not finite."
        )

    final_fixed_xyz = coordinates_nm(
        final_state,
        fixed_global_indices,
    )

    final_ligand_xyz = coordinates_nm(
        final_state,
        ligand_heavy_global_indices,
    )

    final_clashes, final_min_distance = steric_proxy(
        protein_heavy_indices,
        ligand_heavy_global_indices,
        final_state,
    )

    warhead_rmsd = rmsd_angstrom(
        initial_fixed_xyz,
        final_fixed_xyz,
    )

    ligand_heavy_rmsd = rmsd_angstrom(
        initial_ligand_xyz,
        final_ligand_xyz,
    )

    minimized_path = (
        outdir / "complex_minimized.pdb"
    )

    with minimized_path.open("w") as handle:
        PDBFile.writeFile(
            modeller.topology,
            final_state.getPositions(),
            handle,
            keepIds=True,
        )

    results = {
        "protein": str(protein_path),
        "ligand": str(ligand_path),
        "fixed_ligand_indices": fixed_ligand_indices,
        "fixed_global_indices": fixed_global_indices,
        "protein_restraint_k_kj_mol_nm2": args.protein_k,
        "warhead_restraint_k_kj_mol_nm2": args.warhead_k,
        "initial_energy_kj_mol": initial_energy,
        "final_energy_kj_mol": final_energy,
        "energy_change_kj_mol": final_energy - initial_energy,
        "initial_severe_contacts_lt2A": initial_clashes,
        "final_severe_contacts_lt2A": final_clashes,
        "initial_min_protein_ligand_distance_A": initial_min_distance,
        "final_min_protein_ligand_distance_A": final_min_distance,
        "warhead_rmsd_A": warhead_rmsd,
        "ligand_heavy_rmsd_A": ligand_heavy_rmsd,
        "prepared_complex": str(prepared_path),
        "minimized_complex": str(minimized_path),
    }

    result_path = (
        outdir / "minimization_results.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print("\n=== Results ===")
    print(
        f"final_energy_kj_mol={final_energy:.6f}"
    )
    print(
        f"energy_change_kj_mol="
        f"{final_energy - initial_energy:.6f}"
    )
    print(
        f"final_severe_contacts_lt2A={final_clashes}"
    )
    print(
        f"final_min_protein_ligand_distance_A="
        f"{final_min_distance:.4f}"
    )
    print(
        f"warhead_rmsd_A={warhead_rmsd:.6f}"
    )
    print(
        f"ligand_heavy_rmsd_A={ligand_heavy_rmsd:.6f}"
    )
    print(f"results={result_path}")
    print("RESTRAINED_MINIMIZATION_STATUS=OK")


if __name__ == "__main__":
    main()
