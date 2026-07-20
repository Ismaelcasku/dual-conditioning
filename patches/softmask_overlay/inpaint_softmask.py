#!/usr/bin/env python3
"""Order-safe DiffSBDD inpainting with centered stage-wise soft masking.

This additive entry point leaves the original ``inpaint.py`` available. The
soft-mask diffusion method is installed at runtime from
``conditional_model_softmask.py``. Before RDKit reconstruction,
the generated point cloud is canonicalized as [parent in parent order] +
[new atoms] by type-constrained Hungarian assignment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from openbabel import openbabel
from rdkit import Chem
from torch_scatter import scatter_mean

openbabel.obErrorLog.StopLogging()

from equivariant_diffusion.conditional_model_softmask import install_softmask_patch

install_softmask_patch()

import utils
from analysis.molecule_builder import build_molecule, process_molecule
from constants import FLOAT_TYPE
from lightning_modules import LigandPocketDDPM

from softmask_atom_order import (
    AtomOrderError,
    atom_type_indices_to_atomic_numbers,
    canonicalize_point_cloud,
    canonicalize_rdkit_mol,
)


def prepare_from_sdf_files(sdf_files, atom_encoder):
    ligand_coords = []
    atom_one_hot = []
    for file in sdf_files:
        supplier = Chem.SDMolSupplier(str(file), sanitize=False, removeHs=False)
        rdmol = supplier[0] if len(supplier) else None
        if rdmol is None or rdmol.GetNumConformers() == 0:
            raise RuntimeError(f"Could not read fixed scaffold SDF: {file}")
        ligand_coords.append(
            torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        )
        types = torch.tensor(
            [atom_encoder[a.GetSymbol()] for a in rdmol.GetAtoms()], dtype=torch.long
        )
        atom_one_hot.append(F.one_hot(types, num_classes=len(atom_encoder)))
    return torch.cat(ligand_coords, dim=0), torch.cat(atom_one_hot, dim=0)


def prepare_ligand_from_pdb(biopython_atoms, atom_encoder):
    coord = torch.tensor(
        np.array([a.get_coord() for a in biopython_atoms]), dtype=FLOAT_TYPE
    )
    types = torch.tensor(
        [atom_encoder[a.element.capitalize()] for a in biopython_atoms],
        dtype=torch.long,
    )
    one_hot = F.one_hot(types, num_classes=len(atom_encoder))
    return coord, one_hot


def prepare_substructure(ref_ligand, fix_atoms, pdb_model, atom_encoder):
    """Read the fixed scaffold while preserving the requested atom order."""

    if fix_atoms[0].endswith(".sdf"):
        return prepare_from_sdf_files(fix_atoms, atom_encoder)

    chain, resi = ref_ligand.split(":")
    ligand = utils.get_residue_with_resi(pdb_model[chain], int(resi))

    atoms_by_name = {}
    duplicate_names = set()
    for atom in ligand.get_atoms():
        name = atom.get_name()
        if name in atoms_by_name:
            duplicate_names.add(name)
        atoms_by_name[name] = atom

    requested_duplicates = sorted(
        name for name in duplicate_names if name in set(fix_atoms)
    )
    if requested_duplicates:
        raise RuntimeError(
            f"Duplicate requested PDB atom names: {requested_duplicates}"
        )

    missing = [name for name in fix_atoms if name not in atoms_by_name]
    if missing:
        raise RuntimeError(f"Missing fixed PDB atoms: {missing}")

    fixed_atoms = [atoms_by_name[name] for name in fix_atoms]
    return prepare_ligand_from_pdb(fixed_atoms, atom_encoder)


def build_masks(ligand_mask, n_samples, n_fixed, hard_fixed_count):
    if hard_fixed_count < 1:
        raise ValueError("hard_fixed_count must be >= 1")
    if hard_fixed_count > n_fixed:
        raise ValueError(
            f"hard_fixed_count={hard_fixed_count} exceeds n_fixed={n_fixed}"
        )

    lig_fixed = torch.zeros_like(ligand_mask)
    lig_hard = torch.zeros_like(ligand_mask)
    for sample_idx in range(n_samples):
        sel = ligand_mask == sample_idx
        fixed_local = lig_fixed[sel]
        hard_local = lig_hard[sel]
        fixed_local[:n_fixed] = 1
        hard_local[:hard_fixed_count] = 1
        lig_fixed[sel] = fixed_local
        lig_hard[sel] = hard_local
    return lig_fixed, lig_hard


def _write_softmask_debug(
    path,
    x,
    atom_type,
    lig_mask_cpu,
    x_fixed,
    one_hot_fixed,
    n_fixed,
    hard_fixed_count,
    rho,
):
    if not path:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped_x = utils.batch_to_list(x, lig_mask_cpu)
    grouped_types = utils.batch_to_list(atom_type, lig_mask_cpu)
    reference = x_fixed.detach().cpu()
    reference_types = one_hot_fixed.argmax(1).detach().cpu()

    rows = []
    for sample_idx, (sample_x, sample_types) in enumerate(
        zip(grouped_x, grouped_types), start=1
    ):
        parent = sample_x[:n_fixed]
        hard = parent[:hard_fixed_count]
        soft = parent[hard_fixed_count:n_fixed]
        ref_hard = reference[:hard_fixed_count]
        ref_soft = reference[hard_fixed_count:n_fixed]

        def rmsd(first, second):
            if len(first) == 0:
                return 0.0
            return float(torch.sqrt(((first - second) ** 2).sum(1).mean()).item())

        type_match = bool(
            torch.equal(sample_types[:n_fixed].cpu(), reference_types[:n_fixed])
        )
        rows.append(
            {
                "sample": sample_idx,
                "rho": float(rho),
                "n_nodes": int(len(sample_x)),
                "n_fixed": int(n_fixed),
                "n_hard": int(hard_fixed_count),
                "hard_rmsd": rmsd(hard, ref_hard),
                "soft_rmsd": rmsd(soft, ref_soft),
                "parent_rmsd": rmsd(parent, reference),
                "fixed_type_match": type_match,
            }
        )

    path.write_text(json.dumps(rows, indent=2))


def inpaint_ligand(
    model,
    pdb_file,
    n_samples,
    ligand,
    fix_atoms,
    add_n_nodes=None,
    center="ligand",
    sanitize=False,
    largest_frag=False,
    relax_iter=0,
    timesteps=None,
    resamplings=1,
    save_traj=False,
    b_cloud_raw=None,
    lambda_global=0.0,
    guide_r=1.7,
    guide_clip=1.0,
    soft_scaffold_rho=1.0,
    hard_fixed_count=7,
    softmask_debug=None,
):
    if save_traj and n_samples > 1:
        raise NotImplementedError("Can only visualize trajectory with n_samples=1")
    if not 0.0 <= float(soft_scaffold_rho) <= 1.0:
        raise ValueError("soft_scaffold_rho must be in [0,1]")

    frames = timesteps if save_traj else 1
    sanitize = False if save_traj else sanitize
    relax_iter = 0 if save_traj else relax_iter
    largest_frag = False if save_traj else largest_frag

    pdb_model = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
    residues = utils.get_pocket_from_ligand(pdb_model, ligand)
    pocket = model.prepare_pocket(residues, repeats=n_samples)

    x_fixed, one_hot_fixed = prepare_substructure(
        ligand, fix_atoms, pdb_model, model.lig_type_encoder
    )
    n_fixed = len(x_fixed)
    x_fixed = x_fixed.to(model.device)
    one_hot_fixed = one_hot_fixed.to(model.device, dtype=FLOAT_TYPE)

    if add_n_nodes is None:
        num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket["size"]
        )
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_fixed)
    else:
        num_nodes_lig = torch.ones(n_samples, dtype=torch.long) * (
            n_fixed + int(add_n_nodes)
        )

    ligand_mask = utils.num_nodes_to_batch_mask(
        len(num_nodes_lig), num_nodes_lig, model.device
    )
    ligand_data = {
        "x": torch.zeros(
            (len(ligand_mask), model.x_dims),
            device=model.device,
            dtype=FLOAT_TYPE,
        ),
        "one_hot": torch.zeros(
            (len(ligand_mask), model.atom_nf),
            device=model.device,
            dtype=FLOAT_TYPE,
        ),
        "size": num_nodes_lig,
        "mask": ligand_mask,
    }

    for sample_idx in range(n_samples):
        sel = ligand_mask == sample_idx
        x_new = ligand_data["x"][sel]
        h_new = ligand_data["one_hot"][sel]
        x_new[:n_fixed] = x_fixed
        h_new[:n_fixed] = one_hot_fixed
        ligand_data["x"][sel] = x_new
        ligand_data["one_hot"][sel] = h_new

    lig_fixed, lig_hard = build_masks(
        ligand_mask,
        n_samples=n_samples,
        n_fixed=n_fixed,
        hard_fixed_count=hard_fixed_count,
    )

    pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)
    xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.inpaint(
        ligand_data,
        pocket,
        lig_fixed,
        lig_hard=lig_hard,
        soft_scaffold_rho=float(soft_scaffold_rho),
        center=center,
        resamplings=resamplings,
        timesteps=timesteps,
        return_frames=frames,
        b_cloud_raw=b_cloud_raw,
        lambda_global=lambda_global,
        guide_r_raw=guide_r,
        guide_clip_raw=guide_clip,
    )

    if save_traj:
        xh_lig = utils.reverse_tensor(xh_lig)
        xh_pocket = utils.reverse_tensor(xh_pocket)
        lig_mask = torch.arange(xh_lig.size(0), device=model.device).repeat_interleave(
            len(lig_mask)
        )
        pocket_mask = torch.arange(
            xh_pocket.size(0), device=model.device
        ).repeat_interleave(len(pocket_mask))
        xh_lig = xh_lig.view(-1, xh_lig.size(2))
        xh_pocket = xh_pocket.view(-1, xh_pocket.size(2))

    pocket_com_after = scatter_mean(
        xh_pocket[:, : model.x_dims], pocket_mask, dim=0
    )
    xh_pocket[:, : model.x_dims] += (
        pocket_com_before - pocket_com_after
    )[pocket_mask]
    xh_lig[:, : model.x_dims] += (pocket_com_before - pocket_com_after)[lig_mask]

    x = xh_lig[:, : model.x_dims].detach().cpu()
    atom_type = xh_lig[:, model.x_dims :].argmax(1).detach().cpu()
    lig_mask_cpu = lig_mask.detach().cpu()

    grouped_x = utils.batch_to_list(x, lig_mask_cpu)
    grouped_types = utils.batch_to_list(atom_type, lig_mask_cpu)
    parent_xyz_cpu = x_fixed.detach().cpu()
    parent_types_cpu = one_hot_fixed.argmax(1).detach().cpu()

    canonical_x = []
    canonical_types = []

    for sample_number, (sample_x, sample_types) in enumerate(
        zip(grouped_x, grouped_types),
        start=1,
    ):
        try:
            ordered_x, ordered_types, match = canonicalize_point_cloud(
                sample_x,
                sample_types,
                parent_xyz_cpu,
                parent_types_cpu,
                hard_count=hard_fixed_count,
                hard_tolerance=0.20,
                soft_max_distance=None,
            )
        except AtomOrderError as exc:
            print(
                f"[SOFTMASK_ORDER_REJECT] sample={sample_number} "
                f"reason={exc}",
                flush=True,
            )
            continue

        canonical_x.append(ordered_x)
        canonical_types.append(ordered_types)
        print(
            f"[SOFTMASK_ORDER_OK] sample={sample_number} "
            f"parent_rmsd={match.rmsd:.6f} "
            f"parent_max={match.max_distance:.6f}",
            flush=True,
        )

    if not canonical_x:
        return []

    canonical_mask = torch.cat(
        [
            torch.full(
                (len(sample_x),),
                sample_idx,
                dtype=lig_mask_cpu.dtype,
            )
            for sample_idx, sample_x in enumerate(canonical_x)
        ],
        dim=0,
    )
    canonical_x_flat = torch.cat(canonical_x, dim=0)
    canonical_types_flat = torch.cat(canonical_types, dim=0)

    _write_softmask_debug(
        softmask_debug or os.environ.get("DC_SOFTMASK_DEBUG_JSON", ""),
        canonical_x_flat,
        canonical_types_flat,
        canonical_mask,
        parent_xyz_cpu,
        one_hot_fixed.detach().cpu(),
        n_fixed,
        hard_fixed_count,
        soft_scaffold_rho,
    )

    molecules = []
    atom_decoder = model.dataset_info["atom_decoder"]

    for sample_number, (sample_x, sample_types) in enumerate(
        zip(canonical_x, canonical_types),
        start=1,
    ):
        mol = build_molecule(
            sample_x,
            sample_types,
            model.dataset_info,
            add_coords=True,
        )
        mol = process_molecule(
            mol,
            add_hydrogens=False,
            sanitize=sanitize,
            relax_iter=relax_iter,
            largest_frag=largest_frag,
        )
        if mol is None:
            continue

        try:
            expected_atomic_numbers = atom_type_indices_to_atomic_numbers(
                sample_types.numpy(),
                atom_decoder,
            )
            mol, _ = canonicalize_rdkit_mol(
                mol,
                sample_x.numpy(),
                expected_atomic_numbers,
                hard_count=0,
                hard_tolerance=0.20,
                soft_max_distance=0.25,
            )
        except AtomOrderError as exc:
            print(
                f"[SOFTMASK_RDKIT_ORDER_REJECT] sample={sample_number} "
                f"reason={exc}",
                flush=True,
            )
            continue

        molecules.append(mol)

    return molecules

def load_b_cloud(path):
    if path is None:
        return None
    mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        raise SystemExit(f"Could not read B shape ligand: {path}")
    conf = mol.GetConformer()
    points = [
        list(conf.GetAtomPosition(atom.GetIdx()))
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    return torch.tensor(points, dtype=FLOAT_TYPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pdbfile", type=str, required=True)
    parser.add_argument("--ref_ligand", type=str, default=None)
    parser.add_argument("--fix_atoms", type=str, nargs="+", required=True)
    parser.add_argument(
        "--center", type=str, default="ligand", choices={"ligand", "pocket"}
    )
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--add_n_nodes", type=int, default=None)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--resamplings", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--b_shape_ligand", type=str, default=None)
    parser.add_argument("--lambda_global", type=float, default=0.0)
    parser.add_argument("--guide_r", type=float, default=1.7)
    parser.add_argument("--guide_clip", type=float, default=1.0)
    parser.add_argument("--soft_scaffold_rho", type=float, default=1.0)
    parser.add_argument("--hard_fixed_count", type=int, default=7)
    parser.add_argument("--softmask_debug", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).to(device)
    b_cloud_raw = load_b_cloud(args.b_shape_ligand)

    molecules = inpaint_ligand(
        model,
        args.pdbfile,
        args.n_samples,
        args.ref_ligand,
        args.fix_atoms,
        args.add_n_nodes,
        center=args.center,
        sanitize=args.sanitize,
        largest_frag=False,
        relax_iter=200 if args.relax else 0,
        timesteps=args.timesteps,
        resamplings=args.resamplings,
        save_traj=args.save_traj,
        b_cloud_raw=b_cloud_raw,
        lambda_global=args.lambda_global,
        guide_r=args.guide_r,
        guide_clip=args.guide_clip,
        soft_scaffold_rho=args.soft_scaffold_rho,
        hard_fixed_count=args.hard_fixed_count,
        softmask_debug=args.softmask_debug,
    )
    utils.write_sdf_file(args.outfile, molecules)


if __name__ == "__main__":
    main()
