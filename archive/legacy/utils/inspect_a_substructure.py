import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

A_ID = "x0874"
B_ID = "x1093"

READY = Path("data/mpro/manifests/silvr_xchem_ready_manifest.tsv")
OUTDIR = Path("artifacts/reports/substructure")
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIMARY_PAIR = Path("data/mpro/manifests/b0_primary_pair.tsv")
ATOM_TABLE = Path("data/mpro/manifests/b0_x0874_T54_atom_table.tsv")
CANDIDATES_TXT = Path("artifacts/reports/substructure/b0_x0874_T54_substructure_candidates.txt")
FIXED_JSON = Path("data/mpro/manifests/b0_x0874_T54_fixed_atoms.json")


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_pdb_atom_names(pdb_path):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atoms.append({
                "pdb_serial": int(line[6:11]),
                "pdb_atom_name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resseq": line[22:26].strip(),
                "x_pdb": float(line[30:38]),
                "y_pdb": float(line[38:46]),
                "z_pdb": float(line[46:54]),
                "element_pdb": line[76:78].strip() if len(line) >= 78 else "",
            })
    return atoms


def mol_coords(mol):
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    return np.asarray(coords, dtype=float)


rows = read_rows(READY)
a = next(r for r in rows if r["xchem_id"] == A_ID)
b = next(r for r in rows if r["xchem_id"] == B_ID)

mol = Chem.MolFromMolFile(a["ligand_sdf"], sanitize=True, removeHs=False)
if mol is None:
    raise SystemExit(f"RDKit failed to read {a['ligand_sdf']}")

pdb_atoms = read_pdb_atom_names(a["ligand_pdb"])
coords = mol_coords(mol)

if len(pdb_atoms) != mol.GetNumAtoms():
    print(f"WARNING: PDB atom count {len(pdb_atoms)} != RDKit atom count {mol.GetNumAtoms()}")

heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
heavy_coords = coords[heavy_indices]
centroid = heavy_coords.mean(axis=0)

ring_info = mol.GetRingInfo()
rings = list(ring_info.AtomRings())

candidate_sets = []

for i, ring in enumerate(rings, 1):
    heavy_ring = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1]
    candidate_sets.append({
        "name": f"ring_{i}",
        "reason": "RDKit ring",
        "atom_indices_0based": heavy_ring,
    })

# Expanded ring candidates: ring plus directly attached heavy substituent atoms.
for i, ring in enumerate(rings, 1):
    s = set(idx for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)
    for idx in list(s):
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() > 1:
                s.add(nb.GetIdx())
    candidate_sets.append({
        "name": f"ring_{i}_plus_neighbors",
        "reason": "ring plus first heavy neighbors",
        "atom_indices_0based": sorted(s),
    })

# Central fallback: heavy atoms closest to heavy-atom centroid.
dist = []
for idx in heavy_indices:
    d = float(np.linalg.norm(coords[idx] - centroid))
    dist.append((d, idx))
dist.sort()
central = sorted(idx for _, idx in dist[:min(6, len(dist))])
candidate_sets.append({
    "name": "central_6_heavy_atoms",
    "reason": "fallback: closest heavy atoms to centroid",
    "atom_indices_0based": central,
})

# Prefer a moderate rigid set: largest ring plus neighbors if available; otherwise central_6.
if rings:
    ranked = sorted(
        [c for c in candidate_sets if c["name"].endswith("_plus_neighbors")],
        key=lambda c: len(c["atom_indices_0based"]),
        reverse=True,
    )
    chosen = ranked[0]
else:
    chosen = candidate_sets[-1]

atom_rows = []
for atom in mol.GetAtoms():
    idx = atom.GetIdx()
    pdb = pdb_atoms[idx] if idx < len(pdb_atoms) else {}
    atom_rows.append({
        "rdkit_idx_0based": idx,
        "rdkit_idx_1based": idx + 1,
        "pdb_serial": pdb.get("pdb_serial", ""),
        "pdb_atom_name": pdb.get("pdb_atom_name", ""),
        "symbol": atom.GetSymbol(),
        "atomic_num": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "is_aromatic": atom.GetIsAromatic(),
        "is_in_ring": atom.IsInRing(),
        "x": coords[idx][0],
        "y": coords[idx][1],
        "z": coords[idx][2],
        "chosen_fixed": idx in set(chosen["atom_indices_0based"]),
    })

with ATOM_TABLE.open("w", newline="") as f:
    fields = list(atom_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(atom_rows)

with PRIMARY_PAIR.open("w", newline="") as f:
    fields = [
        "role", "xchem_id", "ligand_comp_id", "base",
        "protein_pdb", "ligand_pdb", "ligand_sdf", "complex_pdb"
    ]
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerow({
        "role": "A_local",
        "xchem_id": a["xchem_id"],
        "ligand_comp_id": a["ligand_comp_id"],
        "base": a["base"],
        "protein_pdb": a["protein_pdb"],
        "ligand_pdb": a["ligand_pdb"],
        "ligand_sdf": a["ligand_sdf"],
        "complex_pdb": a["complex_pdb"],
    })
    w.writerow({
        "role": "B_global",
        "xchem_id": b["xchem_id"],
        "ligand_comp_id": b["ligand_comp_id"],
        "base": b["base"],
        "protein_pdb": b["protein_pdb"],
        "ligand_pdb": b["ligand_pdb"],
        "ligand_sdf": b["ligand_sdf"],
        "complex_pdb": b["complex_pdb"],
    })

fixed_payload = {
    "pair": {
        "A_local": A_ID,
        "B_global": B_ID,
        "A_comp": a["ligand_comp_id"],
        "B_comp": b["ligand_comp_id"],
    },
    "chosen_candidate": chosen["name"],
    "reason": chosen["reason"],
    "fixed_atom_indices_0based": chosen["atom_indices_0based"],
    "fixed_atom_indices_1based": [i + 1 for i in chosen["atom_indices_0based"]],
    "fixed_pdb_atom_names": [
        pdb_atoms[i]["pdb_atom_name"] if i < len(pdb_atoms) else ""
        for i in chosen["atom_indices_0based"]
    ],
    "A_ligand_sdf": a["ligand_sdf"],
    "A_ligand_pdb": a["ligand_pdb"],
    "A_protein_pdb": a["protein_pdb"],
    "B_ligand_sdf": b["ligand_sdf"],
    "B_protein_pdb": b["protein_pdb"],
}

with FIXED_JSON.open("w") as f:
    json.dump(fixed_payload, f, indent=2)

with CANDIDATES_TXT.open("w") as f:
    f.write("B0 primary pair and A-local substructure candidates\n")
    f.write("===================================================\n\n")
    f.write(f"A_local = {A_ID} / {a['ligand_comp_id']} / {a['base']}\n")
    f.write(f"B_global = {B_ID} / {b['ligand_comp_id']} / {b['base']}\n\n")
    f.write(f"A SDF = {a['ligand_sdf']}\n")
    f.write(f"B SDF = {b['ligand_sdf']}\n\n")
    f.write(f"A formula = {rdMolDescriptors.CalcMolFormula(mol)}\n")
    f.write(f"A MW = {Descriptors.MolWt(mol):.3f}\n")
    f.write(f"A atoms = {mol.GetNumAtoms()}; heavy = {mol.GetNumHeavyAtoms()}; bonds = {mol.GetNumBonds()}\n")
    f.write(f"A rings = {len(rings)}\n\n")

    for c in candidate_sets:
        idxs = c["atom_indices_0based"]
        names = [pdb_atoms[i]["pdb_atom_name"] if i < len(pdb_atoms) else str(i) for i in idxs]
        elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in idxs]
        f.write(f"- {c['name']} ({c['reason']})\n")
        f.write(f"  n_atoms={len(idxs)}\n")
        f.write(f"  indices_0based={idxs}\n")
        f.write(f"  indices_1based={[i + 1 for i in idxs]}\n")
        f.write(f"  pdb_atom_names={names}\n")
        f.write(f"  elements={elems}\n\n")

    f.write("CHOSEN_INITIAL_FIXED_SET\n")
    f.write("------------------------\n")
    f.write(json.dumps(fixed_payload, indent=2))
    f.write("\n")

print("Wrote:", PRIMARY_PAIR)
print("Wrote:", ATOM_TABLE)
print("Wrote:", CANDIDATES_TXT)
print("Wrote:", FIXED_JSON)
print()
print("Chosen fixed set:", chosen["name"])
print("0-based:", fixed_payload["fixed_atom_indices_0based"])
print("1-based:", fixed_payload["fixed_atom_indices_1based"])
print("PDB names:", fixed_payload["fixed_pdb_atom_names"])
