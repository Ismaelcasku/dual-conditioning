from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT = Path.cwd()
DIFFSBDD = PROJECT / "external/DiffSBDD"
INPAINT = DIFFSBDD / "inpaint.py"
LIGHTNING = DIFFSBDD / "lightning_modules.py"

CKPT = PROJECT / "artifacts/checkpoints/crossdocked_fullatom_cond.ckpt"
READY_MANIFEST = PROJECT / "data/mpro/manifests/silvr_xchem_ready_manifest.tsv"
FIXED_JSON = PROJECT / "data/mpro/manifests/b0_x0874_T54_fixed_atoms.json"
PRIMARY_PAIR = PROJECT / "data/mpro/manifests/b0_primary_pair.tsv"

REPORT = PROJECT / "artifacts/reports/preflight/12_inpaint_contract_preflight.txt"
REPORT.parent.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def source_segment(path: Path, func_name: str):
    txt = read_text(path)
    tree = ast.parse(txt)
    lines = txt.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == func_name:
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            body = "\n".join(f"{i:04d}: {lines[i-1]}" for i in range(start, end + 1))
            return start, end, body

    return None, None, ""


def list_imports(path: Path):
    txt = read_text(path)
    tree = ast.parse(txt)
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])

    return sorted(set(imports))


def extract_argparse_lines(path: Path):
    lines = read_text(path).splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        if "add_argument" in line or "argparse.ArgumentParser" in line:
            out.append(f"{i:04d}: {line}")
    return out


def run_cmd(cmd, cwd=None, env=None):
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, p.stdout, p.stderr


def read_tsv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_pdb_atoms(path: Path):
    atoms = []
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atoms.append({
                "record": line[0:6].strip(),
                "serial": line[6:11].strip(),
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resseq": line[22:26].strip(),
                "icode": line[26:27].strip(),
                "x": line[30:38].strip(),
                "y": line[38:46].strip(),
                "z": line[46:54].strip(),
                "element": line[76:78].strip() if len(line) >= 78 else "",
            })

    return atoms


def main():
    original_stdout = sys.stdout
    with REPORT.open("w") as f:
        sys.stdout = f

        section("0. Scope")
        print("This preflight does NOT launch generation.")
        print("Goal: fully determine the DiffSBDD inpaint.py contract before the next GPU job.")
        print("Checks: source contract, optional imports, model load, data validity, fix_atoms semantics.")
        print("PROJECT =", PROJECT)
        print("DIFFSBDD =", DIFFSBDD)

        section("1. File existence")
        for p in [INPAINT, LIGHTNING, CKPT, READY_MANIFEST, FIXED_JSON, PRIMARY_PAIR]:
            print(f"{p}: exists={p.exists()} size={p.stat().st_size if p.exists() else 'NA'}")

        section("2. CLI contract from inpaint.py")
        for line in extract_argparse_lines(INPAINT):
            print(line)

        section("3. prepare_substructure source")
        start, end, body = source_segment(INPAINT, "prepare_substructure")
        print(f"prepare_substructure lines: {start}-{end}")
        print(body)

        section("4. inpaint_ligand source")
        start, end, body = source_segment(INPAINT, "inpaint_ligand")
        print(f"inpaint_ligand lines: {start}-{end}")
        print(body)

        section("5. Static interpretation of fix_atoms semantics")
        inpaint_txt = read_text(INPAINT)
        prep_txt = body

        if "split(':')" in inpaint_txt:
            print("FOUND: split(':') in inpaint.py")
            print("Interpretation: at least one user-facing spec is parsed as colon-separated text.")
        else:
            print("NOT FOUND: split(':') in inpaint.py")

        if "chain, resi = " in inpaint_txt:
            print("FOUND: pattern similar to `chain, resi = ...`")
            print("Likely official contract: each fixed item may be a residue spec like CHAIN:RESI.")
            print("Consequence: atom names alone, e.g. C02 C04, are invalid.")
        else:
            print("NOT FOUND: direct `chain, resi =` pattern.")

        if "atom" in inpaint_txt and "get_atoms" in inpaint_txt:
            print("FOUND: get_atoms usage.")
            print("Need to inspect whether atom-level filtering exists or only whole-residue fixing.")

        print()
        print("Required decision after preflight:")
        print("- If official contract is CHAIN:RESI only, it fixes the entire ligand residue.")
        print("- For our scientific objective, we need atom-level fixing.")
        print("- Therefore we likely need a controlled patch supporting CHAIN:RESI:ATOM_NAME or index-based fixed atoms.")

        section("6. Import dependency audit from source")
        for path in [INPAINT, LIGHTNING]:
            print(f"\nImports in {path.relative_to(PROJECT)}:")
            for mod in list_imports(path):
                print(" ", mod)

        section("7. Primary pair and selected local substructure")
        pair_rows = read_tsv(PRIMARY_PAIR)
        for r in pair_rows:
            print(json.dumps(r, indent=2))

        fixed = json.loads(FIXED_JSON.read_text())
        print("\nFixed JSON:")
        print(json.dumps(fixed, indent=2))

        section("8. PDB/SDF data contract for A=x0874")
        a_row = next(r for r in pair_rows if r["role"] == "A_local")
        a_lig_pdb = PROJECT / a_row["ligand_pdb"]
        a_lig_sdf = PROJECT / a_row["ligand_sdf"]
        a_protein_pdb = PROJECT / a_row["protein_pdb"]
        a_complex_pdb = PROJECT / a_row["complex_pdb"]

        print("A ligand PDB:", a_lig_pdb, "exists=", a_lig_pdb.exists())
        print("A ligand SDF:", a_lig_sdf, "exists=", a_lig_sdf.exists())
        print("A protein PDB:", a_protein_pdb, "exists=", a_protein_pdb.exists())
        print("A complex PDB:", a_complex_pdb, "exists=", a_complex_pdb.exists())

        lig_atoms = parse_pdb_atoms(a_lig_pdb)
        print("\nLigand PDB atoms:")
        for a in lig_atoms:
            mark = "FIXED" if a["atom_name"] in fixed["fixed_pdb_atom_names"] else ""
            print(
                f"{a['record']:6s} serial={a['serial']:>4s} "
                f"name={a['atom_name']:>4s} res={a['resname']} "
                f"chain={a['chain']} resseq={a['resseq']} elem={a['element']:>2s} {mark}"
            )

        residues = sorted(set((a["chain"], a["resseq"], a["resname"]) for a in lig_atoms))
        print("\nLigand residue identifiers from PDB:")
        for chain, resseq, resname in residues:
            print(f"  chain={chain} resseq={resseq} resname={resname}")
            print(f"  possible whole-residue fix spec: {chain}:{resseq}")
            for name in fixed["fixed_pdb_atom_names"]:
                print(f"  possible atom-level fix spec proposal: {chain}:{resseq}:{name}")

        section("9. RDKit read/sanitize contract")
        try:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(a_lig_sdf), sanitize=False, removeHs=False)
            print("RDKit MolFromMolFile:", mol is not None)
            if mol is not None:
                print("atoms:", mol.GetNumAtoms())
                print("heavy_atoms:", mol.GetNumHeavyAtoms())
                print("bonds:", mol.GetNumBonds())
                print("conformers:", mol.GetNumConformers())
                try:
                    Chem.SanitizeMol(mol)
                    print("sanitize: OK")
                except Exception as e:
                    print("sanitize: FAILED", repr(e))
        except Exception as e:
            print("RDKit import/read failed:", repr(e))

        section("10. Runtime import and checkpoint test")
        print("This part is executed by the surrounding shell inside the sandbox.")
        print("See appended runtime section below.")

        sys.stdout = original_stdout

    print(f"Wrote static report: {REPORT}")


if __name__ == "__main__":
    main()
