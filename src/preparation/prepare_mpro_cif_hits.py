from __future__ import print_function

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict

EXCLUDE_HET = set([
    "HOH", "WAT", "DOD",
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO",
    "SO4", "PO4", "NO3", "ACT", "ACE", "EDO", "GOL", "PEG", "DMS",
    "TRS", "MES", "HEP", "MPD", "IPA", "FMT", "TFA", "CAC"
])

AA3 = set([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE"
])


def norm_value(x):
    if x is None:
        return ""
    if x in [".", "?"]:
        return ""
    return x


def tokenize_cif_line(line):
    try:
        return shlex.split(line, comments=False, posix=True)
    except Exception:
        return line.split()


def read_atom_site(cif_path):
    with open(cif_path, "r") as f:
        lines = f.readlines()

    atoms = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if line != "loop_":
            i += 1
            continue

        i += 1
        tags = []

        while i < n:
            s = lines[i].strip()
            if not s or s.startswith("#"):
                i += 1
                continue
            if s.startswith("_"):
                tags.append(s)
                i += 1
                continue
            break

        if not tags:
            continue

        is_atom_site = any(t.startswith("_atom_site.") for t in tags)
        values = []

        while i < n:
            raw = lines[i]
            s = raw.strip()

            if not s or s.startswith("#"):
                i += 1
                continue

            if s == "loop_" or s.startswith("data_") or s.startswith("save_") or s.startswith("_"):
                break

            if s.startswith(";"):
                # Atom_site rows should not need multiline fields. Skip defensively.
                i += 1
                while i < n and not lines[i].startswith(";"):
                    i += 1
                if i < n:
                    i += 1
                continue

            if is_atom_site:
                values.extend(tokenize_cif_line(raw))
            i += 1

        if is_atom_site:
            cols = [t.replace("_atom_site.", "") for t in tags]
            width = len(cols)
            if width == 0:
                continue

            if len(values) % width != 0:
                print("WARNING: token count not divisible by atom_site column count in {}".format(cif_path), file=sys.stderr)
                print("tokens={} columns={}".format(len(values), width), file=sys.stderr)

            usable = (len(values) // width) * width
            for j in range(0, usable, width):
                row = dict(zip(cols, values[j:j + width]))
                atoms.append(row)

    return atoms


def get_field(atom, keys, default=""):
    for k in keys:
        if k in atom:
            v = norm_value(atom.get(k))
            if v != "":
                return v
    return default


def model_ok(atom):
    m = get_field(atom, ["pdbx_PDB_model_num"], "1")
    return m in ["", "1"]


def alt_ok(atom):
    a = get_field(atom, ["label_alt_id"], "")
    return a in ["", "A", "1"]


def atom_xyz(atom):
    return (
        float(get_field(atom, ["Cartn_x"], "0")),
        float(get_field(atom, ["Cartn_y"], "0")),
        float(get_field(atom, ["Cartn_z"], "0")),
    )


def residue_key(atom):
    comp = get_field(atom, ["auth_comp_id", "label_comp_id"], "UNK").upper()
    chain = get_field(atom, ["auth_asym_id", "label_asym_id"], "A")
    seq = get_field(atom, ["auth_seq_id", "label_seq_id"], "1")
    ins = get_field(atom, ["pdbx_PDB_ins_code"], "")
    return (comp, chain, seq, ins)


def select_ligand(atoms):
    candidates = defaultdict(list)

    for atom in atoms:
        if not model_ok(atom) or not alt_ok(atom):
            continue

        group = get_field(atom, ["group_PDB"], "").upper()
        comp = get_field(atom, ["auth_comp_id", "label_comp_id"], "UNK").upper()
        elem = get_field(atom, ["type_symbol"], "").upper()

        if group != "HETATM":
            continue
        if comp in EXCLUDE_HET:
            continue
        if comp in AA3:
            continue

        candidates[residue_key(atom)].append(atom)

    scored = []

    for key, rows in candidates.items():
        heavy = 0
        carbons = 0
        elems = defaultdict(int)

        for atom in rows:
            elem = get_field(atom, ["type_symbol"], "").upper()
            if elem not in ["H", "D"]:
                heavy += 1
            if elem == "C":
                carbons += 1
            elems[elem] += 1

        scored.append({
            "key": key,
            "atoms": rows,
            "n_atoms": len(rows),
            "heavy_atoms": heavy,
            "carbons": carbons,
            "score": (carbons > 0, heavy, len(rows)),
            "elements": dict(elems),
        })

    if not scored:
        return None, []

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0], scored


def pdb_atom_line(atom, serial, force_record=None):
    group = get_field(atom, ["group_PDB"], "HETATM").upper()
    record = force_record if force_record else ("ATOM" if group == "ATOM" else "HETATM")

    name = get_field(atom, ["auth_atom_id", "label_atom_id"], "X")
    alt = get_field(atom, ["label_alt_id"], "")
    comp = get_field(atom, ["auth_comp_id", "label_comp_id"], "UNK").upper()
    chain = get_field(atom, ["auth_asym_id", "label_asym_id"], "A")
    seq = get_field(atom, ["auth_seq_id", "label_seq_id"], "1")
    ins = get_field(atom, ["pdbx_PDB_ins_code"], "")
    elem = get_field(atom, ["type_symbol"], "")

    try:
        resseq = int(float(seq))
    except Exception:
        resseq = 1

    try:
        occ = float(get_field(atom, ["occupancy"], "1.00"))
    except Exception:
        occ = 1.00

    try:
        bfac = float(get_field(atom, ["B_iso_or_equiv"], "0.00"))
    except Exception:
        bfac = 0.00

    x, y, z = atom_xyz(atom)

    if len(chain) < 1:
        chain = "A"
    chain = chain[0]

    if len(ins) < 1:
        ins = " "
    ins = ins[0]

    if len(name) < 4:
        atom_name = " %-3s" % name
    else:
        atom_name = name[:4]

    return "{:<6}{:5d} {:<4}{:1}{:>3} {:1}{:4d}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2}{:>2}\n".format(
        record[:6],
        serial,
        atom_name[:4],
        alt[:1] if alt else " ",
        comp[:3],
        chain,
        resseq,
        ins,
        x,
        y,
        z,
        occ,
        bfac,
        elem[:2],
        "",
    )


def write_pdb(path, atoms, ligand_atoms=None):
    serial = 1
    with open(path, "w") as f:
        f.write("REMARK generated by prepare_mpro_cif_hits.py\n")

        for atom in atoms:
            if not model_ok(atom) or not alt_ok(atom):
                continue
            group = get_field(atom, ["group_PDB"], "").upper()
            if group == "ATOM":
                f.write(pdb_atom_line(atom, serial, "ATOM"))
                serial += 1

        if ligand_atoms:
            f.write("TER\n")
            for atom in ligand_atoms:
                f.write(pdb_atom_line(atom, serial, "HETATM"))
                serial += 1

        f.write("END\n")


def write_ligand_pdb(path, ligand_atoms):
    serial = 1
    with open(path, "w") as f:
        f.write("REMARK ligand extracted by prepare_mpro_cif_hits.py\n")
        for atom in ligand_atoms:
            f.write(pdb_atom_line(atom, serial, "HETATM"))
            serial += 1
        f.write("END\n")


def run_obabel(ligand_pdb, ligand_sdf):
    obabel = shutil.which("obabel")
    if not obabel:
        return False, "obabel not found"

    cmd = [obabel, "-ipdb", ligand_pdb, "-osdf", "-O", ligand_sdf]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    msg = ""
    try:
        msg += out.decode("utf-8", "ignore")
    except Exception:
        msg += str(out)
    try:
        msg += err.decode("utf-8", "ignore")
    except Exception:
        msg += str(err)

    return p.returncode == 0 and os.path.exists(ligand_sdf) and os.path.getsize(ligand_sdf) > 0, msg


def xchem_id_from_name(name):
    m = re.search(r"(x[0-9]{4})", name, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit("Input directory not found: {}".format(args.input_dir))

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    cif_files = []
    for fn in sorted(os.listdir(args.input_dir)):
        if fn.endswith("_ligand-bound-model.cif"):
            cif_files.append(os.path.join(args.input_dir, fn))

    if not cif_files:
        raise SystemExit("No *_ligand-bound-model.cif files found in {}".format(args.input_dir))

    rows = []

    for cif in cif_files:
        base = os.path.basename(cif).replace("_ligand-bound-model.cif", "")
        xid = xchem_id_from_name(base)

        outdir = os.path.join(args.output_dir, xid + "__" + base)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        atoms = read_atom_site(cif)
        selected, candidates = select_ligand(atoms)

        protein_pdb = os.path.join(outdir, base + "_protein.pdb")
        complex_pdb = os.path.join(outdir, base + "_complex.pdb")
        ligand_pdb = os.path.join(outdir, base + "_ligand.pdb")
        ligand_sdf = os.path.join(outdir, base + "_ligand.sdf")
        meta_json = os.path.join(outdir, base + "_metadata.json")

        status = "OK"
        obabel_ok = False
        obabel_msg = ""

        if selected is None:
            status = "NO_LIGAND_FOUND"
            ligand_atoms = []
            write_pdb(protein_pdb, atoms, None)
            write_pdb(complex_pdb, atoms, None)
        else:
            ligand_atoms = selected["atoms"]
            write_pdb(protein_pdb, atoms, None)
            write_pdb(complex_pdb, atoms, ligand_atoms)
            write_ligand_pdb(ligand_pdb, ligand_atoms)
            obabel_ok, obabel_msg = run_obabel(ligand_pdb, ligand_sdf)
            if not obabel_ok:
                status = "OBABEL_FAILED"

        protein_atoms = 0
        for atom in atoms:
            if model_ok(atom) and alt_ok(atom) and get_field(atom, ["group_PDB"], "").upper() == "ATOM":
                protein_atoms += 1

        key = selected["key"] if selected else ("", "", "", "")
        meta = {
            "xchem_id": xid,
            "source_cif": cif,
            "base": base,
            "status": status,
            "protein_atoms": protein_atoms,
            "selected_ligand": {
                "comp_id": key[0],
                "chain": key[1],
                "seq": key[2],
                "ins": key[3],
                "n_atoms": selected["n_atoms"] if selected else 0,
                "heavy_atoms": selected["heavy_atoms"] if selected else 0,
                "carbons": selected["carbons"] if selected else 0,
                "elements": selected["elements"] if selected else {},
            },
            "n_ligand_candidates": len(candidates),
            "candidate_summary": [
                {
                    "comp_id": c["key"][0],
                    "chain": c["key"][1],
                    "seq": c["key"][2],
                    "ins": c["key"][3],
                    "n_atoms": c["n_atoms"],
                    "heavy_atoms": c["heavy_atoms"],
                    "carbons": c["carbons"],
                    "elements": c["elements"],
                }
                for c in candidates
            ],
            "paths": {
                "protein_pdb": protein_pdb,
                "complex_pdb": complex_pdb,
                "ligand_pdb": ligand_pdb,
                "ligand_sdf": ligand_sdf if os.path.exists(ligand_sdf) else "",
                "metadata_json": meta_json,
            },
            "obabel_ok": obabel_ok,
            "obabel_message": obabel_msg[-2000:],
        }

        with open(meta_json, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        rows.append({
            "xchem_id": xid,
            "base": base,
            "status": status,
            "protein_atoms": protein_atoms,
            "ligand_comp_id": key[0],
            "ligand_chain": key[1],
            "ligand_seq": key[2],
            "ligand_atoms": selected["n_atoms"] if selected else 0,
            "ligand_heavy_atoms": selected["heavy_atoms"] if selected else 0,
            "ligand_carbons": selected["carbons"] if selected else 0,
            "protein_pdb": protein_pdb,
            "complex_pdb": complex_pdb,
            "ligand_pdb": ligand_pdb,
            "ligand_sdf": ligand_sdf if os.path.exists(ligand_sdf) else "",
            "metadata_json": meta_json,
            "source_cif": cif,
        })

        print("{}\t{}\t{}\tligand={} chain={} seq={} heavy={}".format(
            xid,
            base,
            status,
            key[0],
            key[1],
            key[2],
            selected["heavy_atoms"] if selected else 0,
        ))

    fieldnames = [
        "xchem_id", "base", "status",
        "protein_atoms",
        "ligand_comp_id", "ligand_chain", "ligand_seq",
        "ligand_atoms", "ligand_heavy_atoms", "ligand_carbons",
        "protein_pdb", "complex_pdb", "ligand_pdb", "ligand_sdf",
        "metadata_json", "source_cif"
    ]

    with open(args.manifest, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("")
    print("Wrote manifest: {}".format(args.manifest))
    print("Prepared entries: {}".format(len(rows)))


if __name__ == "__main__":
    main()
