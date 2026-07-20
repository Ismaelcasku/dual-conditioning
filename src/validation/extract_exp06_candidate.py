import re
import sys
from pathlib import Path
from rdkit import Chem

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
candidate_id = sys.argv[3]

sample = int(re.search(r"sample(\d+)$", candidate_id).group(1))

supplier = Chem.SDMolSupplier(
    str(src),
    removeHs=False,
    sanitize=False,
)

selected = None

for index, mol in enumerate(supplier):
    if mol is None:
        continue

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    metadata = " ".join(
        [name]
        + [
            f"{key}={mol.GetProp(key)}"
            for key in mol.GetPropNames()
        ]
    ).lower()

    if candidate_id.lower() in metadata:
        selected = (index, mol)
        break

    if (
        f"sample{sample}" in metadata
        or f"sample_{sample}" in metadata
        or f"sample={sample}" in metadata
    ):
        selected = (index, mol)

if selected is None:
    raise SystemExit(
        f"ERROR: no se encontró {candidate_id} en {src}"
    )

index, mol = selected
Chem.SanitizeMol(mol)

mol.SetProp("_Name", candidate_id)
mol.SetProp("exp06_candidate_id", candidate_id)
mol.SetIntProp("source_record_index", index)

dst.parent.mkdir(parents=True, exist_ok=True)

writer = Chem.SDWriter(str(dst))
writer.write(mol)
writer.close()

print(f"selected_record_index={index}")
print(f"output={dst}")
print("CANDIDATE_EXTRACTION_STATUS=OK")
