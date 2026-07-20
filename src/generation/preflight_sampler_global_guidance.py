from pathlib import Path
import re
import json
from datetime import datetime

PROJECT = Path.cwd()
ROOT = PROJECT / "external/DiffSBDD"
OUTDIR = PROJECT / "artifacts/reports/global_guidance_preflight"
OUTDIR.mkdir(parents=True, exist_ok=True)
REPORT = OUTDIR / "16_sampler_global_guidance_preflight.md"
JSON_OUT = OUTDIR / "16_sampler_global_guidance_preflight_candidates.json"

SOURCE_DIRS = [
    ROOT / "equivariant_diffusion",
    ROOT,
]

KEY_PATTERNS = {
    "denoising_loop": [
        r"for .*range",
        r"reversed",
        r"timesteps",
        r"sample_p_zs_given_zt",
        r"sample_p_xh_given_z0",
        r"sample_chain",
        r"sample",
        r"denois",
    ],
    "ligand_update": [
        r"xh_lig",
        r"z_lig",
        r"lig_mask",
        r"xh =",
        r"z =",
        r"sample_p_zs_given_zt",
        r"sample_p_xh_given_z0",
    ],
    "inpainting_mask": [
        r"lig_fixed",
        r"fixed",
        r"fix_atoms",
        r"x_fixed",
        r"one_hot_fixed",
        r"torch.where",
        r"mask",
        r"inpaint",
    ],
    "coordinate_frame": [
        r"center",
        r"pocket",
        r"ligand",
        r"remove_mean",
        r"center_of_mass",
        r"mean",
        r"com",
        r"pos",
    ],
}

FUNCTION_PATTERNS = [
    r"def inpaint",
    r"def sample",
    r"def sample_chain",
    r"def sample_p_zs_given_zt",
    r"def sample_p_xh_given_z0",
    r"def forward",
]


def read(path):
    return path.read_text(errors="replace")


def list_py_files():
    files = []
    seen = set()
    for d in SOURCE_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    return sorted(files)


def line_hits(path, patterns):
    txt = read(path).splitlines()
    hits = []
    for i, line in enumerate(txt, 1):
        for pat in patterns:
            if re.search(pat, line):
                hits.append((i, pat, line.rstrip()))
                break
    return hits


def snippet(path, center_line, before=18, after=42):
    lines = read(path).splitlines()
    start = max(1, center_line - before)
    end = min(len(lines), center_line + after)
    return "\n".join(f"{i:04d}: {lines[i-1]}" for i in range(start, end + 1))


def compact_hits(files, patterns, max_hits=80):
    out = []
    for p in files:
        hits = line_hits(p, patterns)
        for h in hits:
            out.append((p, *h))
    return out[:max_hits], len(out)


def score_file(path):
    txt = read(path)
    score = 0
    for group, pats in KEY_PATTERNS.items():
        for pat in pats:
            if re.search(pat, txt):
                score += 1
    return score


def main():
    files = list_py_files()
    candidates = []

    for p in files:
        s = score_file(p)
        if s > 0:
            candidates.append({
                "path": str(p.relative_to(PROJECT)),
                "score": s,
                "hits": {
                    group: len(line_hits(p, pats))
                    for group, pats in KEY_PATTERNS.items()
                }
            })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    JSON_OUT.write_text(json.dumps(candidates, indent=2))

    md = []
    md.append("# Preflight 16 — Sampler/global-guidance insertion map\n")
    md.append(f"Created: `{datetime.now().isoformat(timespec='seconds')}`\n")
    md.append("Scope: **read-only source inspection**. No code modifications were made.\n\n")

    md.append("## Goal\n\n")
    md.append("This preflight answers three questions before implementing `lambda_global`:\n\n")
    md.append("1. Where is the real denoising loop and how is the ligand tensor updated?\n")
    md.append("2. How is the inpainting/fixed-atom mask reapplied at each step?\n")
    md.append("3. In what coordinate frame do the loop coordinates live, and how must B be transformed?\n\n")

    md.append("## Candidate source files ranked by relevance\n\n")
    md.append("| rank | file | score | denoising_loop hits | ligand_update hits | inpainting_mask hits | coordinate_frame hits |\n")
    md.append("|---:|---|---:|---:|---:|---:|---:|\n")
    for i, c in enumerate(candidates[:20], 1):
        h = c["hits"]
        md.append(
            f"| {i} | `{c['path']}` | {c['score']} | "
            f"{h['denoising_loop']} | {h['ligand_update']} | {h['inpainting_mask']} | {h['coordinate_frame']} |\n"
        )

    md.append("\n## Function definitions of interest\n\n")
    func_hits, total_func_hits = compact_hits(files, FUNCTION_PATTERNS, max_hits=120)
    md.append(f"Total function-definition hits: `{total_func_hits}`. Showing first `{len(func_hits)}`.\n\n")
    md.append("```text\n")
    for p, line_no, pat, line in func_hits:
        md.append(f"{p.relative_to(PROJECT)}:{line_no}: {line}\n")
    md.append("```\n\n")

    for group, patterns in KEY_PATTERNS.items():
        md.append(f"## Grep map: {group}\n\n")
        hits, total = compact_hits(files, patterns, max_hits=160)
        md.append(f"Total hits: `{total}`. Showing first `{len(hits)}`.\n\n")
        md.append("```text\n")
        for p, line_no, pat, line in hits:
            md.append(f"{p.relative_to(PROJECT)}:{line_no}: {line}\n")
        md.append("```\n\n")

    md.append("## Source snippets around highest-value candidates\n\n")
    # Priority snippets: exact functions first, then high-scoring file hits.
    snippet_targets = []
    priority_regexes = [
        r"def inpaint",
        r"def sample_p_zs_given_zt",
        r"def sample_p_xh_given_z0",
        r"def sample_chain",
        r"def sample",
    ]

    for p in files:
        lines = read(p).splitlines()
        for i, line in enumerate(lines, 1):
            for pat in priority_regexes:
                if re.search(pat, line):
                    snippet_targets.append((p, i, line.strip()))
                    break

    # Add first hits for mask and frame in top candidate files.
    for c in candidates[:8]:
        p = PROJECT / c["path"]
        for group in ["inpainting_mask", "coordinate_frame", "ligand_update"]:
            hits = line_hits(p, KEY_PATTERNS[group])
            if hits:
                snippet_targets.append((p, hits[0][0], f"{group}: {hits[0][2].strip()}"))

    seen = set()
    for p, line_no, label in snippet_targets:
        key = (p, line_no)
        if key in seen:
            continue
        seen.add(key)
        md.append(f"### `{p.relative_to(PROJECT)}` around line {line_no}: `{label}`\n\n")
        md.append("```python\n")
        md.append(snippet(p, line_no))
        md.append("\n```\n\n")

    md.append("## Preflight decision template\n\n")
    md.append("Fill this after reading the snippets above:\n\n")
    md.append("| Question | Answer |\n")
    md.append("|---|---|\n")
    md.append("| Real denoising loop file/function | TBD from snippets |\n")
    md.append("| Line where ligand tensor is updated | TBD |\n")
    md.append("| Line where fixed/inpainting mask is reapplied | TBD |\n")
    md.append("| Coordinate frame inside loop | TBD: ligand-centered / pocket-centered / other |\n")
    md.append("| Required transform for B | TBD |\n")
    md.append("| Safe insertion point for global guidance | TBD |\n")
    md.append("| Rule for fixed atoms | Guidance must apply only to non-fixed ligand nodes; A mask must have final overwrite |\n")
    md.append("| Code modification status | None; this is a read-only preflight |\n\n")

    md.append("## Initial implementation principle\n\n")
    md.append("The global term should be inserted only after this report identifies the update and fixed-mask order. The guide must not alter fixed nodes. If the code reimposes `lig_fixed` after each denoising step, the guide should operate before that overwrite or explicitly mask out fixed nodes.\n")

    REPORT.write_text("".join(md))

    print(f"WROTE_REPORT={REPORT}")
    print(f"WROTE_JSON={JSON_OUT}")
    print("Top candidates:")
    for c in candidates[:10]:
        print(c["score"], c["path"], c["hits"])


if __name__ == "__main__":
    main()
