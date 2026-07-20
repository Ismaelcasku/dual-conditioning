# Level-B smoke test (requires GPU + DiffSBDD + checkpoint)

This verifies that the GENERATIVE pipeline is deterministic: a short directed
trajectory with a fixed seed must reproduce a reference output within tolerance.
Level A (verify_numbers.py, verify_figures.py) does NOT need this; it works from
the derived TSVs alone.

## What to provide (TODO, from the cluster)

1. `reference/` — a single short directed trajectory used as ground truth:
   - the exact command (pair, seed, add_n, n_stages, lambda) in `command.txt`
   - the resulting scaffold SDFs per stage
   - the per-stage shape TSV (tani_A, prot_A, tani_B, prot_B) as
     `reference_shape.tsv`
2. `tolerances.json` — acceptable absolute deltas for tani_B / prot_B per stage
   (shape metrics vary slightly across RDKit builds; suggest 1e-3).

## How the test will run (verify_smoke.py, stub below)

  1. setup_diffsbdd.sh  (clone pinned commit, apply patches)
  2. download checkpoint, verify SHA256 (see THIRD_PARTY.md)
  3. run the reference command with the pinned seed
  4. recompute shape with src/growth/recompute_shape_AB.py
     (allowReordering=False, the manuscript convention)
  5. compare to reference_shape.tsv within tolerances.json

Fill `reference/` and `tolerances.json`, then `verify_smoke.py` becomes runnable.
