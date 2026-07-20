# Campaigns, manifests, and SLURM mapping

| manuscript element | campaign | manifest builder | active SLURM launcher |
|---|---|---|---|
| Figure 1 connectivity reference | single-shot lambda sweep | `src/analysis/single_shot/make_single_shot_manifest.py` | `jobs/single_shot_lambda_sweep.slurm` |
| Figure 2 single-step sweep | independent local expansions | `src/growth/make_stage1_manifest.py` | `jobs/single_step_expansion.slurm` |
| Figures 3–5 | fixed-increment staged growth | `src/growth/make_campaign1_manifest.py` | `jobs/campaign1_fixed_increment.slurm` |
| Phase 4 extension | variable-increment greedy/beam search | `src/growth/make_campaign2_manifest.py` | `jobs/campaign2_beam.slurm` |
| Phase 5 extension | adaptive soft-scaffold rigidity | `src/growth/make_softmask_manifest.py` | `jobs/softmask_beam.slurm` |

## Fixed-increment campaign

- schedules: `+3 × 3`, `+4 × 3`, `+5 × 2`, and `+6 × 2` stages;
- ten base seeds;
- A1 and A10: three replicas per pair, seed, and schedule;
- directed branch: one trajectory per pair, seed, and schedule;
- 560 trajectories: 240 A1, 240 A10, and 80 directed.

## Phase 4 — variable-increment search

- two ligand pairs;
- ten seeds;
- greedy (`k=1`) and beam (`k=3`);
- requested increments `+1` to `+5` at every live state;
- ten generated candidates per increment;
- shared trade-off gate and absolute-quality ranking;
- 40 trajectories, manifest rows `0–39`.

Default output root:

```text
artifacts/phase4_evo_beam/
```

## Phase 5 — adaptive soft-scaffold rigidity

- two ligand pairs;
- ten seeds;
- greedy (`k=1`) and beam (`k=3`);
- fixed requested increment `add_n=4`;
- stage 1 action `rho=1`;
- later action set `rho={1, 0.75, 0.5, 0.25, 0}`;
- ten generated candidates per action;
- seven hard warhead atoms; previously generated parent atoms are soft;
- shared Phase-4 gate, ranking, diversity, target, and stopping rules;
- 40 trajectories, manifest rows `0–39`.

Default output root:

```text
artifacts/phase5_softmask/
```

Pilot manifest:

```bash
python src/growth/make_softmask_manifest.py \
  --pilot 1 \
  --out artifacts/phase5_softmask/softmask_manifest_pilot.tsv
```

Full manifest:

```bash
python src/growth/make_softmask_manifest.py \
  --out artifacts/phase5_softmask/softmask_manifest.tsv
```

The SLURM time limit applies independently to each array task. The launcher is
configured as `0-39%2`, requests three GPUs per task, and resumes from the last
completed stage when a nonterminal checkpoint is present.
