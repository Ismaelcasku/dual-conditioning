# Render the four structural examples used in manuscript Figure 5.
# Run from the repository root: pymol -cq reproducibility/pymol/render_figure5_structural_examples.pml

# Shared rendering settings are repeated after each reinitialize.

# ------------------------------------------------------------------
# A. Compatible boundary case
# ------------------------------------------------------------------
reinitialize
load data/structural_examples/01_stable_dual/reference_A_x0434.sdf, refA
load data/structural_examples/01_stable_dual/reference_B_x1093.sdf, refB
load data/structural_examples/01_stable_dual/prepared_ligand.sdf, before
load data/structural_examples/01_stable_dual/minimized_ligand.sdf, after
hide everything, all
show sticks, refA or refB or before or after
color gray60, refA
color cyan, refB
color orange, before
color green, after
set stick_radius, 0.18
set stick_transparency, 0.45, refA
set stick_transparency, 0.30, refB
set stick_transparency, 0.20, before
bg_color white
set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set antialias, 2
set ray_opaque_background, off
orient refA or refB or before or after
zoom refA or refB or before or after, 3.0
ray 2400, 1800
png results/source_renderings/relaxation/01_stable_dual_overlay.png, dpi=600

# ------------------------------------------------------------------
# B. Difficult-pair attenuation
# ------------------------------------------------------------------
reinitialize
load data/structural_examples/02_hard_pair_relaxation/reference_A_x0434.sdf, refA
load data/structural_examples/02_hard_pair_relaxation/reference_B_x2193.sdf, refB
load data/structural_examples/02_hard_pair_relaxation/prepared_ligand.sdf, before
load data/structural_examples/02_hard_pair_relaxation/minimized_ligand.sdf, after
hide everything, all
show sticks, refA or refB or before or after
color gray60, refA
color cyan, refB
color orange, before
color green, after
set stick_radius, 0.18
set stick_transparency, 0.45, refA
set stick_transparency, 0.30, refB
set stick_transparency, 0.20, before
bg_color white
set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set antialias, 2
set ray_opaque_background, off
orient refA or refB or before or after
zoom refA or refB or before or after, 3.0
ray 2400, 1800
png results/source_renderings/relaxation/02_hard_pair_relaxation_overlay.png, dpi=600

# ------------------------------------------------------------------
# C. Global improvement with anchor drift
# ------------------------------------------------------------------
reinitialize
load data/structural_examples/04_local_condition_failure/reference_A_x0874.sdf, refA
load data/structural_examples/04_local_condition_failure/reference_B_x1093.sdf, refB
load data/structural_examples/04_local_condition_failure/prepared_ligand.sdf, before
load data/structural_examples/04_local_condition_failure/minimized_ligand.sdf, after
hide everything, all
show sticks, refA or refB or before or after
color gray60, refA
color cyan, refB
color orange, before
color green, after
set stick_radius, 0.18
set stick_transparency, 0.45, refA
set stick_transparency, 0.30, refB
set stick_transparency, 0.20, before
bg_color white
set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set antialias, 2
set ray_opaque_background, off
orient refA or refB or before or after
zoom refA or refB or before or after, 3.0
ray 2400, 1800
png results/source_renderings/relaxation/04_global_shape_anchor_drift_overlay.png, dpi=600

# ------------------------------------------------------------------
# D. Minimization-rescued pose
# ------------------------------------------------------------------
reinitialize
load data/structural_examples/03_minimization_rescued/reference_A_x0434.sdf, refA
load data/structural_examples/03_minimization_rescued/reference_B_x1093.sdf, refB
load data/structural_examples/03_minimization_rescued/prepared_ligand.sdf, before
load data/structural_examples/03_minimization_rescued/minimized_ligand.sdf, after
hide everything, all
show sticks, refA or refB or before or after
color gray60, refA
color cyan, refB
color orange, before
color green, after
set stick_radius, 0.18
set stick_transparency, 0.45, refA
set stick_transparency, 0.30, refB
set stick_transparency, 0.20, before
bg_color white
set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set antialias, 2
set ray_opaque_background, off
orient refA or refB or before or after
zoom refA or refB or before or after, 3.0
ray 2400, 1800
png results/source_renderings/relaxation/03_minimization_rescued_overlay.png, dpi=600

quit
