reinitialize

load data/structural_examples/04_local_condition_failure/reference_A_x0874.sdf, refA
load data/structural_examples/04_local_condition_failure/reference_B_x1093.sdf, refB
load data/structural_examples/04_local_condition_failure/prepared_ligand.sdf, before
load data/structural_examples/04_local_condition_failure/minimized_ligand.sdf, after
load data/structural_examples/04_local_condition_failure/minimized_protein_noH.pdb, protein

hide everything, all

show sticks, refA
show sticks, refB
show sticks, before
show sticks, after

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

orient refA or refB or before or after
zoom refA or refB or before or after, 3.0

scene local_condition_failure_overlay, store

save results/pymol_sessions/04_local_condition_failure_basic.pse
