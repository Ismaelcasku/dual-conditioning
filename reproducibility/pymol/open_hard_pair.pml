reinitialize

load data/structural_examples/02_hard_pair_relaxation/reference_A_x0434.sdf, refA
load data/structural_examples/02_hard_pair_relaxation/reference_B_x2193.sdf, refB
load data/structural_examples/02_hard_pair_relaxation/prepared_ligand.sdf, before
load data/structural_examples/02_hard_pair_relaxation/minimized_ligand.sdf, after
load data/structural_examples/02_hard_pair_relaxation/minimized_protein_noH.pdb, protein

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

mkdir sessions
save results/pymol_sessions/02_hard_pair_basic.pse
