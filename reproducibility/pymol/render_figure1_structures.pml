reinitialize

load data/structural_examples/02_hard_pair_relaxation/reference_A_x0434.sdf, refA
load data/structural_examples/02_hard_pair_relaxation/reference_B_x2193.sdf, refB
load data/structural_examples/02_hard_pair_relaxation/generated_input.sdf, candidate

# PyMOL uses one-based atom indices.
select fixedA, refA and index 4+14+13+11+8+3+2
select fixedCandidate, candidate and index 1+6+7+8+9+10+11

hide everything, all

set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set antialias, 2
set ray_opaque_background, off
set stick_radius, 0.17
set sphere_scale, 0.32
set transparency_mode, 2

bg_color white

# Object colours
color gray60, refA
color cyan, refB
color orange, candidate

# Fixed anchor
color black, fixedA
color black, fixedCandidate

# Establish one common camera for both panels
orient refA or refB or candidate
zoom refA or refB or candidate, 2.5

# -------------------------------------------------------------
# Panel A: local reference A and global reference B
# -------------------------------------------------------------
hide everything, all

show sticks, refA
show sticks, refB
show spheres, fixedA

set stick_transparency, 0.05, refA
set stick_transparency, 0.35, refB
set sphere_transparency, 0.00, fixedA

ray 2400, 1800

png results/source_renderings/figure1_difficult_input_AB_anchor.png, dpi=600

# -------------------------------------------------------------
# Panel B: generated candidate and global reference B
# -------------------------------------------------------------
hide everything, all

show sticks, candidate
show sticks, refB
show spheres, fixedCandidate

set stick_transparency, 0.00, candidate
set stick_transparency, 0.35, refB
set sphere_transparency, 0.00, fixedCandidate

ray 2400, 1800

png results/source_renderings/figure1_difficult_generated_B_anchor.png, dpi=600

save results/pymol_sessions/figure1_difficult_pair.pse

quit
