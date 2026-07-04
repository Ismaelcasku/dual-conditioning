reinitialize

load results/source_data/fragment_example/sample_2/fragment_01_parent.sdf, parent
load results/source_data/fragment_example/sample_2/fragment_02_secondary.sdf, secondary_1
load results/source_data/fragment_example/sample_2/fragment_03_secondary.sdf, secondary_2
load results/source_data/fragment_example/sample_2/fragment_04_secondary.sdf, secondary_3
load data/structural_examples/02_hard_pair_relaxation/reference_B_x2193.sdf, refB

hide everything, all

set orthoscopic, on
set depth_cue, off
set ray_shadows, off
set antialias, 2
set ray_opaque_background, off
set transparency_mode, 2

set ambient, 0.55
set direct, 0.45
set specular, 0.12
set shininess, 10

set stick_radius, 0.145

bg_color white

# Scientific colour scheme:
# parent component = orange
# disconnected secondary components = neutral grey
# global reference B = cyan
color orange, parent
color gray45, secondary_1
color gray60, secondary_2
color gray75, secondary_3
color cyan, refB

# Establish one common camera for both renderings.
orient parent or secondary_1 or secondary_2 or secondary_3 or refB
zoom parent or secondary_1 or secondary_2 or secondary_3 or refB, 2.4

# -------------------------------------------------------------
# Full disconnected record against B
# -------------------------------------------------------------
hide everything, all

show sticks, parent
show sticks, secondary_1
show sticks, secondary_2
show sticks, secondary_3
show sticks, refB

set stick_transparency, 0.00, parent
set stick_transparency, 0.05, secondary_1
set stick_transparency, 0.05, secondary_2
set stick_transparency, 0.05, secondary_3
set stick_transparency, 0.32, refB

ray 2400, 1800

png results/source_renderings/figure3_full_record_vs_B.png, dpi=600

# -------------------------------------------------------------
# Largest connected component against B
# -------------------------------------------------------------
hide everything, all

show sticks, parent
show sticks, refB

set stick_transparency, 0.00, parent
set stick_transparency, 0.32, refB

ray 2400, 1800

png results/source_renderings/figure3_parent_component_vs_B.png, dpi=600

save results/pymol_sessions/figure3_fragmentation_example.pse

quit
