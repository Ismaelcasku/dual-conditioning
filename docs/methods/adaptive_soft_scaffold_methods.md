# Adaptive soft-scaffold inpainting and beam search

## Hard, soft, and free atom partitions

The fixed-increment staged-growth experiments identified an addition of four atoms per stage as the most effective granularity under the tested protocol. The subsequent adaptive-flexibility experiment therefore held the requested increment constant at \(n_{\mathrm{add}}=4\) and replaced increment size with a stage-wise scaffold-rigidity action.

At stage \(s\), ligand atoms were partitioned into three disjoint sets. The seven-heavy-atom warhead inherited from ligand A formed the hard set \(M_{\mathrm{H}}\). Atoms generated and retained during earlier stages formed the soft parent-scaffold set \(M_{\mathrm{S}}\). Newly generated atoms formed the free set \(M_{\mathrm{U}}\). The complete supplied parent scaffold was

\[
M_{\mathrm{F}} = M_{\mathrm{H}} + M_{\mathrm{S}},
\]

and the free mask was

\[
M_{\mathrm{U}} = 1-M_{\mathrm{F}}.
\]

The hard warhead remained fully inpainted throughout sampling. Soft parent atoms retained their atom-feature channels but their coordinates were only partially reintroduced, allowing previously generated geometry to move without changing the inherited atom identities.

## Soft-scaffold reinjection

Let \(z_t^{\mathrm{known}}=(x_t^{\mathrm{known}},h_t^{\mathrm{known}})\) denote the appropriately noised representation of the supplied parent scaffold at reverse-diffusion step \(t\), and let \(z_t^{\mathrm{model}}=(x_t^{\mathrm{model}},h_t^{\mathrm{model}})\) denote the model state after the corresponding reverse transition. A scalar rigidity coefficient \(\rho_s\in[0,1]\) controlled coordinate reinjection for the soft parent atoms:

\[
x_t =
M_{\mathrm{H}}\odot x_t^{\mathrm{known}}
+
M_{\mathrm{S}}\odot
\left[
\rho_s x_t^{\mathrm{known}}
+
(1-\rho_s)x_t^{\mathrm{model}}
\right]
+
M_{\mathrm{U}}\odot x_t^{\mathrm{model}}.
\]

Atom-feature channels were combined as

\[
h_t =
(M_{\mathrm{H}}+M_{\mathrm{S}})\odot h_t^{\mathrm{known}}
+
M_{\mathrm{U}}\odot h_t^{\mathrm{model}}.
\]

Thus, \(\rho_s=1\) reproduced conventional hard inpainting of the complete parent scaffold, whereas \(\rho_s=0\) released the coordinates of the soft scaffold completely while preserving its atom types. Intermediate values produced partial coordinate retention. The seven warhead atoms were always assigned \(\rho=1\), independently of the stage action.

The \(\rho=1\) branch retained the original hard-inpainting execution path. For \(\rho<1\), mixing hard, soft, and free subsets can introduce a residual translation into the ligand state. After each mixed reinjection, the per-sample ligand coordinate mean

\[
\mu_t =
\frac{1}{N_{\mathrm{lig}}}
\sum_{i=1}^{N_{\mathrm{lig}}}x_{t,i}
\]

was subtracted from all ligand coordinates. The same translation was applied to the corresponding pocket coordinates:

\[
x_{t,i}^{\mathrm{lig}}\leftarrow x_{t,i}^{\mathrm{lig}}-\mu_t,
\qquad
x_{t,j}^{\mathrm{pocket}}\leftarrow x_{t,j}^{\mathrm{pocket}}-\mu_t.
\]

This restored the zero-center-of-mass invariant required by the DiffSBDD reverse process without altering internal ligand distances, internal pocket distances, or ligand–pocket relative geometry.

## Order-safe scaffold propagation

DiffSBDD and the subsequent RDKit graph-construction steps do not guarantee preservation of input atom indices. Parent identity was therefore recovered using a one-to-one Hungarian assignment constrained by atom type and coordinate distance. The seven hard atoms were required to match their reference positions within 0.2 Å. No displacement ceiling was imposed on the soft parent atoms because their movement was the experimental variable.

After assignment, every retained child was renumbered as

\[
[\text{hard warhead}]
+
[\text{soft parent in parent order}]
+
[\text{new atoms}].
\]

This canonical ordering was applied both to the generated coordinate tensor and to the processed RDKit molecule before writing the scaffold supplied to the next stage. Parent atom types were required to remain unchanged.

## Adaptive action search

Each trajectory began from the bare seven-heavy-atom warhead. Because no soft atoms existed at the first stage, stage 1 was sampled only with \(\rho_1=1\). At every later stage, the action set was

\[
\mathcal{A}_{\rho}
=
\{1.00,0.75,0.50,0.25,0.00\}.
\]

For each live parent and each action, ten samples were generated with \(n_{\mathrm{add}}=4\). All \(\rho\) values evaluated from the same parent at the same stage used the same random seed, providing paired diffusion noise across rigidity actions. Sampling otherwise used the frozen model and previously established settings: shape-guidance strength \(\lambda=20\), Gaussian width \(\alpha=0.3\), 50 reverse-diffusion timesteps, five resampling iterations, and a per-atom guidance clipping threshold of 1.0 Å. No model parameters were updated.

Two search variants were compared. Greedy search retained one continuation per stage (\(k=1\)); beam search retained up to three diverse continuations (\(k=3\)). Both variants used identical generation, filtering, ranking, and stopping rules, isolating the effect of beam width.

## Anchor-component filtering and parent retention

Generated records were sanitized when possible and decomposed into connected components. The anchor component was the component containing all seven matched warhead atoms. A candidate was rejected when the warhead could not be matched, exceeded the 0.2 Å tolerance, or was split across components.

For stages after the first, all inherited parent atoms were required to remain in the anchor component with unchanged atom types. The anchor child was additionally required to contain more heavy atoms than its parent:

\[
H(C_{s+1}) > H(C_s).
\]

Consequently, at least one newly generated atom had to become connected to the propagated anchor. Secondary components were discarded. Full-record connectivity was retained as a separate observable and was not required for continuation, because a fragmented raw record could still contain a valid, enlarged anchor component.

## Shape gate, beam ranking, and diversity

Shape was evaluated on the anchor component using RDKit Shape Tanimoto distance \(d_{\mathrm{T}}\) and directional Shape Protrude distance \(d_{\mathrm{P}}\), with `allowReordering=False`. Lower values indicated greater similarity to ligand B.

For a candidate child \(c\) and parent \(p\), local gains were

\[
g_{\mathrm{T}}=d_{\mathrm{T}}(p,B)-d_{\mathrm{T}}(c,B),
\qquad
g_{\mathrm{P}}=d_{\mathrm{P}}(p,B)-d_{\mathrm{P}}(c,B).
\]

After the initial stage, a candidate passed the trade-off gate when

\[
g_{\mathrm{T}}\geq-\varepsilon,
\qquad
g_{\mathrm{P}}\geq-\varepsilon,
\qquad
g_{\mathrm{T}}+g_{\mathrm{P}}>0,
\]

with \(\varepsilon=0.01\). This allowed a small deterioration in one metric only when compensated by a larger improvement in the other. The initial stage had no parent shape value and was filtered by structural validity and anchor growth alone.

Passing candidates were ranked by absolute state quality,

\[
Q(c) =
-\left[
d_{\mathrm{T}}(c,B)+d_{\mathrm{P}}(c,B)
\right],
\]

rather than by local gain. To prevent beam collapse onto near-duplicate states, selected candidates were required to differ by more than 0.75 Å under symmetric Chamfer distance calculated on the newly grown heavy-atom coordinates without structural alignment.

## Graduation and recorded outcomes

The hard pair x0434-to-x2193 and the moderate pair x0874-to-x1093 had target heavy-atom counts of 16 and 19, respectively. An anchor graduated when its heavy-atom count lay within one atom of the corresponding target. Anchors exceeding the upper tolerance were rejected as oversize and were not propagated.

Two endpoint definitions were retained. A component-graduated endpoint reached the target size in the anchor component. A strictly connected endpoint additionally required the complete generated record to contain one heavy-atom component.

For every selected transition, the recorded quantities included the selected \(\rho\), cumulative \(\rho\) schedule, anchor heavy-atom count, full-record connectivity, number of heavy-atom components, warhead RMSD, complete-parent RMSD, soft-scaffold RMSD, maximum parent-atom displacement, Shape Tanimoto distance, Shape Protrude distance, local gain, cumulative gain, and graduation status.

The final campaign comprised two ligand pairs, ten random seeds, and two search variants, yielding 40 independent trajectories. Runs were checkpointed after every completed stage and could terminate by graduation, loss of all valid continuations, or the maximum of eight stages.
