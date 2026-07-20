# Evolutionary optimization of staged growth

The variable-increment campaign treated the number of atoms requested at each
stage as a discrete decision variable. At every live search state, candidates
were generated under all increments

\[
\mathcal{A}_{n}=\{1,2,3,4,5\}.
\]

Ten samples were drawn for every increment. Candidates generated from all live
parents were audited jointly. Two variants were evaluated: greedy search, with
beam width \(k=1\), and beam search, with beam width \(k=3\). Both variants
shared the candidate-generation procedure, structural filters, progression
gate, target size, ranking function, and stopping rules; beam width was the only
intended algorithmic difference.

For parent scaffold \(p\), candidate child \(c\), Shape Tanimoto distance
\(d_T\), and directional Shape Protrude distance \(d_P\), the local gains were

\[
g_T=d_T(p,B)-d_T(c,B),\qquad
g_P=d_P(p,B)-d_P(c,B).
\]

A candidate passed the trade-off gate when

\[
g_T\geq-\varepsilon,\qquad
g_P\geq-\varepsilon,\qquad
g_T+g_P>0,
\]

with \(\varepsilon=0.01\). The tolerance allowed one shape metric to worsen by
at most 0.01 only when the other improved sufficiently to produce a positive
net gain. The gate was parent-relative, but passing candidates were ranked by
absolute state quality,

\[
Q(c)=-\left[d_T(c,B)+d_P(c,B)\right].
\]

Candidates were required to contain a valid warhead-containing anchor
component, contain at least one newly generated heavy atom connected to that
anchor, have a larger anchor-component heavy-atom count than the parent, and
retain every heavy atom of the parent scaffold within 0.5 Å in the child anchor
component.

For beam search, structural diversity was imposed using the symmetric Chamfer
distance between the coordinates of newly grown heavy atoms, excluding the
seven original warhead atoms. No alignment was applied because all candidates
shared the receptor coordinate frame and fixed warhead. A candidate was kept
only when its Chamfer distance from every previously retained state exceeded
0.75 Å.

Target anchor sizes were 16 heavy atoms for x0434-to-x2193 and 19 for
x0874-to-x1093. A candidate graduated when its anchor size lay within one heavy
atom of the target. Candidates exceeding the upper bound were classified as
oversize and were not propagated. `component_graduated` recorded target-size
attainment by the anchor component; `strictly_connected_graduated` additionally
required the complete generated record to contain a single heavy-atom
component.

A trajectory ended when all search states had graduated or died, or after a
maximum of eight stages. The campaign contained two ligand pairs, ten seeds,
and two beam widths, for 40 trajectories.
