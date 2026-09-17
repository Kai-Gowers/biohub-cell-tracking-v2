# Analysis: public 0.933 notebook (933-biohub-bidir30.ipynb)

Read-only analysis of a downloaded Kaggle notebook (not committed, not run) for
ideas to evaluate against our own held-out score. Describes techniques and
rationale, not verbatim code -- do not copy any of this notebook's code
directly into this repo.

## What it actually is

Important context before comparing scores directly: this is the tip of a long
public-notebook lineage, not a from-scratch solo effort. It's an explicit fork
of another public notebook (LB 0.931), which is itself presumably downstream
of others. The single change this specific version makes over its parent is
one swept blend-weight constant (+0.003). The 0.933 reflects a large amount of
accumulated infrastructure across many iterations: two independently-trained
models (different random seeds) ensembled at both detection and edge-scoring
level, a *third* auxiliary model used purely as a safety veto, a real ILP
solver, and roughly ten independent post-processing passes, each with its own
tunable thresholds and rate caps. This is a much bigger compute/engineering
footprint than this repo currently has -- worth calibrating expectations, not
just "why don't we do all of this."

## Architecture-level diff

| | Us (v2) | Them |
|---|---|---|
| Detector | Single 2-frame-window 3D U-Net, cross-frame attention | Same family (UNet + node-transformer, "unet_transformer") |
| Edge scoring | One learned edge scorer, forward-time only | Same model, but **also run backward in time** and fused with the forward pass |
| Ensembling | None (one checkpoint) | **Two independently-trained models** (different seeds), blended at both detection and edge level, with explicit mean/std recalibration before blending |
| Auxiliary models | None | A **third, separately-trained model** ("DeepCenter") used only as a yes/no confirmation gate before accepting a risky repair |
| Detection TTA | 4-way (identity + 3 flips, z excluded) | 8-way (adds 90/270 rotation, transpose, rotate+transpose -- full planar D4 symmetry, z still excluded) |
| Linking | Distance-gated greedy, in/out-degree <=1, no divisions | ILP (`tracksdata` + SCIP/`ilpy`) with explicit appearance/disappearance/division cost terms |
| Repair | None (this repo's own v1 attempt just measured net harmful, see `reports/2026-08-30-repair-v1-gap-rescue.md`) | ~10 composable passes: gap-close (density-adaptive, synthetic-node insertion with image-based refinement), gap2, motion-relink (learned-probability-weighted exact assignment), safe-division (multi-gate), short-track rescue, single-parent/child repair, isolated-node pruning, line-fit position smoothing |

## Ranked findings: what's transferable, in order of expected leverage

### 1. Density-adaptive distance gating -- directly explains why our repair v1 failed

Our gap-rescue widened `LINK_RADIUS_UM` uniformly (15um -> 20um) and it made
things worse (report: `2026-08-30-repair-v1-gap-rescue.md`, SCORE -0.02,
17.6 spurious links per real one recovered). This notebook's gap-closer never
uses a fixed radius: it computes each node's **local neighbor spacing**
(median distance to its k=3 nearest neighbors in-frame) and adjusts the
allowed gap distance up or down from a reference density, clamped to a small
max step. A node in a sparse region gets a wider tolerance; a node buried in
a crowded cluster gets a *tighter* one than the nominal baseline. That is
exactly the missing ingredient our negative result's writeup called out as
the untried lever -- not "widen the radius" but "the right radius depends on
how crowded this specific neighborhood is." This is the single highest-value
thing to evaluate next if repair gets revisited: same conservative shape,
add a density term, re-measure on the same held-out split before deciding
anything.

### 2. A cheap auxiliary "does this look like a real cell" veto changes what "obvious" means

Their riskiest repair operations (inserting a synthetic gap node, accepting a
safe division) are gated behind an independent, separately-trained "center
prior" model scoring a small patch at the proposed location/frame, purely as
an add-only confirmation -- if that second opinion disagrees, the repair is
discarded, no matter what the primary pipeline's own geometry/probability
said. This is a materially different kind of "obvious mistake" test than
anything we've tried: instead of one signal (distance, or edge probability)
gating a repair, it's **two independently-trained models that have to
agree**. We don't have a second model, so this isn't directly portable, but
the *shape* of the idea is: an "obvious" repair candidate should pass more
than one independent check, not just a tighter threshold on the same check
that already produced the candidate. Worth remembering next time a
single-signal repair measures as marginal rather than clearly harmful.

### 3. Safe division uses a conjunction of independent gates, not one threshold

CLAUDE.md/memory record division recovery as measuring inert in the sibling
repo. This notebook's version requires, simultaneously: an existing
single-child edge already present (only ever adds a *second* child, never
proposes a division from nothing), the new candidate is the existing child's
mutual nearest unclaimed neighbor (not just "within range"), and -- the
distinctive one -- **both putative daughters must have their own unambiguous
single successor one frame later, and those grandchildren must have moved
apart more than the daughters are apart now** (a forward-looking divergence
check, using a future frame as corroborating evidence a division is
biologically real rather than two nearby cells that happened to be close in
one frame). Plus per-frame and global rate caps. No single one of these gates
is exotic; the result comes from requiring all of them at once. If divisions
are revisited here, this is a concrete, evidence-shaped design to adapt
(small-scale, one gate at a time, measuring each) rather than the sibling
repo's single ILP division-weight, which had nothing to check against.

### 4. Motion-relink folds the learned edge probability into the geometry, not just distance

Their motion-relink is a two-pass (tight-radius then relaxed-radius) *exact*
bipartite assignment (`scipy.linear_sum_assignment`) per frame pair, same
shape as our own linker, but the cost is `motion-predicted distance + small
raw-distance term - bonus * learned_edge_probability`, where the predicted
position extrapolates from each source node's own previous step (a one-step
velocity model). The bonus term means a pair the trained edge scorer already
liked gets preferred over a slightly-closer pair it didn't -- geometry alone
never overrides the model. Our repair v1 used geometry only, no learned
signal at all (this repo's checkpoint already has a trained edge scorer we
didn't route through the rescue pass). That's a second concrete, cheap fix
to try before or alongside the density-adaptive radius: score rescue
candidates with the existing edge scorer, not just distance.

### 5. Free accuracy: extend detection TTA to full planar D4 symmetry

Currently `TTA_FLIPS` in `config.py` is 4-way (identity + 3 flips). They also
average in both 90/270-degree rotations, a transpose, and rotate+transpose --
8 views total, still excluding z for the same anisotropy reason we already
have documented. This costs proportionally more inference time (2x) for no
retraining and no architecture change, and is close to free to try and
measure on the held-out set -- the lowest-risk item on this list.

### 6. Bidirectional (reverse-time) edge scoring as a consistency filter

They run the *same* trained edge scorer in both directions per frame pair
(source->target and target->source) and fuse the two probability estimates
with a harmonic mean (which penalizes a candidate hard if either direction is
unconfident about it), after recalibrating the reverse pass's logit scale to
match the forward pass. This is a genuinely different idea from anything
above: it's a consistency check on the *existing* trained model's own
output, not a new model and not a geometric heuristic. It requires the model
to support running edges in reverse cleanly, which needs checking against
our `EdgeScorer`'s architecture (is it symmetric in source/target, or would
"reverse" need a second forward pass with swapped features -- probably yes,
mechanically similar to their patch). More invasive to try than TTA, but
conceptually clean and something our existing single checkpoint could
support without training anything new.

### 7. Real ensembling (two independently-trained seeds), with calibration -- expensive, not proposed now

Ensembling two models trained from different seeds, blended at both
detection and edge level with explicit mean/std recalibration before mixing,
and a "low-margin consensus" mode that only leans on the second model when
the first one is uncertain and they agree. This is real signal, but it means
training and maintaining a second full checkpoint -- a much bigger
commitment than anything else on this list, and this repo doesn't have
training throughput to spare given the ongoing bf16/epoch-time investigation.
Noting it as a known lever, not proposing it now.

### 8. Real ILP (appearance/disappearance/division costs) vs. our greedy linker

They use `tracksdata` + SCIP for a genuine multi-term ILP (not just exact
bipartite assignment). Given this repo's linker only does plain distance
gating with no divisions and the sibling repo already measured that, for a
no-division bipartite formulation, an ILP and `scipy`'s exact assignment
produce equal-objective results (`../biohub-cell-tracking/reports/2026-08-25-remove-ortools.md`),
an ILP only becomes worth its complexity once real appearance/disappearance/
division cost terms exist to trade off -- i.e., after divisions or a richer
cost model are added, not before. Not actionable on its own.

## Explicitly not recommending

- Don't add all ~10 of their repair passes at once -- that's the exact
  failure mode this repo's reset was built to avoid, and several of them
  (motion-relink in particular, per the sibling repo's own measurement) can
  improve the metric while making the underlying graph worse-connected. Pick
  one (density-adaptive gating, most likely), measure it alone against the
  held-out set, write it up, then decide the next one.
- Don't copy their code. Everything above is described by rationale/shape
  so it can be re-implemented independently, fit to this repo's own
  `TrackGraph`/`link.py`/`config.py` conventions.
