# Edge scorer: independent BCE -> parent-softmax normalization

## Motivation

Before building a heavier attention-based edge scorer (candidate scores
depending on neighboring candidates), checked the sibling repo's
`reports/` for prior evidence, since its old architecture already had
exactly that (`EdgeHead`: self-attention within a frame's node set, then
cross-attention between frames, then a pairwise MLP). Findings:

- The attention mechanism itself was never cleanly measured in isolation --
  the one available before/after (`2026-08-12-tracknet-first-run.md`,
  `2026-08-15-tracknet-submission-diagnosis.md`) compared the joint
  attention model (0.735 leaderboard) against an older, much simpler
  coordinate-only separate-model baseline (0.769) -- confounded by more than
  just attention.
- `2026-08-25-crowded-ranking-margin.md`: added a hardest-negative margin
  loss on top of the (attention-based) edge head specifically to fix
  crowded-region ranking. Rank-1 accuracy didn't move at all. Conclusion:
  the head lacked *information* in crowded neighborhoods, not discrimination
  pressure -- and v2's `UNet3D` already has `TemporalAttention3d` at every
  encoder stage (unlike the sibling's single 1x1x1-conv bottleneck), so that
  specific gap may already be closed here.
- `2026-08-25-edge-score-calibration.md`: **the one clean, positive
  result.** Removing `parent_softmax` (each edge's probability normalized
  against other candidates competing for the same target node) in favor of
  independent per-pair scoring lost 70% of true edges vs. 3% for the
  normalized version.

Conclusion: try the cheap, evidenced change (normalization) before the
expensive, unproven one (attention). v2's current `EdgeScorer` scores every
candidate pair fully independently -- no such normalization existed here at
all.

## Change

`losses.py`:
- Added `parent_softmax(logits, dst_idx, n_dst)`, a segmented softmax over
  candidate edges grouped by target (destination) node -- adapted from the
  sibling repo's function of the same name.
- Replaced `edge_loss`'s independent weighted-BCE (`EDGE_NEG_ALPHA`
  down-weighting) with BCE over the normalized probabilities. A target with
  no true parent among its candidates (parent undetected this frame, or the
  target genuinely starts a new track) can't be represented by a softmax
  that must sum to 1 while every candidate is pushed toward 0 -- that's a
  contradiction, not a hard negative, so such targets are masked out of the
  loss entirely (mask = "this target's candidate group contains at least
  one positive label").
- `EDGE_NEG_ALPHA` removed from `config.py` -- imbalance is handled
  structurally by the mask + per-target softmax now, not by a global
  negative-class weight; v2's candidate groups are small (`LINK_RADIUS_UM`
  gates them), so the extreme-imbalance justification for a 0.05 weight no
  longer applies the way it did for independent scoring.

`edge_train.py`: `sample_edge_pair` now also returns `dst_idx`/`n_dst` so
the loss can group candidates by target.

`train.py`: `compute_edge_loss`'s reported accuracy changed from a raw
`>=0.5` threshold accuracy to rank-1 accuracy (does the highest-probability
candidate match the true parent, among targets that have one) -- this is
what `link.py`'s greedy selection actually consumes, and a `>=0.5` cutoff
isn't meaningful against softmax-normalized probabilities (a target with
several competing candidates can have a correct top pick well under 0.5).
Reported as pooled correct/total counts across the whole epoch rather than
an average of per-step ratios, since a single sampled frame pair often has
zero labeled targets early in training and a naive per-step average would
let that NaN out the whole epoch's number.

`predict.py`: inference now applies `parent_softmax` to the edge scorer's
logits (grouped by destination among each frame pair's real detections)
before `link.link_frames` thresholds them, matching how the model was
supervised. Using raw `sigmoid(logits)` at inference, as before, would be a
train/inference mismatch once training targets normalized probabilities
instead of independent ones.

## Verified but not yet measured

Ran a 4-volume/8-epoch local smoke train (`scripts/train.py --limit-volumes
4 --epochs 8 --frames-per-volume 15 --edge-every 1`) and a 3-frame local
predict (`scripts/predict.py --volume 44b6_0113de3b --t-max 3`) against the
resulting checkpoint. Both complete without error; `edge_acc` (now
pooled rank-1 accuracy) reports sensible non-degenerate numbers
(0.14-0.57) once the detector starts producing enough detections for edge
supervision to activate, matching the known bootstrapping behavior
documented in `edge_train.py`. This is a plumbing check, not a result --
4 volumes / 8 epochs is far too small to show whether the change helps.

**Still open**: no held-out (`score_local.py --held-out`) or leaderboard
comparison yet -- needs a real training run on the full 179-volume split.
Also open: `LINK_SCORE_THRESHOLD = 0.5` was calibrated against the old
independent-sigmoid scores. Parent-softmax probabilities aren't
directly comparable across targets with different candidate-set sizes (an
isolated target with one candidate always gets probability 1.0; a strongly-
correct pick among several competitors can legitimately sit well under
0.5), so the fixed threshold may now reject correct crowded-region links
that used to pass, or accept isolated weak links that used to fail. Worth
checking whether `LINK_SCORE_THRESHOLD` needs its own retuning once a real
held-out number is in hand -- not changed yet, to keep this one variable at
a time.

## Full-split training run (Kaggle, 2026-08-28)

Ran the standard `kaggle_train.ipynb` recipe (`epochs=90`, `lr=5e-4`,
`weight_decay=1e-4` default, `val_frames_per_volume=16`, `patience=15`) on
the repackaged code (this change plus the two purely-additive diagnostic
commits already on top of the 0.841 checkpoint's code state -- see the
conversation record for the exact diff accounting).

- Best checkpoint: epoch 24, `val_loss=0.01582`. The 0.841 checkpoint's run
  bottomed out at `val_loss=0.01879` (per
  `2026-08-25-four-sample-solution-features.md`) -- a ~16% relative
  improvement, and a same-architecture/different-config comparison, which
  this repo's own metric guidance treats as trustworthy (unlike
  cross-checkpoint comparisons).
- `edge_acc` (pooled rank-1 accuracy, the new metric) held stably in
  0.94-0.97 for the whole run. Not comparable to any prior number -- the
  old `edge_acc` was `>=0.5` threshold accuracy on independent scores, a
  different quantity -- so this is a baseline, not a demonstrated delta.
- Still hard-diverged to NaN, at epoch 34 batch 876/895 -- later than any
  prior run in this thread (0.841 config recurred at epoch 26; the
  weight-decay test at epoch 22), but this is n=1 against a divergence
  epoch that has ranged 9-34 across unrelated runs already, so not claiming
  this change delayed or fixed the divergence.
- `|W|_conv` growth to the last full epoch (34): 31.44 -> 60.45, ~92%.
  Growth at epoch 19 specifically (the horizon used for prior comparisons):
  ~50%, somewhat lower than the ~60-62% seen in earlier runs, but not a
  strong claim -- edge-loss gradients flow back into the shared trunk via
  feature sampling, so some interaction is plausible, but this is one run.

**Still missing the number that actually matters**: no `score_local.py
--held-out` or leaderboard `SCORE` yet for this checkpoint. The 0.841
checkpoint never had a local held-out `SCORE` recorded either (table in
`2026-08-25-four-sample-solution-features.md` marks it "not separately
measured") -- only the leaderboard number. Next step: download
`detector_best.pt` (epoch 24) and run local held-out scoring and/or submit,
to get an actual apples-to-apples comparison against 0.841 rather than
inferring from `val_loss` alone.

## Result: regression (2026-08-29)

Submitted the epoch-24 checkpoint. **Official leaderboard: 0.788**, down
from 0.841 -- a real regression, not a local-metric artifact: leaderboard
numbers aren't subject to this repo's cross-checkpoint local-`SCORE`
unreliability caveat (`metric.py`'s docstring), so this is a trustworthy
same-architecture, one-variable-changed comparison. `val_loss` and
training-time rank-1 accuracy both improved; the actual linking outcome got
worse. Recorded local held-out numbers for the record (`--geff-dir
dist/preds_val_t05 --held-out dist/models/detector_best.pt`, 20 volumes,
built via `scripts/predict.py --link-score-threshold`, added this session):

- `SCORE = 0.7486` at `link_score_threshold=0.5` (the submitted config).
  Detection recall 94.6% within 7 um, node ratio 1.054x -- detection stage
  looks healthy; the regression is downstream of it.

Two candidate mechanisms were checked and ruled out:

1. **Threshold miscalibration** (predicted before the run: parent-softmax
   probabilities aren't comparable across differently-sized candidate
   groups, so the fixed 0.5 cutoff might reject correct crowded-region
   top-picks scoring under 0.5). Tested by dropping the threshold to 0.0
   (radius-gating only): `SCORE` got *worse*, not better (0.7486 -> 0.7381;
   TP 11565->11624, FP 1587->1884). Loosening the filter admits more noise
   than it recovers signal. **Refuted.**
2. **Singleton-candidate-group degeneracy** (a group of size 1 always
   softmaxes to probability 1.0 regardless of correctness, so isolated
   spurious detections could be passing unconditionally). Measured the
   actual candidate-group-size distribution on 4 held-out volumes at
   `LINK_RADIUS_UM=15`: only 1.8% of groups are singletons; median group
   size is 6. Too small a population to account for a 0.05 leaderboard
   drop. **Refuted.**

No confirmed root cause for the regression. The sibling repo's evidence for
`parent_softmax` (removing it lost 70% of true edges there) does not
transfer to this codebase's setup -- plausibly because the two candidate
populations differ (v2's independent-BCE baseline was already at 0.841,
not obviously broken the way the sibling's pre-normalization baseline was),
but that's speculation, not a measured explanation.

**Decision**: revert to independent BCE (the 0.841 config) rather than
continue chasing an unconfirmed mechanism. `losses.edge_loss`'s
parent-softmax normalization is a real, evidenced idea elsewhere, but it
does not help here on the metric that actually matters, and improving
`val_loss`/rank-1 while leaderboard score drops is exactly the kind of
local-metric/leaderboard divergence this repo's `metric.py` already warns
about. Reverted in `ed782fb`.

## Follow-up diagnosis (2026-08-29)

Before considering an attention-based edge scorer (a strict superset of
this same "make scores context-dependent" risk, plus more architectural/
stability surface on top of the still-unresolved NaN-divergence saga), dug
into *why* parent-softmax regressed, using a genuine apples-to-apples local
comparison this run enabled: `detector_best.zip` in Downloads (epoch 23,
`val_loss=0.018794`, matching the report's numbers exactly) turned out to
be the actual 0.841 checkpoint, on the identical 20-volume held-out split.
Scored both checkpoints locally with the (now-reverted) independent-BCE
inference code:

| checkpoint | local `SCORE` | TP | FP | FN | node ratio |
|---|---|---|---|---|---|
| 0841 (independent BCE, epoch 23) | 0.8031 | 11867 | 1036 | 1825 | 0.994x |
| parent-softmax (epoch 24) | 0.7486 | 11565 | 1587 | 2127 | 1.054x |

The local `SCORE` drop (0.8031 -> 0.7486) tracks the real leaderboard drop
(0.841 -> 0.788) in both direction and rough magnitude -- for this specific
same-architecture comparison, the local held-out metric turned out to be
trustworthy despite this repo's general cross-checkpoint caveat. Both FP
(+53%) and FN (+16%) got worse simultaneously, not a precision/recall
trade -- a genuine quality drop, not a threshold-shape issue.

Two more mechanisms checked, both also refuted:

3. **Forced confident guesses in "unresolvable" groups.** Training's
   `edge_loss` masks out any target whose true parent isn't among its
   candidates (undetected, or a genuine track start) -- reasonable in
   isolation (softmax can't represent "none of these"), but it means those
   groups get zero training signal, so their post-softmax probabilities at
   inference are uncalibrated noise that softmax still forces to sum to 1,
   potentially manufacturing a confident-looking wrong pick where
   independent BCE could correctly push every candidate below threshold.
   Classified every contradicted false-positive edge (both checkpoints,
   all 20 volumes) into "true parent was detected & in-radius" (a ranking
   failure) vs. "true parent wasn't even a candidate" (a forced guess).
   Result: nearly identical mix in both checkpoints (~35% / ~14%,
   independent-BCE and parent-softmax alike) -- parent-softmax has more
   errors of *both* kinds, not a disproportionate share of forced-guess
   errors. **Refuted** as the dominant mechanism.
4. **Crowding.** The regression turned out *not* to be uniform across
   volumes: per-volume `adjusted_edge_jaccard` delta shows 13 of 15
   `6bba_*` volumes got worse (some substantially, e.g. `6bba_c27cba08`
   -0.174) while `44b6_*` volumes were mostly flat or improved. Checked
   whether `6bba_*` volumes are simply more crowded (matching the sibling's
   own crowding-vs-edge-accuracy finding) -- they're actually *less*
   crowded on average (mean candidate-group-size 4.36 vs. 6.60 for
   `44b6_*`). **Refuted**, and in the wrong direction.

**Status**: real, reproducible, dataset-split-concentrated regression with
no confirmed mechanism after four ruled-out hypotheses (threshold
calibration, singleton-group degeneracy, forced-guess masking, crowding).
The remaining lead -- something specific to the `6bba_*` volumes that
parent-softmax handles worse than independent BCE -- would need a
different kind of investigation (e.g. what else differs between the two
prefix groups: motion speed, cell morphology, imaging characteristics) than
more architecture changes. Not pursued further this session; recommending
against building the attention-based edge scorer next, since it inherits
the same "scores depend on neighbors" risk this investigation couldn't
fully explain, on top of new stability surface.
