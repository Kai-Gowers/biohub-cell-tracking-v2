# Edge scorer: dedicated wider feature neck

## Motivation

Follows the parent-softmax negative result
(`2026-08-28-parent-softmax-edge-loss.md`): making edge scores depend on
*other candidates* (neighbor-context) regressed the leaderboard score
(0.841 -> 0.788) for reasons that resisted four separate diagnostic
attempts. Rather than continue down that path (an attention-based edge
scorer would be a strict superset of the same risk, plus more architectural
surface interacting with the still-unresolved NaN-divergence investigation),
this tries the other lever: give the edge scorer a richer view of *each
node itself*, without touching how candidates interact with each other.

The bottleneck was real and easy to name: `EDGE_FEATURE_DIM` was tied
directly to `UNET_BASE_CHANNELS=16`, and `EdgeScorer` sampled its node
embeddings from the exact same tensor `out_proj` reads to produce the
detection logit -- a 16-channel bottleneck shared between two different
tasks (precise per-voxel localization vs. pairwise relationship scoring).

## Change

`models/detector.py`'s `UNet3D` gets one new module, `edge_neck`: a single
`Conv3d(UNET_BASE_CHANNELS, EDGE_FEATURE_DIM, 3) -> InstanceNorm3d -> GELU`
branching off the same finest-decoder-stage output `out_proj` reads.
`out_proj` is untouched -- detection logits are bit-identical whether or
not `return_features=True`, for a FIXED set of trunk weights (verified via
synthetic tensors). `return_features=True` now returns `edge_neck`'s wider
output instead of the raw finest-decoder features, so `EdgeScorer`
(`models/edge_model.py`) gets `EDGE_FEATURE_DIM` channels per node instead
of 16.

`config.py`: `EDGE_FEATURE_DIM` decoupled from `UNET_BASE_CHANNELS`, set to
64 (matching `EDGE_HIDDEN_DIM`). No dropout on `edge_neck`, for the same
sub-voxel-precision reasoning the finest decoder stage itself already
skips it.

`train.py`: `model.edge_neck` added to the `conv_params` grouping used by
the per-component gradient/weight-norm diagnostics.

Adds 27,776 params (1.83% of the model's 1.52M total).

**Breaking change, expected**: existing checkpoints (the 0.841 baseline,
the parent-softmax run) do not have `edge_neck` weights and will fail to
load (`strict=True` `load_state_dict`) against this architecture.

## Verified but not yet measured (plumbing)

- `UNet3D.forward` with `return_features=True` vs. `False`: logits
  bit-identical (`torch.allclose`) in eval mode across a synthetic batch,
  for a fixed set of trunk weights.
- 4-volume/8-epoch local smoke train and a 3-frame local predict both run
  clean end to end.

## Result: regression, smaller than parent-softmax's (2026-08-29)

Full training run (epochs=90, same recipe as every other run in this
thread) stopped via **patience** at epoch 21 -- no improvement in 15
epochs, best at epoch 6 (`val_loss=0.018884`). First run in this whole
NaN-divergence-chasing thread to stop cleanly rather than diverging; one
data point, not claiming `edge_neck` fixed or caused that either way given
how much the divergence epoch has varied across unrelated runs already
(9, 22, 23, 26, 34).

Initially assumed `val_loss` couldn't be informative here, since `out_proj`
reads the same input with or without `edge_neck` -- **that reasoning was
incomplete**. It holds only for a *fixed* trunk. In joint training,
`edge_neck` sits between the shared trunk and the edge loss, so it changes
what gradient signal reaches that shared trunk during training, even
though the forward computation for detection alone is provably identical
for any given set of weights. The early plateau (best at epoch 6 vs. the
0.841 baseline steadily improving to epoch 23) was plausibly a real
symptom, not noise.

Scored locally against the same 20-volume held-out split used for every
comparison in this thread:

| checkpoint | local `SCORE` | TP | FP | FN | node ratio |
|---|---|---|---|---|---|
| 0841 (independent BCE, epoch 23) | 0.8031 | 11867 | 1036 | 1825 | 0.994x |
| parent-softmax (epoch 24) | 0.7486 | 11565 | 1587 | 2127 | 1.054x |
| edge_neck (epoch 6) | 0.7801 | 11812 | 1334 | 1880 | 1.039x |

`edge_neck` sits between the other two -- a real regression from baseline
(-0.023), smaller than parent-softmax's (-0.055) but the same direction.

## Conclusion

Two different, independent levers on the edge scorer -- parent-softmax
(neighbor-dependent scoring) and this feature neck (more per-node
capacity) -- have now both regressed the held-out score relative to the
0.841 baseline. That pattern is more informative than either result alone:
it argues against "the edge scorer lacks context" or "the edge scorer
lacks capacity" as the actual bottleneck, and toward the joint
detector+edge-scorer training itself being sensitive to perturbation right
now -- plausibly entangled with the still-unresolved recurring
NaN-divergence instability this repo has been chasing since
`2026-08-26`. Every run in this thread, regardless of edge-scorer variant,
diverges or plateaus at a different, hard-to-predict epoch (6, 9, 21, 22,
23, 24, 26, 34) -- that variance makes it hard to cleanly attribute a
result to any one architectural change while it's unresolved.

**Recommendation**: revert `edge_neck`, back to the 0.841 baseline. Before
trying a third edge-scorer variant, root-causing the training-stability
issue itself is likely higher-value than continuing to probe the edge
scorer's information/capacity along axes that have now both come back
negative. All three checkpoints and their held-out predictions are kept
locally (`dist/models/detector_best_0841.pt`,
`detector_best_parentsoftmax_epoch24.pt`, `detector_best_edgeneck.pt`;
`dist/score_0841.json`, `dist/score_parentsoftmax.json`,
`dist/score_edgeneck.json`) for reference.
