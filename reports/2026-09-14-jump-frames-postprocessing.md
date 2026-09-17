# 2026-09-14 — Whole-frame jumps: the pair MLP fails them, the transformer fixes them, the notebook post-processing breaks them again

Follow-up to the PI's "global shift compensation" question (overview report, section 10; jump
inventory in `figures/2026-09-14-error-analysis-0842/motion_jumps.csv`). Held-out 20 videos.

## The failure mode (`main`, 0.842)

`44b6_d754aa59` t=51->52 (fig12, browser case #545): the whole field moves ~9 µm in -y. The source
cell has two equally bright nuclei ~9 µm away in t+1, one up (its true partner, scored 0.99) and one
down (scored 1.00). The pair scorer's inputs are `[feat_src, feat_dst, rel_um, dist]`
(`models/edge_model.py`): it sees the pair and the offset, never the neighbours, so it cannot know
which way the field moved. Greedy took 1.00. Nearest other detection 13 µm, so not a crowding error.

## Does the 0.942 pipeline fix it? Yes at the ILP stage, no after post-processing

Missed-link rate of every annotated link, by the frame pair's whole-frame image shift
(same held-out videos, same GT; per-link outcome from the metric's 7 µm matching):

| | shift < 3 µm (13,387 links) | 3-5 µm (255) | >= 5 µm (50) |
|---|---|---|---|
| main 0.842 (pair MLP + greedy) | 12.0% | 14.1% | 30.0% |
| 0.942, ILP stage only (transformer, no post-processing) | 4.4% | 6.3% | **10.0%** |
| 0.942, full (transformer + ILP + notebook post-processing) | 4.5% | 7.5% | **32.0%** |

- The node transformer (cross-attention over all detections of both frames) resolves the jump
  frames: case #545 is linked correctly at the ILP stage, as are `44b6_d754aa59` t=55 (15 µm jump)
  and all 16 links of `6bba_283bf9f1` t=91.
- The notebook post-processing undoes it: its motion relink *replaces every ILP edge* with a
  motion-model assignment capped at 5.5 µm (tight) / 10 µm (relaxed), and the edge sanity filter
  drops anything over 14 µm (`postprocess.PostprocessConfig`). Hops of 9-15 µm during a jump are
  therefore cut or re-routed; #545 becomes "undetected" (its node is removed by the short-track
  filter after the relink), t=55 becomes a wrong link.

## Net effect of the post-processing on our held-out set (shipped weights)

| adjusted edge Jaccard | total | 44b6 | 6bba | `44b6_d754aa59` |
|---|---|---|---|---|
| ILP stage only | 0.9182 | 0.8404 | 0.9276 | 0.975 |
| full pipeline | 0.9094 | 0.8314 | 0.9190 | 0.897 |

Post-processing costs 0.009 edge Jaccard and buys ~0.008 through the 0.1 x division term (it is the
only stage that emits divisions), so the competition score is a wash here (0.9175 full vs 0.9182
ILP-only), while on the one video with sustained motion it costs 0.078. Caveat: the shipped weights
were trained on all 199 videos including these 20, so absolute numbers are optimistic; the
relative ILP-vs-full comparison uses identical weights and is not affected.

## What to do with it (experiments, not yet run)

1. On `replicate-0942`, score our own checkpoints with `--stage ilp` vs full on the held-out set
   and on the jump videos (`44b6_d754aa59`, `6bba_3abfe10a`) specifically. If the pattern holds
   with non-leaky weights, the post-processing's motion relink needs to become motion-aware.
2. Cheapest fix to test: subtract the frame pair's median predicted displacement (from the ILP
   edges) before applying the 5.5 / 10 µm relink radii, or skip the relink for a frame pair whose
   median displacement exceeds ~4 µm. Keep the division step (it is where the division term comes
   from).
3. For `main`'s pair scorer the equivalent lever is a per-frame median-displacement feature, but
   `main` is superseded by the transformer, which already sees the neighbours.
