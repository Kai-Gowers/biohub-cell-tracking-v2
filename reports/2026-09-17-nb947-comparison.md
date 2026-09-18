# 2026-09-17 -- What the 0.947-LB public notebook changes, and whether it helps here

`context/biohub-cell-tracking-0-947-lb.ipynb` (untracked, 538 KB) documents a chain
0.933 -> 0.934 -> 0.939 -> 0.941 -> 0.946 -> 0.947 on the public leaderboard, built on the same pilkwang
support pack as our port. The question was whether any of it transfers.

## Code: nothing new

The notebook applies ten hex-encoded exact-text patches to the pack script. Decoded and compared with our
`pack_predict.py` (which is bit-exact against the 0.942 notebook we ported), every mechanism is already
present: eight-view D4 detection TTA, primary edge-feature TTA, secondary edge-feature TTA with the 0.75
blend, moment-matched dual-seed detection blend with the 0.90 retention guard, low-margin consensus for the
secondary edge logits, and harmonic bidirectional fusion at weight 0.15. The author's write-up presents the
edge-feature TTA steps as the 0.941 -> 0.947 gains; the 0.942 notebook (a different author, preset
`v29_edge_tta_tight55`) already had them. Our `PredictConfig` defaults equal the 0.947 values except for
one weight (below).

## Configuration: six differences

| parameter | ours (0.942) | 0.947 |
|---|---|---|
| `motion_relink_tight_um` | 5.5 | 6.0 |
| `gap_close_um` | 5.8 | 5.0 |
| `deepcenter_safe_div_veto` / threshold | off / 0.26 | on / 0.20 |
| `secondary_edge_weight` | 0.20 | 0.15 |
| DeepCenter D4 TTA on the veto heatmap | not present | on |
| post-process selection on 4 train volumes per embryo at inference time | not present | on (7 candidates, +0.001 margin) |

The last two are new code. DeepCenter TTA is now `PostprocessConfig.deepcenter_tta` (default False, i.e. the
0.942 behaviour). The inference-time sweep scores the pipeline on training volumes the weights trained on and
picks among `gap_close_um=4.5`, `motion_relink_tight_um=5.5`, `motion_relink_relaxed_um=9.0`,
`motion_relink_learned_bonus=1.25`, `gap2_max_step_um=4.0`, `gap_close_reuse_um=2.8`,
`deepcenter_gap_threshold=0.35`; it was not ported (leaky by construction, and see the numbers below), but
its seven candidates were run as arms.

## Measured on our clean 20-volume split (our blend weights, one change each)

| arm | score | 44b6 | 6bba | div (tp/fp/fn) | FP | FN | delta |
|---|---|---|---|---|---|---|---|
| baseline (0.942 config) | 0.8952 | 0.7932 | 0.9085 | 2/15/17 | 705 | 884 | |
| relink tight 6.0 | 0.8952 | 0.7966 | 0.9080 | 2/15/17 | 707 | 882 | 0.0000 |
| gap close 5.0 | 0.8955 | 0.7937 | 0.9087 | 2/15/17 | 701 | 884 | +0.0003 |
| DeepCenter safe-div veto @0.20 | 0.8965 | 0.7935 | 0.9097 | 2/10/17 | 698 | 886 | +0.0013 |
| secondary edge weight 0.15 | 0.8952 | 0.7932 | 0.9085 | 2/15/17 | 705 | 884 | 0.0000 |
| DeepCenter D4 TTA | 0.8953 | 0.7932 | 0.9086 | 2/15/17 | 704 | 884 | +0.0001 |
| **all five together (= 0.947 config)** | 0.8938 | 0.7969 | 0.9059 | 1/6/18 | 695 | 888 | **-0.0014** |
| sweep candidate gap 4.5 | 0.8956 | 0.7942 | 0.9088 | 2/15/17 | 699 | 884 | +0.0004 |
| sweep candidate relaxed 9.0 | 0.8935 | 0.7947 | 0.9064 | 2/18/17 | 696 | 912 | -0.0017 |
| sweep candidate bonus 1.25 | 0.8954 | 0.7950 | 0.9084 | 2/15/17 | 704 | 882 | +0.0002 |
| sweep candidates gap2 step 4.0 / reuse 2.8 / dc gap 0.35 | 0.8952-0.8953 | | | | | | 0 |
| **ours: motion relink off** | **0.9066** | 0.8166 | 0.9178 | 5/25/14 | 638 | 866 | **+0.0114** |
| 0.947 config + motion relink off | 0.9059 | 0.8301 | 0.9142 | 3/11/16 | 619 | 877 | +0.0107 |

Every 0.947 change is within +-0.0015 here, and combined they are slightly negative (the DeepCenter veto on
divisions cuts the division term from 0.059 to 0.040). Their own sweep candidates are flat too. Added on top
of our relink-off configuration they give 0.9059 vs 0.9066: nothing. The one change that matters is ours,
motion relink off, and it is cross-validated on three weight sets in
`reports/2026-09-17-error-analysis-0942.md` (+0.011 blend, +0.009 single seed, +0.018 shipped weights).

Why the public chain saw gains we do not: each step was accepted on the public leaderboard (+0.001 to
+0.005 per step, within the local noise we measure, 0.002-0.005) after selection on training volumes. The
one substantive idea in the write-up, spatial ensembling of the association features, we already had.

## Decision

- Disregard the 0.947 parameter set. `PostprocessConfig` and `PredictConfig` stay at the 0.942 values.
- Adopt motion relink off as the shipped configuration: `scripts/predict.py --preset tuned` (now the default;
  `--preset notebook` reproduces the notebook exactly, and the dataclass defaults and the parity check are
  unchanged). This needs a leaderboard confirmation: the same package with `--preset tuned` is the next
  submission.
- Keep `deepcenter_tta` and the `--post-set` / `--predict-set` overrides as test hooks (default off).

Scores: `dist/score_ea_nb947_*.json`, `dist/score_ea_pp_*.json`; predictions `dist/preds_val_ea_*`.
