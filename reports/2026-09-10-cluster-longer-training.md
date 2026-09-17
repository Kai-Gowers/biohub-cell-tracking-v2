# First full training run on the BC HPC cluster: batch_size=16, no divergence, held-out score regressed

## Setup

First real training run off Kaggle, on the BC HPC cluster (SLURM, L40S GPU,
`torch 2.6.0+cu124`, `bf16` autocast confirmed genuinely fast on this GPU via
`scripts/bench_amp_dtype.py` -- not the T4 false-positive). Motivation: escape
Kaggle's 9.5h-session/T4 constraint and try a longer schedule + the batch size
a bigger GPU affords.

`scripts/train.py` gained three CLI flags this session
(`--val-frames-per-volume`, `--batch-size`, `--max-hours`) exposing existing
`train()` kwargs that were previously only reachable from Python directly.
Command:

```
python scripts/train.py --epochs 100 --batch-size 16 --grad-accum 1 \
  --val-frames-per-volume 16 --patience 20 --no-resume \
  --out dist/models/detector.pt --history-json dist/models/history_100ep.json
```

- `epochs=100`: a deliberate stretch past the only previously-confirmed-stable
  schedule (`T_max=30`, hard-tied 1:1 to `--epochs`), without committing to
  400 epochs on an architecture that has never run that long.
- `batch_size=16` (vs. the historical effective batch of 4x2=8): the new
  lever a bigger GPU unlocks, aimed at the recurring instability where
  `conv_grad_norm_max` sits pinned at the clip ceiling from epoch 2 onward in
  every prior run (`reports/2026-08-29-nan-divergence-bf16.md`) -- larger
  batches should reduce per-step gradient variance.
- `val_frames_per_volume=16` (vs. default 4): the historical val split (4
  frames x 20 volumes) made `val_loss` swing ~2x epoch-to-epoch and risked
  triggering early stopping on noise rather than a real signal.
- `lr`, weight decay, and gradient clipping were deliberately left unchanged
  -- already tested and ruled out as fixes for the divergence (weight decay:
  `reports/2026-08-28-weight-decay-1e-2-negative-result.md`; split clipping:
  `reports/2026-08-30-conv-grad-clip.md` /
  `reports/2026-08-30-nan-recurrence-after-conv-clip.md`).

Also fixed in `src/cell_tracking/train.py` this session: `conv_grad_norm_max`
/ `attn_grad_norm_max` / `edgescorer_grad_norm_max` are now measured
**before** `clip_grad_norm_` runs, not after. Previously they were computed
post-clip, which can only show a group's *share* of the already-shrunk joint
vector -- indistinguishable whether the true pre-clip value was 1.01 or 100.
This is a pure diagnostic change with no effect on training.

## Result: 50/100 epochs, early-stopped on patience, no divergence

```
Early stop at epoch 50/100: no val_loss improvement in 20 epochs (best 0.01923 at epoch 30).
```

- **No hard divergence and zero `skipped_steps` for the entire run** -- the
  run went past every historical divergence point (the split-clip experiment
  died at epoch 15; joint-clip baselines died at 22-34) without incident.
  This is the furthest any run in this project's history has gotten.
- `--patience 20` fired at epoch 50 (20 epochs with no improvement past the
  epoch-30 best), so the 100-epoch schedule was never actually tested to
  completion -- `T_max=100`'s cosine LR never finished annealing either.
- The new pre-clip diagnostic revealed real information the old post-clip one
  couldn't: per-step joint `grad_norm` ranged roughly 5-80 across the run
  (nowhere near "barely over the 1.0 cap"), and **`edge_grad_norm_max` turned
  out to be a comparably-sized contributor to the shared clip budget as
  conv**, not negligible as a tiny 3-volume smoke test had suggested (e.g.
  epoch 16: `conv=25.31, attn=1.13, edge=24.44`, consistent with the logged
  joint `grad_norm=35.199` via `sqrt(25.31^2+1.13^2+24.44^2)`). Both
  conv and edge norms trended down in magnitude and frequency of large spikes
  over the back half of the run (epochs 41-47 mostly single/low-double-digit,
  vs. 25-80 in epochs 4-16) rather than escalating -- the opposite of the
  historical "cumulative growth, no sign of leveling off" pattern, though
  `|W|_conv` was still growing slowly (79.95 by epoch 50) when the run ended.

## Held-out score: 0.7574 -- at the low end of this project's history, not an improvement

```
edge_jaccard (un-adjusted)      = 0.7570   (TP=11370 FP=1328 FN=2322)
adjusted_edge_jaccard           = 0.7574
division_jaccard (micro)        = 0.0000   (TP=0 FP=0 FN=19)
SCORE = adj + 0.1*div           = 0.7574
detection recall (within 7um)   = 92.4% (13104/14177)
```

Scored via `scripts/predict.py` + `scripts/score_local.py --held-out` against
the same 20-volume held-out split every checkpoint in this project's history
has used (`VAL_SEED=0`, deterministic -- confirmed by extracting `val_names`
from the checkpoint itself, not re-derived).

Comparing against every other `dist/score_*.json` in this repo, recomputed
the same way (weighted `adjusted_edge_jaccard` + `0.1 * division_jaccard`,
same 20 volumes in every file):

| checkpoint | SCORE |
|---|---|
| tmax30_epoch23 | 0.8074 |
| 0841 baseline | 0.8031 |
| repair_off | 0.8031 |
| repair_on_v2 | 0.7904 |
| repair_on | 0.7831 |
| edgeneck | 0.7801 |
| tmax30_epoch8 | 0.7972 |
| **this run (epoch 30 best)** | **0.7574** |
| parentsoftmax | 0.7486 |

This run's held-out score is **at the low end of this project's history**,
better than only the parent-softmax experiment and below every other variant
-- including the ones that themselves ended via a NaN crash rather than a
clean patience stop. Per `metric.py`'s own documented caveat and this
project's history (cross-checkpoint comparisons were anti-correlated with the
Kaggle leaderboard, r=-0.799, in the sibling repo), this table is context,
not a ranking to trust at face value -- but it does not show the improvement
the batch-size change was aimed at producing, and on this one number it looks
like a regression.

## Status / open questions

- **Training stability improved, prediction quality did not (this run).**
  The instability symptom this change targeted (pinned clip-ceiling gradient,
  periodic hard divergence) did not recur, and training ran further than any
  prior run. But the resulting checkpoint scores worse on the held-out set
  than checkpoints that *did* crash early. These are two different claims and
  this run only supports the first one.
- **Too many variables changed at once to attribute the score regression.**
  This run differs from history in batch size, val sampling, hardware
  (L40S/bf16 vs. Kaggle T4/mixed), *and* stopped via patience at a different
  point in training than any historical comparison run. A same-hardware,
  single-variable follow-up (e.g. `batch_size=16` vs. the historical
  effective 8, same GPU, same epoch count) would be needed to actually
  attribute this to the batch size change rather than to some other
  difference or to this particular run's random draw.
- **`patience=20` may have stopped this run before it found a better optimum.**
  Val_loss was still noisy (0.021-0.035) rather than clearly converged when
  patience fired at epoch 50; it is not established whether a longer patience
  budget or a different `--select-by` would have found a better checkpoint
  later in the schedule.
- Not yet checked: whether the edge-scorer's own predictions are contributing
  disproportionately to the FN/FP counts behind the lower `adjusted_edge_jaccard`
  (the per-volume breakdown in `dist/preds_val/score.json` has this, not yet
  compared against a historical per-volume breakdown volume-by-volume).
