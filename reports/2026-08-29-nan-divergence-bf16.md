# NaN divergence: switch AMP to bfloat16, log the triggering sample

## Motivation

Every full training run in this project's history has been cut short by
the recurring NaN divergence, regardless of edge-scorer variant: the 0.841
baseline diverged at epoch 23 (right after its best), the weight_decay=1e-2
test at 22, parent-softmax at 34. None has gotten past ~38% of the intended
90-epoch cosine schedule -- meaning the LR has never actually annealed down
to a meaningfully low value in any run to date, and every "best checkpoint"
so far is an artifact of when the run happened to crash, not of the
schedule completing. Weight decay was already tested and refuted as a fix
(`2026-08-28-weight-decay-1e-2-negative-result.md`). This is a bigger
unrealized lever than either edge-scorer architecture change tried so far
(both of which regressed the held-out score), so it's worth another pass.

## What the diagnostics actually show

Reviewing the per-component gradient diagnostics across every run that has
them:

- `conv_grad_norm_max` sits at ~0.97-1.0 (the clip ceiling) starting at
  epoch 2 and stays there for the entire run, in every run logged --
  **not** an escalating rare event, but the steady-state condition from
  early on. `attn_grad_norm_max` stays comfortably low (0.03-0.27)
  throughout, by contrast.
- `n_skipped_steps` (GradScaler catching an actual inf/nan gradient, not
  just a large one) recurs roughly every 6-8 epochs, in every run,
  regardless of edge-scorer variant.
- Conv weight norm grows 50-90% over a run with no sign of leveling off,
  and reasonable weight decay can't meaningfully slow that growth at this
  learning rate (already measured).

Together this reads as a numerical-precision problem, not a
learning-rate or architecture problem: the conv/decoder stack operates at
the edge of `float16`'s dynamic range for its whole training life (that's
what constant near-ceiling clipping means), and as its weights keep
growing, the safety margin before an actual overflow shrinks until some
batch eventually tips it into a genuine, unrecoverable inf/nan.

## Changes

`train.py`:

1. **AMP dtype**: `torch.autocast("cuda", enabled=amp)` defaulted to
   `float16`. Now prefers `bfloat16` when the GPU supports it
   (`torch.cuda.is_bf16_supported()`) -- `bfloat16` has `float32`'s
   exponent range, so the same weight/activation magnitudes seen in this
   model can't overflow it, at the cost of less mantissa precision. Falls
   back to `float16` (with `GradScaler`'s loss-scaling safety net) on
   pre-Ampere GPUs where bf16 would only run emulated and slow (e.g.
   Kaggle's T4s) -- worth checking which GPU the Kaggle session actually
   gets before expecting a speed-neutral run. `GradScaler` is now only
   enabled when actually autocasting to `float16`; it exists specifically
   to counter `float16` underflow and bf16 doesn't need it.

   **Known tradeoff**: with `GradScaler` disabled under bf16, the
   `n_skipped_steps` leading-indicator (comparing scale before/after) can
   no longer fire, since there's no scale to compare. If bf16 works, this
   diagnostic should simply become moot (the failure mode it detects
   shouldn't occur); if a run still diverges under bf16, `grad_norm_max`
   and the hard non-finite-loss check still catch it, just without that
   specific early-warning signal.

2. **Sample-level divergence logging**: `compute_edge_loss` now also
   returns the `(volume_name, t)` it sampled. `run_epoch` accumulates every
   detection batch's `(name, t)` pairs plus any edge sample into
   `accum_samples`, cleared after each optimizer step (a skip or a hard
   divergence is specific to whatever was accumulated into that step's
   gradient). Both a caught inf/nan skip and `TrainingDivergedError` now
   report exactly which volume(s)/frame(s) were involved, printed
   immediately and stored in `exc.diag["batch_samples"]`. This is the
   diagnostic flagged as a next step in the weight-decay report but never
   implemented -- lets the next divergence (if bf16 doesn't fully fix it)
   show whether it's the same volume/frame recurring (a data problem) or
   different ones every time (a pure numerics-margin problem).

## Verified

- Local smoke train (4 volumes, 8 epochs) runs clean on the `disabled`-AMP
  path (no CUDA locally) -- confirms the 3-tuple `compute_edge_loss`
  return doesn't break either the training or validation branch.
- Deterministically forced a non-finite loss (monkeypatched `compute_loss`
  to return NaN on the 3rd call) to exercise `TrainingDivergedError`'s
  path without waiting on organic hyperparameter-induced blowup. Output
  correctly reported the exact accumulated samples at the point of
  failure: `[('44b6_33b596bf', 44), ('6bba_7d3058ae', 35),
  ('44b6_9be80b04', 91), ('44b6_95029e92', 77), ('6bba_c73a1d11', 1)]` --
  4 detection samples (`batch_size=4`) plus 1 edge sample, matching what
  had accumulated since the last optimizer step.
- Cannot verify the bf16 path itself locally (no CUDA on this machine);
  `torch.cuda.is_bf16_supported()` and the dtype selection logic only run
  on the next real Kaggle GPU session.

## Still open

No held-out or leaderboard result yet -- needs a real training run on
Kaggle to see whether bf16 actually prevents the divergence (or at least
lets a run reach much further into the 90-epoch schedule), and if not,
whether the new sample-level logging reveals a data-specific culprit.
