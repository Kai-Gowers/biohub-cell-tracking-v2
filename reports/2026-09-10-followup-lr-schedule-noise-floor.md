# Follow-up: peak LR vs. schedule length, full-schedule runs, and the run-to-run noise floor

## Setup

Six single-variable runs submitted the afternoon of 2026-09-10 to resolve
the open questions in `reports/2026-09-10-overnight-batch-11-experiments.md`,
same `scripts/slurm_train.sbatch` / L40S / bf16 / `--val-frames-per-volume 16
--no-resume` as that batch, each scored on the 20-volume held-out split
(both last-epoch and `val_loss`-selected `_best`) by a chained
`scripts/slurm_score.sbatch` job. Reference config ("historical") is
`--epochs 30 --lr 5e-4 --patience 15`, batch 4 x grad_accum 2.

| name | job | differs from historical config by | tests |
|---|---|---|---|
| `lr5e4_ep100` | 2992650 | epochs 100, patience 20 | peak LR alone, against `ctrl_bs4` (lr 1e-3, epochs 100) |
| `lr1e3_ep30` | 2992653 | lr 1e-3 | peak LR alone, the other direction |
| `tmax30_nopatience` | 2992656 | no patience | the 30-epoch schedule run to completion |
| `bs16_tmax30` | 2992659 | batch 16, grad_accum 1 | batch alone under the good schedule |
| `bs16_tmax30_nopat` | 2992662 | batch 16, no patience | batch + full schedule |
| `replicate_tmax30_r2` | 2992665 | nothing | run-to-run noise (torch init is unseeded) |

Four last-epoch checkpoints were additionally re-linked with `--no-edge-model`
(distance-only linking, same detections) to measure the edge scorer's
contribution on a fixed detector.

## Results

| run | stopped | best ep | final lr | **SCORE last** | **SCORE best** | rec last | ratio last | FP / FN last |
|---|---|---|---|---|---|---|---|---|
| `replicate_tmax30` (this morning) | 22/30 | 7 | 8.3e-5 | **0.8056** | 0.7730 | 0.953 | 1.043 | 1104 / 1697 |
| `tmax30_nopatience` | 30/30 | 24 | 0 | **0.8007** | 0.7985 | 0.948 | 1.063 | 1042 / 1778 |
| `lr5e4_ep100` | 31/100 | 11 | 3.9e-4 | **0.7996** | 0.7634 | 0.949 | 1.023 | 1148 / 1758 |
| `bs16_tmax30_nopat` | 30/30 | 17 | 0 | 0.7976 | 0.7730 | 0.923 | 0.961 | 1035 / 2027 |
| `bs16_tmax30` | 25/30 | 10 | 3.3e-5 | 0.7741 | 0.7615 | 0.938 | 1.042 | 1315 / 2013 |
| `replicate_tmax30_r2` | 30/30 | 25 | 0 | **0.7513** | 0.7471 | 0.951 | 1.053 | 1535 / 2136 |
| `lr1e3_ep30` | 30/30 | 19 | 0 | 0.7469 | 0.7614 | 0.949 | 1.048 | 1646 / 2129 |
| ref: `ctrl_bs4` (lr 1e-3, ep 100) | 33/100 | 13 | 7.6e-4 | 0.7696 | 0.7622 | 0.938 | 0.990 | 1440 / 2035 |

Distance-only relink of the same last-epoch detections:

| checkpoint | with edge scorer | distance only | edge scorer worth |
|---|---|---|---|
| `replicate_tmax30` | 0.8056 | 0.7940 | +0.012 |
| `tmax30_nopatience` | 0.8007 | 0.7887 | +0.012 |
| `bs16_tmax30_nopat` | 0.7976 | 0.7825 | +0.015 |
| `replicate_tmax30_r2` | 0.7513 | 0.7333 | +0.018 |

## What this says

1. **The run-to-run noise floor is about 0.05, not 0.01.** Three runs of the
   identical config scored 0.8056, 0.8007 and 0.7513. The 0.7513 run is not
   a training failure: it has the *lowest* `val_loss` of the three (0.0155),
   the highest validation edge accuracy (0.84), the same detection recall
   (95.1%) and node ratio. Its whole deficit is ~500 extra FP edges and
   ~400 extra FN edges -- linking, not detection recall. Model init is
   unseeded (`train.py` seeds only the sampler), so this is genuine
   variance between otherwise identical runs. **Every single-run
   comparison in this repo's history, including this morning's table and
   the original 0.7574-vs-0.8074 gap, has to be read against this floor.**
   Differences under ~0.03 between two single runs mean nothing.

2. **Peak LR still separates, even against that floor.** Pooling both
   batches: the six lr 5e-4 runs scored 0.8056, 0.8007, 0.7996, 0.7976,
   0.7741, 0.7513; the twelve lr 1e-3 runs scored 0.7237-0.7702 with a
   maximum of 0.7702. Four of six lr 5e-4 runs beat every lr 1e-3 run.
   Within the follow-up pair: `lr5e4_ep100` (0.7996) vs `ctrl_bs4`
   (0.7696) differ only in LR and favour 5e-4; `lr1e3_ep30` (0.7469) vs the
   three historical-config runs (median 0.8007) differ only in LR and
   favour 5e-4 again. Schedule length (T_max 30 vs 100) did not matter on
   its own: `lr5e4_ep100` stopped at 31/100 with the LR still at 3.9e-4,
   barely annealed, and scored 0.7996.

3. **Running the 30-epoch schedule to completion is fine but not a boost.**
   `tmax30_nopatience` (epoch 30, lr annealed to 0) scored 0.8007, in the
   same band as the patience-stopped 0.8056. Val_loss kept improving to
   epoch 24-25 in both no-patience runs, so `--patience 15` was not
   cutting anything important; either way is acceptable.

4. **Batch 16 under the good schedule: inconclusive, probably a small cost.**
   0.7741 (patience-stopped at 25) and 0.7976 (full 30). Both under the
   lr 5e-4 batch 4x2 median but inside the noise floor. `bs16_tmax30_nopat`
   is notable for reaching 0.7976 with the *lowest* detection recall of
   any run (92.3%, node ratio 0.96): it under-detects and links what it
   finds precisely (FP 1035, the lowest of any cluster run). Batch 16 is
   not a free lunch and not clearly a loss.

5. **The edge scorer is worth a consistent +0.012 to +0.018 over distance
   linking**, on all four detectors tested, including the bad `_r2` run.
   So the `_r2` deficit is *not* the edge MLP: with distance-only linking
   the `_r2` detector still scores 0.7333 vs 0.7940 for its twin. Something
   about that run's *detections* -- positions, or spurious near-duplicate
   detections that steal links under both linkers -- is worse, even though
   recall-within-7um and node count are unchanged. Not yet diagnosed.

6. **Last-epoch vs `_best`:** last-epoch won 4 of 6 here (+0.036, +0.025,
   +0.013, +0.004, +0.002; lost -0.015 on `lr1e3_ep30`). Over all 17
   cluster runs: last-epoch >= `_best` in 14. The `_best` file remains the
   wrong default for inference on this pipeline.

## Decisions

- Keep `lr=5e-4` as the canonical peak LR. That is the one finding that
  survives the noise floor.
- Keep `--epochs 30`; patience on or off does not matter measurably.
- Stop treating any single-run score difference under ~0.03 as a result.
  Any future config comparison needs >=3 seeds per arm, or a seeded init
  (`torch.manual_seed`) so that a *paired* comparison is possible -- the
  latter is the cheaper fix and is not implemented yet.
- Best checkpoints available for a submission, in order:
  `detector_replicate_tmax30.pt` (0.8056), `detector_tmax30_nopatience.pt`
  (0.8007), `detector_lr5e4_ep100.pt` (0.7996). Given the noise floor these
  three are indistinguishable; all three beat everything trained at lr 1e-3.

## Open questions

- Seed `torch` init and re-run the historical config 3x to see whether the
  0.75 outlier recurs at a ~1-in-3 rate or was rarer.
- Diagnose the `_r2` detector: compare per-frame detection positions
  against its twin on the volumes where its FP count jumped most.
- Whether averaging weights or ensembling TTA across the three 0.80
  checkpoints buys anything -- cheap to test with `predict.py` and no
  training.
- bf16 vs fp16 and the clip cap (1.0 vs 5) remain untested single
  variables; both now need a multi-seed design to be readable.
