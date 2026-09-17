# Error categories of the 0.842 pipeline, ranked by frequency, with typical examples

*Kai Gowers, 2026-09-16. Follows up `2026-09-14-error-analysis-0842.md` (whose fig09/fig10 showed one
extremal example per hand-named kind). Same inputs: the `main` checkpoint `detector_tmax30_epoch23.pt`,
20 held-out videos, prediction dump `dist/preds_val_0842_dump`. Figures and per-error tables in
`reports/figures/2026-09-16-error-examples/`.*

## What changed since the 09-14 report

The 09-14 figures picked one example per kind by an extreme criterion (dimmest cell, largest margin), the kinds
were named by eye, and the "merged pair" panel (fig10 row 3) did not actually show which cell had claimed the
detection. This iteration:

- assigns **every** counted error to one category, ranks categories by count, and picks the example closest
  to each category's median (a typical case, not the worst one), two per category;
- draws an **xz side view** next to the xy projections, marks the detection nearest to the missed cell and
  the GT cell that claimed it, and shows the cell's own track at t-1 and t+1;
- re-runs the detector on CPU at every missed cell and reads its probability field there, so the detector
  categories are mechanisms, not appearance guesses;
- along the way found two things that change the reading of the 09-14 report: detector **z jitter** is behind
  most linker errors, and the local metric **truncates coordinates** before matching.

## Detector: 685 missed annotated cells (4.8% of 14,177)

Peaks are local maxima of the sigmoid field above τ = 0.985 within a 5x5x5 window (±3.25 µm) on the 1.625 µm grid.
Probing the field at each missed cell (max within ±2 grid cells) separates two very different failures:

| rank | category | n | % | 44b6 / 6bba | brightness (x frame median, median) | field max near cell |
|---|---|---|---|---|---|---|
| 1 | **D3 field above threshold at the cell, but the nearest local maximum is a neighbour's, 7-10 µm away** | 448 | 65 | 74 / 374 | 3.9 | 1.00 (at the cell itself ≥ τ in 332 of 448) |
| 2 | **D5 local-metric truncation artefact**: a detection 5.6-7 µm away in float coordinates, unclaimed, that the metric rejects because it truncates coordinates to integers before measuring distance | 85 | 12 | 21 / 64 | 2.6 | 1.00 |
| 3 | **D1 field below threshold for ≥ 3 consecutive frames** (dim cell the detector never sees) | 134 | 20 | 0 / 134 | 1.7 | 0.96 |
| 4 | D2 field below threshold for 1-2 frames | 12 | 2 | 0 / 12 | 1.9 | 0.97 |
| 5 | D4 nearest detection within 7 µm but claimed by a neighbouring *annotated* cell (true merged pair) | 6 | 1 | 6 / 0 | 1.2 | 1.00 |

Rank 1 is the surprise. Two thirds of the misses are cells where the detector's field is above threshold at the
cell (median 1.00), but the field keeps rising toward a neighbour 7-10 µm away, so the only peak within the NMS
window belongs to that neighbour. The neighbour is almost always unannotated (only 22 of 448 detections are
claimed by another GT cell), which is why the 09-14 count of "merged pairs" (13%) missed it. These cells are of
normal brightness (median 3.9x, vs 6.4x for all cells; 68 of them are brighter than 8x) and happen in both
embryos; 178 are single-frame flickers, 181 last 3+ frames. The offset to the neighbour's peak is 6 µm in xy and
6 µm in z (medians), so in the xy projection the cell often looks clearly separate from the detection that
absorbed it (fig13 rows 5-6).

Re-extracting peaks from the same field with other rules (appendix table) shows what these cells are: with a 3x3x3
window instead of 5x5x5, 331 of the 448 get a peak within 7 µm, so the field does have a bump at the cell, but a
higher value 2 cells away on the neighbour's flank suppresses it. Swapping the sigmoid for the logits (no
saturation ties) changes nothing (32 recovered), so this is not the tie-breaking. The 3x3x3 window is not a fix on
its own: it raises detections per frame by 63% (171 -> 320), i.e. it splits ordinary nuclei into several peaks. A
rescue rule (keep 5x5x5 peaks, add 3x3x3 peaks that are far from every kept peak) is measured below.

The dim-cell story survives but is smaller than stated on 09-14: 146 misses (21%) are cells where the field never
reaches threshold, all in 6bba, median brightness 1.7x the frame median, 116 of them in three videos
(`6bba_268e1230`, `6bba_3a1849c2`, `6bba_3abfe10a`). They are lost for 11 frames at a time (median).

The "large bright pre-mitotic nucleus" kind from 09-14 does not hold up: the 82 misses brighter than 8x are not
within 3 frames of any annotated division (0 of 82), not at track ends (2 of 82), and 68 of them fall in D3. They
are bright cells whose peak was absorbed by a neighbour, not a distinct appearance class.

**Examples** (`fig13_detector_examples.png`, one row per example, columns = xy at t-1, t, t+1 and xz at t;
white ring = the missed cell, red ring = nearest detection, yellow ring = GT cell that claimed it):

| row | category | case | what to look at |
|---|---|---|---|
| 1-2 | D1 sustained, below threshold | `6bba_268e1230` t=87 (12 frames), `6bba_3a1849c2` t=37 (32 frames) | faint blob at the ring, field 0.87 / 0.75 at the cell, nothing detected within 10 µm in any frame |
| 3-4 | D2 transient, below threshold | `6bba_5c824876` t=16, `6bba_3abfe10a` t=68 | same cell detected at t-1 and t+1 (red ring on the white ring), field dips to 0.72 / 0.01 for one or two frames |
| 5-6 | D3 absorbed by a neighbour's peak | `6bba_7b5d3b2c` t=1, `44b6_d5e7d891` t=16 | field 0.92 / 0.999 at the cell, yet the only detection is on the neighbour 12 / 8 µm away; the cell is visibly separate |
| 7 | D4 true merged pair | `44b6_706092f0` t=51 | two GT cells 2.4 µm apart, one detection between them, yellow line to the cell that got it |
| 8-9 | D5 truncation artefact | `44b6_e57ff5c6` t=41, `6bba_283bf9f1` t=48 | detection 7.0 / 6.4 µm away, mostly in z (side view), field 1.00; counted as a miss only because of integer truncation |
| 10 | bright cell (10x), absorbed | `6bba_c27cba08` t=96 | bright nucleus, detection 7.1 µm away, 6.8 µm of it in z |

## Linker: 708 counted FN edges whose two endpoints were both detected

| rank | category | n | % | 44b6 / 6bba | typical scores | detections of the true pair differ in z by > 3 µm |
|---|---|---|---|---|---|---|
| 1 | **L1 the linker chose a different partner** | 386 | 55 | 113 / 273 | | 72% |
| | L1a near-tie: chosen beats true by < 0.1 | 150 | 21 | 51 / 99 | 0.99 vs 0.98 | 69% |
| | L1b confident: margin > 0.5 | 130 | 18 | 34 / 96 | 0.99 vs 0.11 | 79% |
| | L1b intermediate margin | 79 | 11 | 22 / 57 | 0.99 vs 0.74 | 65% |
| | L1c true partner ranked first, but the greedy pass had given it away | 23 | 3 | 5 / 18 | 0.93 vs 0.99 | 65% |
| | L1d true partner beyond the 15 µm gate | 4 | 1 | 1 / 3 | | |
| 2 | **L2 an unannotated competitor took the true target; the cell was left with no link** | 207 | 29 | 55 / 152 | competitor 0.99, this cell 0.86 | 69% |
| 3 | **L3 the true pair scored below the 0.5 threshold; no competitor; cell left unlinked** | 97 | 14 | 5 / 92 | 0.19 | 36% |
| 4 | L4 division, second child never linked | 18 | 3 | 3 / 15 | | |

L2 was invisible in the 09-14 breakdown (it was folded into "unlinked"). It is the greedy-competition mechanism at
scale: a source 4-10 µm away, usually above or below in z (77% differ by > 3 µm), scores 0.99 for our cell's true
target, the greedy pass gives it away, and our cell's next candidate is below 0.5 in 83% of cases so it ends up with
no edge. In 69% our cell had scored its true target ≥ 0.5 and would have linked correctly had the target been free.
Adding L1c, competition for one target accounts for 230 errors (32%).

L3 is almost entirely a 6bba phenomenon (92 of 97): the pair is 1.7 µm apart, both detections are fine, and the
scorer still gives 0.19. These pairs have no z jitter to speak of; the detections are simply off-centre (median
localisation error 3.6 µm vs 1.7 µm for correct links).

**Examples** (`fig14_linker_examples.png`; columns = xy at t, xy at t+1, xz at t, xz at t+1; white ring = GT
source (t) / true target (t+1), red ring = their detections, yellow ring = the detection the linker chose, magenta =
the competing source and its link, red x = where the source detection was):

| row | category | case | what to look at |
|---|---|---|---|
| 1-2 | L1a near-tie | `6bba_5c824876` t=1, `44b6_d5e7d891` t=18 | GT moved 2 µm, but the two detections of the true pair are 6 µm apart in z (side views); the chosen detection sits at the source's z |
| 3-4 | L1b confident | `6bba_3abfe10a` t=89, `6bba_2312ac41` t=23 | true pair's detections 10 µm apart in z; true partner scored 0.11 / 0.17, rank 5 / 3 |
| 5-6 | L1b intermediate | `6bba_d1acb6ff` t=24, `44b6_e57ff5c6` t=40 | fast cells (GT moved 7.4 / 5.7 µm), true partner rank 3 / 2 |
| 7 | L1c greedy | `44b6_1574802b` t=21 | the 09-14 fig09 row 1 case; competitor 9.5 µm away and 6 µm higher in z took the target with 1.00 |
| 8 | L1d beyond gate | `44b6_d754aa59` t=55 | GT jumped 15.4 µm (the jump-frame video from `2026-09-14-jump-frames-postprocessing.md`) |
| 9-10 | L2 stolen target | `6bba_c328f2fd` t=82, `6bba_cf35214c` t=69 | our cell scored its target 0.99 / 1.00 (rank 1); a competitor 7.3 µm away (row 10: 6 µm higher in z) also scored 0.99 / 1.00 and won |
| 11-12 | L3 under threshold | `6bba_3a1849c2` t=9, `6bba_afb141ff` t=45 | pair 1 µm apart, no competitor, score 0.06 / 0.09 |
| 13 | L4 division | `6bba_7af54fde` t=55 | second child 10.8 µm away, scored 0.02 |

## Finding 1: detector z jitter, not crowding, is the main driver of linker errors

For every GT edge with both endpoints detected, compare the z offset between the two *detections* with the z offset
between the two *GT positions*:

| | n | GT pair |dz| (median) | detection pair |dz| (median) | detection |dz| > 3 µm |
|---|---|---|---|---|
| correct links | 12,036 | 1.6 µm | 0.6 µm | 15% |
| L1 wrong partner | 386 | 1.6 µm | 5.3 µm | 72% |
| L2 stolen target | 207 | 1.6 µm | 4.7 µm | 69% |
| L3 under threshold | 97 | 0.0 µm | 1.6 µm | 36% |

The cells did not move in z; the detector placed them 2-4 slices apart in consecutive frames. Association recall
against the jitter the detector introduced, |dz_detections - dz_GT|:

| jitter | < 1.7 µm | 1.7-3.3 | 3.3-5 | 5-6.6 | > 6.6 |
|---|---|---|---|---|---|
| link correct | 97.9% | 93.8% | 74.7% | 42.0% | 12.6% |
| n | 9,895 | 2,012 | 470 | 174 | 159 |

Controlling for jitter < 1.7 µm, recall at < 7 µm crowding is 88% (vs 72% uncontrolled), 96% at 7-10 µm, 99% beyond.
The 09-14 conclusion "crowding decides" stands only in part: crowding is where the detector's z estimate becomes
unreliable, and the pair scorer, fed a 5 µm z displacement, prefers the neighbour that stayed at the source's z
(chosen detection: 1.7 µm xy, 2.2 µm z from the source; true detection: 5.1 µm xy, 5.3 µm z).

Z localisation error of detections against GT: median 1.4 µm, 90th percentile 3.3 µm, 95th 4.3 µm (GT is on integer
slices, so ±0.8 µm of that is quantisation). One in ten detections is off by two or more slices.

## Finding 2: the local metric truncates coordinates before matching

`GeffGraph.nodes_by_t` casts z, y, x to `int` (floor), and `metric.match_nodes_per_frame` matches on those. A
detection at z = 61.96 becomes 61, a 1.56 µm error in one axis, always downward. The submission writer rounds
(`int(round())`), so Kaggle scores different coordinates than the local metric does. Effect on the 0.842 held-out
prediction, same graphs:

| | truncated (current) | float coordinates | Δ |
|---|---|---|---|
| total score | 0.8073 | 0.8108 | +0.0035 |
| 44b6 | 0.6153 | 0.6330 | +0.018 |
| 6bba | 0.8327 | 0.8341 | +0.001 |
| edge TP / FP / FN | 12,036 / 1,053 / 1,656 | 12,063 / 1,022 / 1,629 | |

85 of the 685 "missed" cells (12%) are this artefact. Not fixed here (it shifts every number in earlier reports);
the fix is to round in `nodes_by_t` as `write_submission` does, or to match on floats and round only when writing.

## What this means for the current work

- **Detector**: the largest category (65%) is not "cells too dim" but cells whose peak is absorbed by a neighbour's
  on the 1.625 µm grid, and the second linker category (29%) plus 72% of the first are downstream of z
  localisation errors of 2-4 slices. Both point at the detector's output resolution and peak extraction, not at its
  recall on faint objects. The 0.942 pipeline uses the same 4x-decimated grid and a (3,3,3) max-pool, so the same
  probe should be run on it once our checkpoints finish.
- **Linker**: any pair scorer that sees only (source, target, offset) will be fooled by a 5 µm z jitter it cannot
  distinguish from real motion. The 0.942 node transformer sees all detections of both frames, which is the right
  shape for the L1/L2 competition errors, but it cannot undo a wrong z. A z-consistent detector (sub-slice
  refinement or temporal smoothing of z) would help both.
- **Metric**: fix the truncation before the next round of held-out comparisons; the 44b6 numbers move by 0.02.

## Appendix A: peak-extraction variants on the same detector field

The detector was re-run (CPU, same TTA) on the 515 frames that contain a missed cell; the pipeline rule reproduces
the dumped detections exactly (nearest-detection distance identical for all 685 misses). "Recovered" = a peak
within 7 µm of the missed cell.

| rule | D1 (134) | D2 (12) | D3 (448) | D4 (6) | D5 (85) | all (685) | detections per frame (median) |
|---|---|---|---|---|---|---|---|
| pipeline: 5x5x5 NMS on sigmoid, τ = 0.985 | 0 | 0 | 0 | 6 | 85 | 91 | 171 |
| 5x5x5 NMS on logits | 0 | 0 | 32 | 6 | 67 | 105 | 171 |
| 3x3x3 NMS on sigmoid | 5 | 2 | 331 | 6 | 76 | 420 | 320 |
| 3x3x3 NMS on logits | 5 | 2 | 330 | 6 | 76 | 419 | 319 |

(D4/D5 counts are 7 µm-in-float-coordinates hits; they are already "detected" by that criterion.)

**Rescue rule** (keep the 5x5x5 peaks, add every 3x3x3 peak that is at least R µm from all of them):

| R | D3 recovered (of 448) | all misses recovered (of 685, 91 already) | detections per frame (median, pipeline 171) | detections vs pipeline |
|---|---|---|---|---|
| 4.0 µm | 300 | 396 | 254 | +34% |
| 5.0 µm | 294 | 389 | 245 | +29% |
| 5.5 µm | 293 | 387 | 240 | +27% |
| 6.5 µm | 269 | 362 | 224 | +19% |

Two thirds of the D3 cells come back at any R, but at the price of 19-34% more detections per frame, i.e. the
detector's field has many secondary bumps 5-7 µm from a kept peak that are *not* annotated cells (or are
unannotated ones; the sparse GT cannot tell). Whether the trade pays on the metric (the node-count adjustment
penalises the extra nodes, and the linker gets more candidates) is an experiment for a full held-out run, not
decided here. The honest reading: the 5x5x5 window is a compromise between splitting nuclei and swallowing
neighbours, and the detector's field is not sharp enough to make the choice for it.

## Appendix C: why the detector's z estimate jitters

Measured on 558 detections in three frames (`44b6_d5e7d891` t=18, `6bba_5c824876` t=1, `6bba_2312ac41` t=23):

| through the peak | along z | along x |
|---|---|---|
| raw image, half-maximum width | 15 grid cells (24 µm) | 6 grid cells (10 µm) |
| detector probability ≥ 0.985, contiguous extent | 9 cells (15 µm) | 3 cells (5 µm) |
| logit curvature at the peak (sharper = more negative) | -1.1 | -4.1 |

A nucleus is 2.5x longer along z than along x in the image (light-sheet axial blur plus stacked neighbours), the
detector's response is correspondingly flat along z (its curvature is a quarter of the x curvature; 65% of peaks
are "flat in z" by the 1/3 criterion), so the argmax along z is decided by small intensity differences between
slices and moves by 1-2 slices from frame to frame. The consequence on correct detections: z error median 1.4 µm
vs xy 0.8 µm; 41% are off by at least one slice in z, 11% by two or more, vs 17% off by 1.6 µm in xy. The
saturated-plateau tie-break was checked and ruled out: 83% of peaks have no plateau (p < 0.9999) and the kept
voxel is the logit maximum along z in 100% of cases. The training target is a single voxel (`detection_target`),
which asks the network for a precision in z that the image does not carry.

## Appendix B: reproduction

All on the `main` worktree (`/projects/twist2d/gowers/biohub-cell-tracking-v2-main`, uncommitted):

```
PYTHONPATH=src python scripts/error_examples_build.py    # -> dist/error_examples/tables.pkl  (~2 min)
PYTHONPATH=src python scripts/error_examples_probe.py    # -> dist/error_examples/probe.pkl   (CPU, ~9 min)
PYTHONPATH=src python scripts/error_examples_probe2.py   # -> dist/error_examples/probe2.pkl  (peak-rule variants)
PYTHONPATH=src python scripts/error_examples.py --out-dir reports/figures/2026-09-16-error-examples
```

`misses.csv` and `linker_fn.csv` in the figure folder carry every counted error with its category and all the
fields used above; `examples.json` lists the cases shown. Every case can be opened in the 3D browser
(`scripts/error_browser.py`, see the 09-14 report, section 9).
