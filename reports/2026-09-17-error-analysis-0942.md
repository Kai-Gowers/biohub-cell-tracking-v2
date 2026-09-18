# 2026-09-17 -- Error analysis of the 0.942-port pipeline with our own weights

**What was analysed.** The exact configuration that scored 0.927 on the leaderboard (primary
`pack_s314159_ddp_best`, secondary `pack_s0_ddp_best`, our DeepCenter, full post-processing), run on the 20
held-out volumes the weights never trained on (`dist/heldout_split.json`: 5 x `44b6`, 15 x `6bba`). The hidden
test set has no labels, so this is the closest available stand-in for "errors on the test data"; the caveat
is that both held-out embryos are also the two training embryos, while the test embryos are different.
Prediction folders: `dist/preds_val_ddp_best_blend` (full) and `dist/preds_val_ddp_best_blend_ilp` (ILP
stage, before post-processing). Ground truth is sparse: 14,177 annotated cells and 13,692 annotated links
against ~387,000 predicted cells, so only predicted edges that touch an annotated cell are counted.

## 1. The headline numbers

| | value |
|---|---|
| local score (adjusted edge Jaccard + 0.1 x division Jaccard) | **0.8952** = 0.8893 + 0.1 x 0.059 |
| per embryo | 44b6 0.793, 6bba 0.909 |
| counted edges | TP 12,808, FN 884, FP 705 |
| annotated cells detected (7 um) | 96.7 % (97.1 % before post-processing) |
| of detected pairs, correctly linked | 97.6 % |
| divisions | 2 of 19 found, 15 false |

Two thirds of the lost score is missed links (FN 884 vs FP 705); the division term is small in absolute
terms but the 0.059 is where post-processing earns its keep (section 4).

## 2. Which part is accountable

`scripts/analyze_errors.py` assigns every counted error one cause. Taken at face value:

| FN cause (884) | n | % | | FP cause (705) | n | % |
|---|---|---|---|---|---|---|
| an endpoint was not detected | 564 | 64 | | true partner not detected | 184 | 26 |
| both detected, linked elsewhere / unlinked | 309 | 35 | | true partner detected, another chosen | 518 | 73 |
| second child of a division | 11 | 1 | | both matched, wrong pair | 3 | 0 |

That reads as "detector 64 %, linker 35 %" for FN and the reverse for FP. It is not the whole story: the
7 um matching tolerance is large relative to cell spacing in crowded regions (median nearest annotated
neighbour 20 um, but 10 % of cells have a neighbour within 14 um), so when a cell's own detection is missing
or displaced, the matcher hands the annotated cell a *neighbouring* cell's detection, and that neighbour's
perfectly good track then contradicts the annotation. The localisation error of the matched node separates
the two:

| matched endpoint -> its annotated cell | median | >= 4 um |
|---|---|---|
| sources of the 12,808 correct links | 1.7 um | 6 % |
| contradicted endpoints of the 705 false links | 5.0 um | 67 % |
| sources of the 311 "both detected, wrong link" misses | 4.2 um | 52 % |

So roughly two thirds of the "linker" errors sit on a detection that is 4-7 um from the cell it was credited
with, i.e. a substituted or badly placed peak. Folding that in, **about four fifths of all counted errors
originate in detection (missing, displaced or merged peaks); genuine association mistakes on well-localised
detections are about one fifth.** The transformer + ILP linker is not the weak part; the detector in
crowded regions is.

## 3. The main error cases

Figures: `reports/figures/2026-09-17-error-analysis-0942/` (cyan circle = the case, green = annotated
cells, red x = predictions; only nodes in the projected slab are drawn).

**Case A -- a peak 7-10 um from the cell, not a missing peak (71 % of misses; `miss_offset.png`).**
Of the 417 cells missed at the ILP stage, 374 have an *unmatched* prediction within 14 um, and 71 % of
misses have one within 7-10 um (the base rate for a detected cell to have a second prediction that close is
30 %). The offset is z-dominated in 39 % of cases, |dz| median 4.9 um = 3 planes, dxy median 6.8 um. Only 23
cells (5.5 %) have nothing within 14 um. The detector fires, but on the wrong voxel or on the merged blob of
two touching cells. This is the same "peak absorbed by a neighbour" failure the old architecture had
(reports/2026-09-16-error-categories-with-examples.md); the new detector halves the rate but does not change
the mode.

**Case B -- dim cells in 6bba, bright crowded cells in 44b6 (`miss_dim.png`).**

| | 6bba | 44b6 |
|---|---|---|
| intensity of detected cells (median, /frame p99) | 0.59 | 0.62 |
| intensity of missed cells | **0.33** | **0.76** |
| miss rate, intensity < 0.5 x p99 | 10-13 % | (few such cells) |
| miss rate, intensity >= p99 | 1.0 % | 5.9 % |
| miss rate, last quarter of the video | 3.9 % | **10.7 %** |
| miss rate, first quarter | 1.9 % | 1.6 % |

The two embryos fail differently. 6bba loses dim cells (deep or weakly labelled). 44b6 loses bright cells
late in the video, when the field is dense: its miss rate quintuples between the first and last quarter.
Both embryos miss 21 % of cells whose nearest annotated neighbour is under 7 um.

**Case C -- the "stayed put" false link (`fp_stay.png`).** 259 FPs have a detected source and a detected
true child, and in 95 % of them the linker chose a node ~1.7 um from the source over the true child's node
~9 um away. The dump shows why: the source node itself sits 4-6 um from its annotated cell, and its chosen
child continues that same track (95 % have their own child at t+2). It is a neighbouring cell's track,
credited to the annotated cell by the 7 um matcher because the annotated cell's own peak was missing or
displaced at t. A detector error, scored as a linker error.

**Case D -- wrong link in a crowded frame (`fn_wrong_crowded.png`).** 311 misses have both endpoints
detected but linked elsewhere; 77 % of them have another detection within 10 um of the source, 29 % involve
a z move of 2 planes or more (vs 1 % of all annotated links), and 15 % fall in whole-frame-jump pairs
(median frame displacement >= 5 um; 6 % of pairs), a 2.5x enrichment but not the majority. Association
recall by crowding: 99.8 % when the nearest detection is >= 14 um, 95.8 % at 7-10 um, 91.0 % under 7 um.

## 4. What the post-processing stage does

`scripts/error_stage_diff.py` keys every counted edge by its annotated endpoints and compares the ILP stage
with the full output:

| post-processing ... | TP | FP |
|---|---|---|
| kept | 12,597 | 378 |
| **added** | 211 (fixed) | 327 (new errors) |
| **removed** | 203 (damage) | 276 (cleaned) |
| net | +8 | +51 |

On edges it is a wash: 0.8903 -> 0.8893 adjusted Jaccard. It also drops 121 annotated cells that were
detected (short-track filter) and adds 65 (gap nodes). Its whole contribution is the division term (0 ->
0.059, worth +0.006 of score). Per pass, from the ablations below: **motion relink is net harmful**,
line-fit smoothing and safe divisions are net helpful, and gap closing, short-track filter,
enforce-next-frame, prune-isolated and single-parent repair are within noise.

Turning motion relink off: **0.8952 -> 0.9066** (+0.011; 44b6 +0.023, 6bba +0.009), FP 705 -> 638, FN 884 ->
866, divisions 2/15 -> 5/25 (term 0.059 -> 0.114); 16 of 20 volumes improve, the largest by +0.067
(`44b6_e57ff5c6`) and +0.030, the worst regression -0.008. Without it the stage adds 181 TP / 255 FP and
removes 155 TP / 271 FP. This confirms the 2026-09-14 finding (relink radii 5.5/10 um undo the transformer's
links when cells move) with a clean, same-checkpoint measurement.

Cross-checks (same arm, other weights), all on the same 20 volumes:

| weights | relink on | relink off | delta | 44b6 | 6bba | FP | FN |
|---|---|---|---|---|---|---|---|
| our blend (s314159 + s0) | 0.8952 | **0.9066** | +0.011 | 0.793 -> 0.817 | 0.909 -> 0.918 | 705 -> 638 | 884 -> 866 |
| our single seed s314159 | 0.8976 | **0.9068** | +0.009 | 0.784 -> 0.786 | 0.913 -> 0.923 | 700 -> 612 | 908 -> 902 |
| shipped pack weights (leaky, blend) | 0.9175 | **0.9355** | +0.018 | 0.831 -> 0.846 | 0.931 -> 0.948 | 659 -> 570 | 639 -> 607 |
| our blend, relink off + gap close off | | 0.9078 | +0.013 | 0.817 | 0.919 | 605 | 919 |

The gain is not weight-specific: three weight sets, both embryos, fewer FP *and* fewer FN every time, and a
larger division term (relink was also stealing division daughters). The reference notebooks kept motion relink
because they tuned it on volumes their weights had trained on; on clean data it is the single most harmful
pass in the pipeline.

## 5. Component ablations (same weights, one change each)

| arm | score | 44b6 | 6bba | det % | FP | FN | delta |
|---|---|---|---|---|---|---|---|
| **baseline** (blend, full) | 0.8952 | 0.7932 | 0.9085 | 96.7 | 705 | 884 | |
| motion relink off | **0.9066** | 0.8166 | 0.9178 | 96.7 | 638 | 866 | **+0.0114** |
| gap close off | 0.8956 | 0.7955 | 0.9090 | 96.5 | 686 | 908 | +0.0004 |
| det threshold 0.975 (default 0.965) | 0.8958 | 0.8000 | 0.9082 | 96.6 | 694 | 900 | +0.0006 |
| det threshold 0.95 | 0.8946 | 0.7917 | 0.9077 | 96.6 | 702 | 888 | -0.0006 |
| short-track filter off | 0.8950 | 0.7919 | 0.9085 | 97.1 | 709 | 853 | -0.0002 |
| enforce next frame / prune isolated / single-parent repair off | 0.8952 | | | | | | 0 |
| edge threshold 0.40 (default 0.48) | 0.8947 | 0.7948 | 0.9077 | 97.1 | 728 | 846 | -0.0005 |
| edge-feature TTA off | 0.8935 | 0.7872 | 0.9076 | 96.7 | 722 | 898 | -0.0017 |
| edge threshold 0.52 | 0.8899 | 0.7831 | 0.9033 | 96.4 | 702 | 944 | -0.0053 |
| bidirectional fusion off | 0.8899 | 0.7880 | 0.9028 | 96.5 | 710 | 926 | -0.0053 |
| safe divisions off | 0.8898 | 0.7929 | 0.9017 | 96.7 | 691 | 890 | -0.0054 |
| line-fit smoothing off | 0.8880 | 0.7846 | 0.9016 | 96.8 | 780 | 919 | -0.0072 |
| TTA flips only (2 views) | 0.8893 | 0.7761 | 0.9054 | 96.4 | 731 | 955 | -0.0059 |
| no TTA (1 view) | 0.8859 | 0.7814 | 0.9004 | 96.5 | 791 | 959 | -0.0093 |
| post-processing off (`--stage ilp`) | 0.8903 | 0.7953 | 0.9018 | 97.1 | 654 | 892 | -0.0049 |
| **ILP off** (greedy candidates) | 0.8587 | 0.7336 | 0.8749 | 97.4 | 1007 | 970 | **-0.0365** |
| single seed s314159 (no blend) | 0.8976 | 0.7835 | 0.9127 | 96.6 | 700 | 908 | +0.0024 |

Reading: the ILP is the most valuable single component (+0.037), then TTA (+0.009), line-fit (+0.007),
divisions, bidirectional fusion and the edge threshold (0.005 each). The detection threshold is flat across
0.95-0.975: the misses are not below-threshold peaks, they are displaced peaks (case A), so no threshold
recovers them. Blending two seeds is worth nothing here.

## 6. What to do about it, in order

1. **Motion relink off** -- done: the cross-check on three weight sets confirmed it (section 4), so
   `scripts/predict.py --preset tuned` (the new default) switches it off while the dataclass defaults stay
   faithful to the notebook (`--preset notebook`). Worth ~+0.01 locally on both embryos; needs a leaderboard
   confirmation, which is the next submission.
2. **Detection in crowded regions is the real ceiling** (four fifths of errors). Levers, cheapest first:
   more training data (the all-199 runs, in progress); the (3,3,3) max-pool peak picker merges peaks closer
   than 2 grid cells (3.25 um) and the 5 um matching in the training targets treats such pairs as one, so a
   finer peak picker or a target that keeps touching cells apart is the next experiment; and, for 6bba's dim
   cells, the opt-in intensity augmentations (`--aug intensity,noise`) in a training run.
3. **Do not spend time on**: detection/edge thresholds, gap closing, short-track filter, the three no-op
   repair passes, or the second seed.
4. The DeepCenter veto is inert here (2026-09-16 sweep) and could be dropped from the Kaggle package to
   save runtime if the budget gets tight.

## Appendix

**Scripts (new today).** `scripts/error_detection_misses.py` (per-annotated-cell profile: depth,
intensity, crowding, boundary, nearest prediction and its z/xy offset; `dist/error_analysis_0942/misses_{ilp,full}/`),
`scripts/error_stage_diff.py` (ILP vs full edge attribution under the metric's own FP definition, plus
per-error geometry; `dist/error_analysis_0942/stagediff{,_norelink}/`), `scripts/error_examples_0942.py`
(the figure panels). `scripts/analyze_errors.py` output: `dist/error_analysis_0942/errors_*.json`. Ablation
arms: `dist/score_ea_*.json`, predictions `dist/preds_val_ea_*`, launched with `scripts/slurm_score.sbatch`
(23 arms, one GPU each; `--stage raw` failed because the GEFF writer enforces in-degree <= 1, which the raw
candidate graph violates by design, so raw-candidate recall was not measured).

**Definitions.** Detected = matched within 7 um by the metric's per-frame bipartite matching. "Contradicted
endpoint" = the metric's FP rule: a non-TP predicted edge whose matched source has an annotated child or
whose matched target has an annotated parent. Crowding = distance to the nearest *predicted* node in the
frame (annotations are too sparse to measure crowding). Intensity = normalised frame value (the model's
input), max over a 3x3x3 grid neighbourhood at the annotated position, divided by the frame's 99th
percentile.

**Miss rate by depth** is flat (2.2-3.9 % across z bins) and by distance to the volume face is flat
(1.9-3.8 %); neither is a driver. The 6bba/44b6 intensity contrast in section 3 is the one strong
image-level signal.

**Per-volume, motion relink on vs off** (adjusted edge Jaccard): 44b6_1574802b 0.865 -> 0.895;
44b6_e57ff5c6 0.610 -> 0.677; 6bba_d1acb6ff 0.853 -> 0.875; 6bba_283bf9f1 0.940 -> 0.959; 6bba_c328f2fd
0.765 -> 0.782; 6bba_c27cba08 0.940 -> 0.953; 44b6_d754aa59 0.901 -> 0.914 (the jump video); regressions:
6bba_3abfe10a 0.741 -> 0.733, 6bba_3a1849c2 0.848 -> 0.843.

**Duplicate detections** are impossible at the ILP stage (the max-pool NMS enforces >= 3.25 um between
peaks; 0 of 393k nodes have a closer neighbour) and appear only as gap-closing synthetic nodes afterwards
(0.12 %).

**Numbers to carry forward.** Detector recall 97.1 % / localisation median 1.7 um; association recall
97.6 %; four fifths of counted errors detection-borne; motion relink -0.011.

## 7. Addendum (2026-09-18): is the evidence for a missed cell in the detection map?

Prompted by Ultrack's hypothesis-selection idea. `predict.py --dump-det-dir` writes the final per-frame
detection probability map (after TTA, dual-seed blend and retention guard) and `scripts/error_detection_ceiling.py`
looks within the 7 um matching radius of every annotated cell (`dist/error_analysis_0942/ceiling/`):

| of the 417 missed cells (ILP stage) | n | % |
|---|---|---|
| a local maximum **above the 0.965 threshold** within 7 um -- the detector emitted a candidate there | 221 | 53 |
| best local maximum within 7 um between 0.5 and 0.965 -- a lower-threshold hypothesis would exist | ~85 | 20 |
| best local maximum between 0.1 and 0.5 | ~20 | 5 |
| no local maximum within 7 um at all | 54 | 13 |
| (detected cells: 100 % have a local max > 0.965 within 7 um) | | |

So the largest single bucket of "missed" cells is not missed by the detector at all: the peak passed the
threshold and became a candidate, and the **ILP dropped it**. Across the 20 volumes the ILP keeps 393,178 of
446,067 detections, dropping 11.9 % (4 % to 26 % per volume; 25.9 % on `44b6_e57ff5c6`, the worst-scoring
volume). With appearance cost 0 and disappearance cost 2, a detection whose link candidates are weak or
absent (edge_prob below the 0.48 candidate threshold, or its neighbours in t+-1 themselves missing) costs 2
to keep and is discarded. In crowded regions, where the transformer's edge probabilities are lowest, that is
exactly where true cells get dropped.

Reading against sections 2-3: the "displaced / merged peak" story (case A) is real but smaller than the raw
numbers suggested -- the unmatched prediction 7-10 um away is usually a *neighbouring* surviving node, while
the cell's own peak existed and was removed downstream. Candidate inflation from a lower detection threshold
is modest (+11 local maxima per frame at 0.5, on ~200 candidates), so a hypothesis-style test is cheap, but
the first lever is the ILP's disappearance cost and the edge candidate threshold, which decide whether an
above-threshold peak with weak links survives. Arms for both are queued (results in the next report).
