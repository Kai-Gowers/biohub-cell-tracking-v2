# 2026-09-14 — Error analysis of the 0.842 pipeline (`main`, `detector_tmax30_epoch23.pt`)

> Detailed appendix. The short overview the PI asked for is `2026-09-14-error-analysis-0842.md`; read that first.


Requested by the PI: "numbers and figures, e.g. global statistics, qualitative examples" for the
original pipeline (2-frame U-Net detector + pair-MLP edge scorer + distance-gated greedy linker,
no divisions; leaderboard 0.842 on 2026-08-31). Everything below is on the 20 held-out volumes
(`dist/heldout_split.json`: 5 x `44b6`, 15 x `6bba`) that this checkpoint never trained on.

Figures: `reports/figures/2026-09-14-error-analysis-0842/fig01..fig11*.png`; all numbers:
`stats.json` in the same folder. Code: `scripts/error_report.py` on `main` (uncommitted, in the
worktree `/projects/twist2d/gowers/biohub-cell-tracking-v2-main`), on top of the existing
`cell_tracking.analysis.decompose_volume` (every counted edge error gets exactly one cause;
reconciles with the metric's TP/FP/FN).

## 0. Reproduction

Re-predicted with `--dump-candidates` (needed for rank / margin analysis) on one L40S in ~4 min
(`dist/preds_val_0842_dump`, `dist/score_0842_dump.json`). Held-out SCORE 0.8073 vs 0.8074 recorded
on 2026-08-31; ten volumes differ by 1-3 nodes (GPU nondeterminism), TP/FP/FN differ by at most 1.

## 1. Global statistics

### 1.1 Score and its decomposition (fig01, fig02)

| | all 20 | 44b6 (5) | 6bba (15) |
|---|---|---|---|
| held-out SCORE (weighted adjusted edge Jaccard) | **0.8073** | 0.6153 | 0.8327 |
| GT nodes / GT edges | 14177 / 13692 | 1444 / 1419 | 12733 / 12273 |
| detection recall (7 µm bipartite) | 95.2% | 93.0% | 95.4% |
| TP / FP / FN edges | 12036 / 1053 / 1656 | 1069 / 306 / 350 | 10967 / 747 / 1306 |
| association recall (both endpoints detected) | 94.6% | 86.0% | 95.5% |
| FN: detector-attributable (an endpoint undetected) | 948 (57.2%) | 174 (49.7%) | 774 (59.3%) |
| FN: linker-attributable (both detected, wrong/no link) | 690 (41.7%) | 173 (49.4%) | 517 (39.6%) |
| FN: division second child (structurally impossible) | 18 (1.1%) | 3 (0.9%) | 15 (1.1%) |
| FP: true partner detected, linker chose another | 767 (72.8%) | 227 (74.2%) | 540 (72.3%) |
| FP: true partner undetected | 282 (26.8%) | 79 (25.8%) | 203 (27.2%) |
| FP: both endpoints matched, wrong pair | 4 (0.4%) | 0 | 4 (0.5%) |
| greedy -> exact assignment on the same scores, ΔTP / ΔFP | -45 / +190 | -19 / +84 | -26 / +106 |
| global node ratio (pred / estimated true nodes) | 1.055 | 1.014 | 1.082 |
| GT divisions / predicted forks | 19 / 0 | 4 / 0 | 15 / 0 |

- The per-volume spread is large (fig01): 0.54-0.84 on `44b6`, 0.66-0.95 on `6bba`. The metric
  weight is the GT edge count, so `44b6_d5e7d891` (893 edges, 0.59) alone is 5x the weight of the
  other four `44b6` volumes together.
- **Headroom (micro Jaccard, ignoring the node-ratio adjustment):** current 12036/(12036+1053+1656)
  = 0.816. A perfect linker on the current detections (all linker FNs recovered, all
  partner-detected FPs removed) -> 0.910. A perfect detector with the current linker (all
  detector FNs recovered, partner-undetected FPs removed) -> 0.898. So the two halves are worth
  about the same, ~+0.09 each; neither alone reaches 0.9+ on `44b6`.
- Divisions: 19 GT divisions on the held-out set, none predicted, so the 0.1 x division-Jaccard
  term is exactly 0. It is worth at most +0.1 and realistically a few hundredths; the "division
  second child" FNs are only 1.1% of FN.
- Node ratio 1.055 costs little (`adjusted_jaccard` vs raw: 0.842 vs 0.880 on the worst volume,
  ratio 1.43, otherwise < 0.02). fig08: detections per frame are stable in time; two `44b6` volumes
  sit at ~620 detections/frame, the maximum density in the held-out set.

### 1.2 Association errors are a crowding phenomenon; detection errors are not (fig03)

Association recall (GT edges whose two endpoints were both detected) by the distance from the
source detection to its nearest other detection:

| nearest other detection | <5 µm | 5-6 | 6-7 | 7-8 | 8-10 | 10-14 | >=14 |
|---|---|---|---|---|---|---|---|
| all (n) | 50% (30) | 72% (129) | 74% (411) | 86% (865) | 92% (2874) | 96% (3919) | 99.2% (4482) |
| 44b6 | 75% | 78% | 74% | 84% | 89% | 92% | 100% |
| 6bba | 46% | 68% | 75% | 87% | 93% | 96% | 99.2% |

- Below 8 µm one association in four fails, above 14 µm one in 130. The curve is the *same* for
  both embryos at a given crowding; `44b6` is worse overall because more of its edges are crowded
  (fig03 middle/right, fig11: `44b6_d5e7d891` has 469 detections in the frame shown vs 626 for
  `6bba_3abfe10a` in a larger visible area -- but the `44b6` volume's GT nodes sit in its dense core).
- Recall also falls with the number of gated candidates (99.6% with <2 candidates, 82% with >=12).
- **Detection recall is flat** against local density (91-98% in every bin of "other detections
  within 15 µm"), against depth (z), and against time in the video (fig04). It only dips for the
  dimmest nodes (88% when the 3x5x5 box mean is < 2x the frame median; median intensity ratio of
  missed nodes 2.9 vs 5.4 for detected). Detection misses are therefore about contrast and
  appearance, not about neighbours.

### 1.3 Where the missed detections are (fig04, fig10)

Of 685 missed GT nodes: 91 (13%) had a detection within 7 µm that the bipartite matching had
already given to another GT node (two nuclei -> one detection, "merged pair"; fig10 row 3);
431 (63%) had their nearest detection at 7-10 µm (a neighbour's detection, or a detection displaced
by more than the 7 µm radius); 163 (24%) had nothing within 10 µm at all. Localisation error of
matched detections: median 1.76 µm on `6bba`, 2.09 µm on `44b6` (about one 1.625 µm grid cell);
p90 4.0 / 5.4 µm. The `44b6` tail is heavier, which pushes more `44b6` matches toward the 7 µm limit.

### 1.4 What the edge scorer gets wrong (fig05, fig06)

Rank of the true partner among the source's 15 µm-gated candidates (edges with both endpoints
detected):

| | rank 1 | rank 2 | rank 3 | >=4 | not a candidate |
|---|---|---|---|---|---|
| 44b6 (n=1238) | 89.4% | 6.3% | 2.3% | 1.9% | 0.1% |
| 6bba (n=11472) | 97.2% | 2.0% | 0.4% | 0.4% | 0.0% |

- Among the 382 wrong links (source detected and linked to the wrong detection, true target also
  detected): 66% have the true partner at rank 2, 28% at rank >= 3, and **6% at rank 1** -- the
  scorer preferred the true partner but the greedy pass had already given that target to a
  competing source (fig05 second panel, negative margins; fig09 row 1). The 15 µm gate loses
  nothing (0.0-0.1% not a candidate).
- Score margins are bimodal: 39% of wrong links have the chosen link within 0.1 of the true one
  (near-ties, fixable by a better-informed scorer), 34% are confident mistakes with margin > 0.5
  (fig09 row 2: true partner scored 0.01, chosen 0.99, rank 9).
- The scorer is well calibrated in the aggregate (fig05 third panel): true partners of TP edges pile
  up at ~1.0; the true partners of wrong-link edges are spread over 0-1; counted FP edges have scores
  near 1.0 -- i.e. **FPs are not low-confidence edges that a threshold would remove**, they are
  confident mistakes.
- Displacement: GT cells move 1.7-1.9 µm/frame (median), p99 8.7 µm; association recall drops from
  95% below 4 µm to 82% at 6-8 µm and 62% at 8-10 µm. Wrong links do *not* systematically pick the
  shorter hop (46% shorter than the true displacement, fig06 right), so the failure is not a
  distance prior; the MLP is confusing look-alike neighbours.
- Exact assignment (Hungarian) on the same scores is worse than greedy (-45 TP, +190 FP), as found
  on 2026-09-11 for the replicate checkpoint. The scores, not the selection rule, are the limit.

### 1.5 Track-level view (fig07)

| | 44b6 | 6bba |
|---|---|---|
| GT tracks (chains split at divisions) | 32 | 486 |
| median track length (edges) | 35 | 18 |
| tracks fully recovered | 12.5% | 51.6% |
| breaks per 100 GT edges | 24.5 | 10.5 |
| median longest correct segment (% of track) | 39% | 100% |

`44b6` tracks are twice as long and break 2.3x more often per edge, so a typical `44b6` lineage is
reproduced as three fragments. Half of `6bba` tracks are perfect end to end.

## 2. Qualitative examples (fig09, fig10, fig11)

Each panel is an xy max-projection over +-4 z-slices (+-6.5 µm) around the GT node, 41 x 41 voxels
(17 x 17 µm); green o = GT node, red + = detection, green arrow = GT edge, red dashed = predicted
edge, white ring = the node in question. Selected automatically by criterion, seed 0.

fig09, linker:
1. `44b6_1574802b` t=21: **greedy competition**. True partner scored 0.96 (rank 1) but that
   detection was claimed by another source; the cell was linked to a 0.63 alternative that is only
   1.4 µm away in xy but 9 µm away in z (drawn as a hollow square with its dz, since it lies outside
   the projected slab). The xy projection hides this kind of mistake.
2. `44b6_d5e7d891` t=56: **confident mistake in a dense field**. GT displacement 1.7 µm; the true
   partner scored 0.01 (rank 9 of ~10), the chosen neighbour 0.99. Nearest other detection 5.1 µm.
3. `6bba_5c824876` t=57: crowded (<7 µm), two touching nuclei, true partner rank 2 (0.65 vs 0.99).
4. `6bba_d1acb6ff` t=50: GT displacement 19 µm -- beyond the 15 µm gate, so the true partner is not
   a candidate. Only ~0.1% of edges; likely an annotation jump or a mitotic cell. Not worth a lever.
5. `44b6_d5e7d891` t=97: a correct link with the nearest other detection 5.8 µm away, for contrast.

fig10, detector:
1. `6bba_3abfe10a` t=43: **very dim, shallow** nucleus (z=9, intensity 0.5x the frame median),
   nothing detected within 16 µm. Looks like a limit of the imaging, not of the model.
2. `44b6_706092f0` t=50: two GT nuclei 3.6 µm apart, one detection between them -- the pair is
   resolved by the annotators but merged by the detector at the 4x-decimated 1.625 µm grid.
3. `44b6_d5e7d891` t=84: same merged-pair pattern in the dense core (nearest detection 5.9 µm,
   already matched to the neighbour).
4. `6bba_268e1230` t=60: a **large, very bright nucleus (17.6x median) with no detection within
   10 µm** -- the appearance of a cell about to divide / in prophase. The detector, trained on
   compact interphase nuclei, emits its peak at the neighbouring cell instead. This is the kind of
   miss that also blocks any future division modelling.

fig11: whole-frame projections of the worst `44b6` (0.59) and worst `6bba` (0.66) volume at t=50
with all detections; visually the detector covers every nucleus, and the sparse GT (3-9 nodes per
frame) shows why local scores are noisy per volume.

## 3. Takeaways for the PI

1. Errors split ~57/42 detector/linker by FN count, but the linker share is larger on the embryo
   that matters (`44b6`: 49/49, and 74% of its FPs are linker choices). Both halves are worth about
   +0.09 micro-Jaccard each if fixed in isolation; neither alone closes the `44b6` gap.
2. Association failure is a function of crowding and is identical across embryos at a given
   crowding; `44b6` is worse only because it is denser. Any cross-embryo generalisation claim should
   be checked at matched crowding.
3. The pairwise MLP is the weak link: the true partner is rank 2+ in 10.6% of crowded `44b6`
   associations, a third of wrong links are confident (margin > 0.5), and FP edges score ~1.0, so
   thresholding or exact assignment cannot help. A scorer that sees the competing candidates
   (what the 0.942 pipeline's node transformer does) is the right fix; it is what the
   `replicate-0942` branch is training now.
4. Detection misses are about appearance: dim/deep nuclei and large bright pre-mitotic nuclei; and
   merged pairs at < 4 µm spacing on the 1.625 µm grid. Not depth, not time, not density per se.
5. Divisions are currently worth exactly 0 of the available 0.1.

## 4. Caveats

- Ground truth is sparse (~7 annotated nodes per frame of ~60-620 detections), so 98.5% of
  predicted edges are ignored, not scored; over-linking is invisible here.
- "Crowding" is measured on detections, not GT (GT is too sparse); for undetected nodes the
  nearest-detection measure is biased, which is why the detector panel uses counts within 15 µm.
- Two embryos only; `44b6` has 32 GT tracks in the held-out set, so its track-level percentages are
  coarse.

**fig12_jump_44b6_d754aa59_t51.png** (added on request): frames 51 -> 52 of `44b6_d754aa59`, where the
whole field shifts ~8.7 µm in -y (top row: full frame, all detections and the one GT link of the
pair; bottom: 33 µm crop). The source cell's true partner is 9.3 µm up (scored 0.99, rank 2); the
linker chose the cell 9 µm *down* (scored 1.00). Nearest other detection 13.2 µm, so not a crowding
error: with two equally plausible ~9 µm hops and no notion of the field's motion, the pairwise scorer
guessed the wrong direction. Every neighbour moved -y, which a flow prior (or a scorer that sees
the neighbours) would have used. Browser case #545.
