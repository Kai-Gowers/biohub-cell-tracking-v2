# 2026-09-18 -- ILP disappearance cost and edge candidate threshold (follow-up to the ceiling diagnostic)

## Why these arms

The detection-map diagnostic (`reports/2026-09-17-error-analysis-0942.md`, section 7) found that 53 % of
the annotated cells we miss had an above-threshold peak within 7 um: the detector emitted the candidate and
the ILP dropped it, because no chain of links confident enough to pay the disappearance cost (2.0) ran
through it. The obvious question was whether the ILP's costs, or the 0.48 edge candidate threshold that
decides which links exist at all, were set wrong for our weights. All arms below run on top of the
relink-off configuration (baseline 0.9066 with our blend weights, 0.9355 with the shipped weights) on the
clean 20-volume split, one change at a time.

## Our blend weights (s314159 + s0 DDP best)

| arm | score | 44b6 | 6bba | det % | FP | FN | volumes better / worse |
|---|---|---|---|---|---|---|---|
| baseline (relink off) | 0.9066 | 0.8166 | 0.9178 | 96.7 | 638 | 866 | |
| disappearance 0.5 | 0.8990 | 0.7965 | 0.9116 | 97.2 | 699 | 835 | |
| disappearance 1.0 | 0.9046 | 0.8172 | 0.9156 | 97.0 | 664 | 841 | 6 / 14 |
| **disappearance 3.0** | 0.9106 | 0.8240 | 0.9212 | 96.4 | 598 | 887 | **18 / 2** |
| disappearance 4.0 | 0.9083 | 0.8026 | 0.9230 | 95.9 | 579 | 935 | 14 / 6 |
| appearance 1.0 (was 0) | 0.9011 | 0.8112 | 0.9128 | 96.5 | 660 | 860 | 11 / 9 |
| edge threshold 0.44 (was 0.48) | 0.9096 | 0.8216 | 0.9202 | 96.9 | 636 | 820 | 11 / 9 |
| edge threshold 0.40 | 0.9087 | 0.8206 | 0.9194 | 97.1 | 653 | 800 | 11 / 9 |
| detection threshold 0.90 (was 0.965) | 0.9052 | 0.8064 | 0.9175 | 96.7 | 641 | 873 | |
| division cost 0.8 / 1.6 (was 1.2) | 0.9040 / 0.9066 | | | | | | |
| **disappearance 3.0 + edge 0.44** | **0.9123** | 0.8240 | 0.9229 | 96.8 | 613 | 827 | **15 / 5** |
| disappearance 4.0 + edge 0.44 | 0.9111 | 0.8093 | 0.9249 | 96.4 | 590 | 867 | 15 / 5 |

## Shipped pack weights (leaky on these volumes; direction is what matters)

| arm | score | 44b6 | 6bba | det % | FP | FN | volumes better / worse |
|---|---|---|---|---|---|---|---|
| notebook configuration | 0.9175 | 0.8314 | 0.9305 | 98.3 | 659 | 639 | |
| relink off | 0.9355 | 0.8461 | 0.9480 | 98.3 | 570 | 607 | 16 / 4 vs notebook |
| relink off + disappearance 3.0 | 0.9364 | 0.8488 | 0.9484 | 98.1 | 548 | 625 | 16 / 4 |
| **relink off + disappearance 3.0 + edge 0.44** | **0.9403** | 0.8666 | 0.9486 | 98.4 | 559 | 566 | **15 / 5** |

## Reading

1. **Keeping more weakly linked detections does not help.** Lowering the disappearance cost recovers annotated
   cells (recall up, FN down) but every recovered cell brings more false links than correct ones. The
   candidates the ILP drops are mostly *not* the annotated cells' peaks; they are spurious or duplicate
   detections whose removal is doing good, and the annotated cells lost the same way are the minority we
   cannot separate by cost alone. The 53 % figure from the ceiling diagnostic was therefore a description
   of where the misses go, not a recipe. A stricter cost (3.0) wins on 18 of 20 volumes; 4.0 overshoots.
2. **A lower edge candidate threshold helps a little** (more links admitted, FN down, FP flat) and combines
   additively with the stricter disappearance cost: the two act on different things (which links exist vs
   how much evidence a track needs). Alone it splits the volumes 11 / 9, so it is only adopted as part of
   the pair, which is consistent on both weight sets (15 / 5 each).
3. **Detection threshold, division cost and appearance cost**: no gain. The appearance cost in particular
   suppresses divisions (83 false divisions, term 0.059) and loses.

## Decision

`scripts/predict.py --preset tuned` (the default) now carries three deviations from the notebook:
`output_motion_relink=False`, `disappearance_weight=3.0`, edge candidate threshold 0.44 (dual and single).
Together, on the clean split: blend 0.8952 -> 0.9123 (+0.017), single seed 0.8976 -> ~0.912, shipped
weights 0.9175 -> 0.9403 (+0.023). The dataclass defaults stay at the notebook's values and `--preset
notebook` reproduces it exactly. None of this is leaderboard-confirmed yet; the submission in flight tests
relink off alone with the shipped weights, so a second package with the full tuned preset is the natural
follow-up once that number lands (the difference between the two isolates the ILP/edge change).

Scores in `dist/score_ea_t_*.json` and `dist/score_ea_pack_t_*.json`.
