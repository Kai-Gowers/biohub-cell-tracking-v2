# Clean Approach — Lightweight Local CV Context


> Extracted from `clean-approach-lightweight-local-cv-no-hack.ipynb`.


# 🧬 Biohub 132 | Clean Short-Track Rescue + Light Local CV | No Hack

## Clean baseline 0.908 → recover only high-confidence five-node tracks

Kaggle has addressed the historical metric exploit, so **0.908 is treated as the legitimate clean baseline**. Biohub 132 targets a real model weakness: valid short trajectories may be removed by the global minimum-track-length filter.

This notebook keeps the attached baseline pipeline intact and enables a deliberately conservative adaptive rescue. Only existing components of length five are eligible, and they must have high learned-edge confidence and short mean physical displacement. No nodes or edges are fabricated.

**No hack:** no artificial hubs, fake forks, negative-time nodes, out-of-volume coordinates, cross-clip edges, or hidden-test labels.

> **Attach exactly these inputs before running:**
> 1. `Biohub - Cell Tracking During Development`
> 2. `pilkwang/biohub-tracking-support-pack-50ep-v1`
>
> The clean hidden-test `submission.csv` is written and validated first. The fixed-8 CV then runs under a guard and cannot overwrite the submission file.

## Controlled experiment

This is a clean post-processing A/B test from the legitimate `0.908` baseline.

| Setting | 0.908 baseline | Biohub 132 |
|---|---:|---:|
| Motion relaxed gate | `9.5 µm` | `9.5 µm` |
| Minimum retained track length | `6` | `6` |
| Adaptive rescue | disabled | **enabled** |
| Eligible rescued length | none | **exactly 5 nodes** |
| Minimum mean learned-edge probability | n/a | **`0.90`** |
| Maximum mean edge distance | n/a | **`2.75 µm`** |
| Rescue budget | n/a | **min(`0.6%` of nodes, `60`)** |

**Hypothesis:** fixed-8 CV retains meaningful node-recall and edge-FN headroom. A small set of highly coherent five-node components may be genuine trajectories incorrectly removed by the length-six cutoff.

## Pipeline overview

1. **3D detector** proposes candidate cell centers in each frame.
2. **Learned edge scorer** estimates which cells in adjacent frames should be linked.
3. **ILP graph builder** chooses a globally consistent set of detections and links.
4. **Post-processing** repairs short gaps, removes weak isolated fragments, smooths trajectories, and adds only geometrically safe divisions.
5. **Schema validation and diagnostics** check that the final `submission.csv` has valid rows, IDs, and graph statistics.

The main lesson behind this notebook is that cell-tracking competitions are graph problems. A small change to graph construction can matter as much as a detector threshold change.

## Selected settings

| Setting | Value |
|---|---:|
| Clean leaderboard baseline | `0.908` |
| Detection threshold | `0.96875` |
| ILP appearance/disappearance cost | `0.0 / 1.575` |
| Motion tight/relaxed gate | `6.0 / 9.5 µm` |
| Minimum track length | `6` |
| Adaptive short-track rescue | **enabled** |
| Rescue eligible length | **`5` only** |
| Minimum mean edge probability | **`0.90`** |
| Maximum mean edge distance | **`2.75 µm`** |
| Maximum rescued nodes | **`60` or `0.6%`** |
| Safe-division parent/sister radii | `4.66 / 8.5 µm` |

## Artifact and Dependency Setup

The notebook expects a compact support artifact containing the inference source,
trained weights, and optionally offline dependency wheels. The primary Kaggle
attachment path is:

```text
/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json
```

The setup cell first uses already-installed modules, then attached wheels, and
only attempts an internet install when explicitly enabled for local development.
By default, the artifact resolver accepts only the 50-epoch package named by
`TARGET_ARTIFACT_SLUG`; set `BIOHUB_ALLOW_ARTIFACT_FALLBACK=1` only for local
debugging against an older package.

## Predict Candidate Graphs

The inference step writes one `.geff` graph per test video. Keeping graph
prediction separate from CSV conversion makes the graph repair and diagnostics
transparent.

## Build `submission.csv`

Rows are streamed directly to disk with the required schema. This avoids holding
the full hidden-test submission table in memory.

The standard gap closer intentionally handles only one missing frame. Two-missing-frame repair is handled by the stricter `gap2` pass so that a loose environment override cannot introduce non-consecutive edges.

## Promotion rule

- Legitimate clean leaderboard baseline: `0.908`.
- This notebook changes only the adaptive short-track rescue branch.
- Promote only when fixed-8 LocalCV improves, node recall or edge TP increases, and FP growth remains controlled.
- Reject the candidate when rescue adds many rows without a positive paired CV delta.

## Fixed-8 Official-Spec Lite CV

The hidden-test `submission.csv` is written and protected before diagnostics. The fixed public-train split then measures whether rescued five-node components improve node recall and edge TP without adding excessive FP.

Inspect `short_track_rescue_components`, `short_track_rescue_nodes`, edge TP/FP/FN, node recall, and the paired score delta together.

## Submission output

After the Version completes, inspect the final Lite CV summary. Submit `submission.csv` only when the conservative rescue is activated and produces a positive paired LocalCV delta over the legitimate `0.908` clean baseline.
