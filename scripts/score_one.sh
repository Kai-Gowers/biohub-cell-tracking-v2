#!/bin/bash
# Predict + score one checkpoint on the held-out split (no SLURM directives;
# the body shared by slurm_score.sbatch and the batch runner).
#
#   scripts/score_one.sh <tag> <checkpoint> [extra predict.py args...]
#
# Writes dist/preds_val_<tag>/ and dist/score_<tag>.json.
#   KEEP_PREDS=0   delete dist/preds_val_<tag> afterwards (sweeps: the json is what we keep)
set -euo pipefail
REPO_ROOT=/home/gowers/biohub-cell-tracking-v2
ENV=/projects/twist2d/gowers/miniconda3/envs/cell-tracking
TAG="$1"; CKPT="$2"; shift 2
cd "$REPO_ROOT"
export POLARS_PREFER_PKG=32
echo "[$TAG] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-all} ckpt=$CKPT args=$*"
# Extra args go to predict.py: e.g. --secondary-checkpoint X --deepcenter Y --stage ilp --no-bidir
"$ENV/bin/python" -u scripts/predict.py \
  --split dist/heldout_split.json --checkpoint "$CKPT" \
  --test-dir data/biohub-cell-tracking-during-development/train \
  --out-dir "dist/preds_val_$TAG" --dump-stats "dist/preds_val_$TAG/stats.json" "$@" | tail -n 2
"$ENV/bin/python" -u scripts/score_local.py --geff-dir "dist/preds_val_$TAG" --split dist/heldout_split.json \
  --json-out "dist/score_$TAG.json" | grep -E "OVERALL|SCORE|n="
[ "${KEEP_PREDS:-1}" = "1" ] || rm -rf "dist/preds_val_$TAG"
echo "[$TAG] done"
