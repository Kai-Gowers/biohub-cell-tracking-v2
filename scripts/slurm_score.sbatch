#!/bin/bash
# Predict + score one checkpoint on its held-out split, on a GPU node.
#
#   sbatch scripts/slurm_score.sbatch <tag> <checkpoint> [extra predict.py args...]
#
# Writes dist/preds_val_<tag>/ and dist/score_<tag>.json.
#SBATCH --job-name=ct-score
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gowers/biohub-cell-tracking-v2/dist/logs/%x-%j.out

set -euo pipefail
REPO_ROOT=/home/gowers/biohub-cell-tracking-v2
ENV=/projects/twist2d/gowers/miniconda3/envs/cell-tracking
TAG="$1"; CKPT="$2"; shift 2
# Any further arguments are passed to predict.py verbatim (e.g. --tta-z,
# --link-radius-um 10, extra --checkpoint X for an ensemble).
PREDICT_ARGS=("$@")
cd "$REPO_ROOT"
echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "tag: $TAG  checkpoint: $CKPT"
"$ENV/bin/python" -u scripts/predict.py \
  --split dist/heldout_split.json \
  --checkpoint "$CKPT" \
  --test-dir data/biohub-cell-tracking-during-development/train \
  --out-dir "dist/preds_val_$TAG" \
  --dump-candidates "dist/preds_val_$TAG/candidates" "${PREDICT_ARGS[@]}"
"$ENV/bin/python" -u scripts/score_local.py \
  --geff-dir "dist/preds_val_$TAG" \
  --held-out "$CKPT" \
  --json-out "dist/score_$TAG.json"
