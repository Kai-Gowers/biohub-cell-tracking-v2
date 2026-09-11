#!/bin/bash
# Score every periodic snapshot of one training run on its held-out split.
#
#   sbatch scripts/slurm_score_sweep.sbatch <name>
#
# Scores dist/models/detector_<name>_epoch*.pt, plus detector_<name>.pt (last)
# and detector_<name>_best.pt, into dist/score_<name>_epoch<N>.json etc.
#SBATCH --job-name=ct-sweep
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gowers/biohub-cell-tracking-v2/dist/logs/%x-%j.out

set -euo pipefail
REPO_ROOT=/home/gowers/biohub-cell-tracking-v2
ENV=/projects/twist2d/gowers/miniconda3/envs/cell-tracking
NAME="$1"
cd "$REPO_ROOT"
echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
for ck in dist/models/detector_${NAME}_epoch*.pt dist/models/detector_${NAME}.pt dist/models/detector_${NAME}_best.pt; do
  [ -e "$ck" ] || continue
  base=$(basename "$ck" .pt); tag=${base#detector_}
  [ "$tag" = "$NAME" ] && tag="${NAME}_last"
  echo "=== $tag"
  "$ENV/bin/python" -u scripts/predict.py --split dist/heldout_split.json --checkpoint "$ck" \
      --test-dir data/biohub-cell-tracking-during-development/train --out-dir "dist/preds_val_$tag" | tail -n 1
  "$ENV/bin/python" -u scripts/score_local.py --geff-dir "dist/preds_val_$tag" --held-out "$ck" \
      --json-out "dist/score_$tag.json" | grep -E "OVERALL|SCORE"
  rm -rf "dist/preds_val_$tag"   # ~20 geff dirs per snapshot; the score json is what we keep
done
