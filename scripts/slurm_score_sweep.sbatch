#!/bin/bash
# Score every periodic snapshot of one training run on its held-out split.
#
#   sbatch scripts/slurm_score_sweep.sbatch <name> [extra predict.py args...]
#
# Scores dist/models/<name>_epoch*.pt, plus <name>.pt (last) and <name>_best.pt,
# into dist/score_<name>_epoch<N>.json etc. Extra args (e.g. --deepcenter X)
# go to predict.py for every checkpoint.
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
NAME="$1"; shift
cd "$REPO_ROOT"
echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
for ck in dist/models/${NAME}_epoch*.pt dist/models/${NAME}.pt dist/models/${NAME}_best.pt; do
  [ -e "$ck" ] || continue
  tag=$(basename "$ck" .pt)
  [ "$tag" = "$NAME" ] && tag="${NAME}_last"
  echo "=== $tag"
  KEEP_PREDS=0 scripts/score_one.sh "$tag" "$ck" "$@"
done
