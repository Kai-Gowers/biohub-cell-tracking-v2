#!/bin/bash
# Submit a training run to the cluster's GPU partitions.
#
#   sbatch scripts/slurm_train.sbatch                      # defaults, see train.py --help
#   sbatch scripts/slurm_train.sbatch --epochs 40 --patience 5
#
# `short`/`medium`/`long` (12h/2d/5d) all schedule onto the same GPU nodes
# (v100 x4 per node) plus gb001 (h200 x8) -- pick based on expected runtime.
# `interactivegpu` (l40s x4) is capped at 1h, smoke-test only.
#SBATCH --job-name=cell-tracking-train
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gowers/biohub-cell-tracking-v2/dist/logs/%x-%j.out
# stderr is not split out separately -- SLURM merges it into --output above
# when --error is omitted, so this one file has everything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV=/projects/twist2d/gowers/miniconda3/envs/cell-tracking

echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "log: dist/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

# -u: unbuffered stdout/stderr. Without it, Python fully buffers output once
# it's redirected to a file (rather than a terminal), so `tail -f` on this
# log would sit silent for a long stretch and then dump a big chunk at once.
"$ENV/bin/python" -u "$REPO_ROOT/scripts/train.py" "$@"
