#!/bin/bash
# Submit a (resumable, self-chaining) training run to the cluster's GPU partitions.
#
#   sbatch scripts/slurm_train.sbatch --epochs 400 --seed 0 --out dist/models/pack_s0.pt
#   sbatch --partition=medium --time=2-00:00:00 scripts/slurm_train.sbatch --epochs 400 --max-hours 47 ...
#
# The trainer stops cleanly at --max-hours (pass a value below the wall limit)
# and, when CT_CHAIN=1 (default), this script resubmits itself with
# --dependency=afterany so the run continues from `<out>` until `--epochs`
# is reached (`afterok` would never fire after a TIMEOUT). Set CT_CHAIN=0 to
# run a single job.
#
# `short`/`medium`/`long` (12h/2d/5d) schedule onto v100 x4 nodes plus gb001
# (h200 x8); `interactivegpu` (l40s x4) is capped at 1h and only takes srun.
#SBATCH --job-name=ct-train
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gowers/biohub-cell-tracking-v2/dist/logs/%x-%j.out

set -euo pipefail
REPO_ROOT=/home/gowers/biohub-cell-tracking-v2
ENV=/projects/twist2d/gowers/miniconda3/envs/cell-tracking
export POLARS_PREFER_PKG=32
cd "$REPO_ROOT"

echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo n/a)"
echo "log: dist/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "args: $*"

# Locate --out and --epochs in the args so the chain can decide whether to continue.
OUT=""; EPOCHS=400
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --out) OUT="${args[$((i+1))]}";;
    --epochs) EPOCHS="${args[$((i+1))]}";;
  esac
done

"$ENV/bin/python" -u "$REPO_ROOT/scripts/train.py" "$@"

if [ "${CT_CHAIN:-1}" = "1" ] && [ -n "$OUT" ] && [ -e "$OUT" ]; then
  DONE=$("$ENV/bin/python" - "$OUT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ck["epoch"]))
PY
)
  if [ "$DONE" -lt "$EPOCHS" ]; then
    echo "epoch $DONE/$EPOCHS -- resubmitting to continue"
    sbatch --dependency=afterany:"$SLURM_JOB_ID" --partition="$SLURM_JOB_PARTITION" \
      --time="$(squeue -h -j "$SLURM_JOB_ID" -o %l)" --job-name="$SLURM_JOB_NAME" \
      "$REPO_ROOT/scripts/slurm_train.sbatch" "$@"
  else
    echo "epoch $DONE/$EPOCHS -- run complete"
  fi
fi
