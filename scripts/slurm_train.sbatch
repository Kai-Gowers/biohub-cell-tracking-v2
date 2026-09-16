#!/bin/bash
# Submit a (resumable, self-chaining) training run to the cluster's GPU partitions.
#
#   sbatch scripts/slurm_train.sbatch --epochs 400 --seed 0 --out dist/models/pack_s0.pt
#   sbatch --partition=medium --time=2-00:00:00 scripts/slurm_train.sbatch --epochs 400 --max-hours 47 ...
#   # 4 GPUs on one node, same recipe (batch split across ranks, SyncBatchNorm), ~3.8x faster
#   # (reports/2026-09-14-4gpu-throughput-model-size-augmentation.md); resumes an existing <out>:
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=160G scripts/slurm_train.sbatch --ddp --epochs 400 --seed 0 --out dist/models/pack_s0.pt ...
#
# The trainer stops cleanly at --max-hours (pass a value below the wall limit)
# and, when CT_CHAIN=1 (default), this script resubmits itself with
# --dependency=afterany so the run continues from `<out>` until `--epochs`
# is reached (`afterok` would never fire after a TIMEOUT). Set CT_CHAIN=0 to
# run a single job.
#
# `short`/`medium`/`long` (12h/2d/5d) schedule onto a mixed pool (v100 x4,
# a100 x4, l40s x4, gb001 h200 x8); `interactivegpu` (l40s x4) is capped at 1h
# and only takes srun. The v100 nodes (g001, g002) run this recipe at ~3020 s
# per epoch vs ~1520 s on a100/l40s, so they are excluded
# (reports/2026-09-12-training-progress-exclude-v100.md).
#SBATCH --job-name=ct-train
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002
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

# Hand-over: CT_CANCEL_JOBNAME=<old job name> sbatch ... cancels every job of that name
# (running and its pending self-resubmission) the moment this job starts, so a run can be
# moved to another partition/GPU count without two jobs writing the same <out>.
if [ -n "${CT_CANCEL_JOBNAME:-}" ] && [ "$CT_CANCEL_JOBNAME" != "$SLURM_JOB_NAME" ]; then
  echo "cancelling old jobs named '$CT_CANCEL_JOBNAME': $(squeue -h -u "$USER" -n "$CT_CANCEL_JOBNAME" -o '%i(%T)' | tr '\n' ' ')"
  scancel -u "$USER" -n "$CT_CANCEL_JOBNAME" || true
  sleep 30   # let the old job's checkpoint write (tmp + atomic rename) settle before we read it
fi

# Locate --out and --epochs in the args so the chain can decide whether to continue.
OUT=""; EPOCHS=400
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --out) OUT="${args[$((i+1))]}";;
    --epochs) EPOCHS="${args[$((i+1))]}";;
  esac
done

# Fork: CT_FORK_FROM=<checkpoint> sbatch ... --out <new path> starts the run from a copy of an
# existing checkpoint when <new path> does not exist yet, so a variant (e.g. --ddp) can continue
# a run under a different file while the original keeps going.
# Always echo the hand-over variables: on 2026-09-14 two --ddp jobs silently started from
# scratch because CT_FORK_FROM never reached the job environment (reports/2026-09-14-ddp-runs-started-from-scratch.md).
echo "CT_FORK_FROM=${CT_FORK_FROM:-<unset>}  CT_CANCEL_JOBNAME=${CT_CANCEL_JOBNAME:-<unset>}  CT_CHAIN=${CT_CHAIN:-1}  out=${OUT:-<none>} exists=$([ -n "$OUT" ] && [ -e "$OUT" ] && echo yes || echo no)"
if [ -n "${CT_FORK_FROM:-}" ]; then
  if [ ! -e "$CT_FORK_FROM" ]; then echo "CT_FORK_FROM=$CT_FORK_FROM does not exist; refusing to start from scratch" >&2; exit 1; fi
  if [ -z "$OUT" ]; then echo "CT_FORK_FROM set but no --out given" >&2; exit 1; fi
  if [ -e "$OUT" ]; then
    echo "fork skipped: $OUT already exists (resuming it instead)"
  else
    echo "forking $OUT from $CT_FORK_FROM (epoch $("$ENV/bin/python" -c 'import sys,torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"])' "$CT_FORK_FROM"))"
    cp "$CT_FORK_FROM" "$OUT"
  fi
fi

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
