#!/bin/bash
# Run files of independent shell commands, 4 at a time, one per GPU, on the
# interactivegpu node (L40S x4, 1h cap, QoS allows one running job per user,
# and the partition accepts only srun/salloc -- not sbatch). Use this when
# the short/medium/long GPU partitions are backlogged and the tasks are short
# (scoring a checkpoint takes ~40 s here).
#
#   setsid nohup srun --partition=interactivegpu --gres=gpu:4 --cpus-per-task=32 \
#       --mem=160G --time=01:00:00 -J ct-batch scripts/slurm_gpu_batch.sh \
#       <taskfile> [taskfile...] > dist/logs/ct-batch-<name>.out 2>&1 &
#
# Each non-blank, non-comment line of a taskfile is run with `bash -c` and
# CUDA_VISIBLE_DEVICES set to its GPU slot. Task files run in order; per-task
# output is interleaved into stdout prefixed with the task number, and a
# joblog with exit codes lands in dist/logs/.
set -uo pipefail
REPO_ROOT=/home/gowers/biohub-cell-tracking-v2
cd "$REPO_ROOT"
echo "job: ${SLURM_JOB_ID:-none}  host: $(hostname)  gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',')  start: $(date)"
overall=0
for TASKFILE in "$@"; do
  JOBLOG="dist/logs/batch-${SLURM_JOB_ID:-nojob}-$(basename "$TASKFILE" .txt).joblog"
  echo "=== tasks: $TASKFILE ($(grep -cve '^\s*$' -e '^\s*#' "$TASKFILE") lines)  $(date)"
  grep -ve '^\s*$' -e '^\s*#' "$TASKFILE" | parallel -j 4 --will-cite --lb --tagstring '[{#}]' \
    --joblog "$JOBLOG" 'export CUDA_VISIBLE_DEVICES=$(( {%} - 1 )); bash -c {}'
  rc=$?
  echo "=== $TASKFILE exit: $rc  failed tasks: $(awk 'NR>1 && $7!=0' "$JOBLOG" | wc -l)  $(date)"
  [ $rc -ne 0 ] && overall=$rc
done
exit $overall
