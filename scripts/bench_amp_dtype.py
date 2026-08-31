#!/usr/bin/env python3
"""Isolate why bf16 autocast measured ~6x slower than fp16 on Kaggle.

cudnn.benchmark=True already ruled out algorithm *selection* as the cause
(reports/2026-08-30-cudnn-benchmark-bf16-slowdown.md) -- this times the
model's two structurally different pieces separately (the Conv3d/
InstanceNorm3d encoder-decoder stack, and TemporalAttention3d's
nn.MultiheadAttention at its unusually large flattened-batch/short-sequence
shape, see models/detector.py's docstring) under both dtypes, so a Kaggle
run can show which one is actually slow instead of guessing.

    python scripts/bench_amp_dtype.py

Needs a CUDA GPU (run on Kaggle, not locally -- this repo's dev machine has
no CUDA). Takes well under two minutes; no data, no checkpoint required.
"""

from __future__ import annotations

import time

import torch

from cell_tracking.config import ATTN_HEADS, UNET_BASE_CHANNELS
from cell_tracking.models.detector import ConvBlock3d, TemporalAttention3d, UNet3D

WARMUP = 3
ITERS = 10
BATCH = 4
WINDOW = 2


def _time_step(step_fn, dtype: torch.dtype) -> float:
    for _ in range(WARMUP):
        step_fn(dtype)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        step_fn(dtype)
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS


def bench_full_model(device: torch.device) -> dict[str, float]:
    model = UNet3D().to(device)
    x = torch.randn(BATCH, WINDOW, 64, 64, 64, device=device)
    y = torch.randint(0, 2, (BATCH, 64, 64, 64), device=device, dtype=torch.float32)

    def step(dtype: torch.dtype) -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x)[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

    return {"fp16": _time_step(step, torch.float16), "bf16": _time_step(step, torch.bfloat16)}


def bench_conv_stack(device: torch.device) -> dict[str, float]:
    # Stage-0 shape: (B*T, 1, 64, 64, 64) -> (B*T, base_channels, 64, 64, 64),
    # the heaviest single ConvBlock3d in the model (full spatial resolution).
    block = ConvBlock3d(1, UNET_BASE_CHANNELS).to(device)
    x = torch.randn(BATCH * WINDOW, 1, 64, 64, 64, device=device)

    def step(dtype: torch.dtype) -> None:
        block.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = block(x)
            loss = out.float().pow(2).mean()
        loss.backward()

    return {"fp16": _time_step(step, torch.float16), "bf16": _time_step(step, torch.bfloat16)}


def bench_attention(device: torch.device) -> dict[str, float]:
    # First attention stage shape (after one MaxPool3d(2)): (B, T, C, 32, 32, 32) --
    # the docstring's own "4 x 32^3" flattened-batch example.
    attn = TemporalAttention3d(UNET_BASE_CHANNELS, ATTN_HEADS).to(device)
    x = torch.randn(BATCH, WINDOW, UNET_BASE_CHANNELS, 32, 32, 32, device=device)

    def step(dtype: torch.dtype) -> None:
        attn.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = attn(x)
            loss = out.float().pow(2).mean()
        loss.backward()

    return {"fp16": _time_step(step, torch.float16), "bf16": _time_step(step, torch.bfloat16)}


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required -- run this on Kaggle.")
    device = torch.device("cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("bf16 natively supported:", torch.cuda.is_bf16_supported())
    print(f"torch {torch.__version__}  cudnn {torch.backends.cudnn.version()}")

    for benchmark_mode in (False, True):
        torch.backends.cudnn.benchmark = benchmark_mode
        print(f"\n=== cudnn.benchmark={benchmark_mode} ===")
        for name, fn in (
            ("full_model", bench_full_model),
            ("conv_stack (stage-0 ConvBlock3d)", bench_conv_stack),
            ("attention (first-stage TemporalAttention3d)", bench_attention),
        ):
            result = fn(device)
            ratio = result["bf16"] / result["fp16"] if result["fp16"] else float("nan")
            print(
                f"  {name:45s} fp16={result['fp16']*1000:7.1f}ms  "
                f"bf16={result['bf16']*1000:7.1f}ms  bf16/fp16={ratio:5.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
