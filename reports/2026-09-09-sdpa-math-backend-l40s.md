# SDPA fused backends fail outright on L40S; forced MATH backend in TemporalAttention3d

Moving training off Kaggle onto the BC HPC cluster (SLURM, L40S/V100/H200 GPUs,
torch 2.6.0+cu124), the very first smoke-test training step crashed inside
`TemporalAttention3d.forward`:

```
RuntimeError: CUDA error: invalid configuration argument
  ... F.multi_head_attention_forward -> scaled_dot_product_attention
```

## Isolated repro

`nn.MultiheadAttention(channels=32, heads=4)` called with the shape this
module actually produces at the first attention stage -- `batch=4, Z=Y=X=32`
flattened to a `(4*32^3, T=2, C=32)` token batch (131072) -- on an L40S
(g019, `interactivegpu` partition):

| SDPA backend | fp32 | fp16 | bf16 |
|---|---|---|---|
| default (auto-select) | FAIL | FAIL | FAIL |
| FLASH_ATTENTION | FAIL (fp32: "no available kernel") | FAIL | FAIL |
| EFFICIENT_ATTENTION | FAIL | FAIL | FAIL |
| CUDNN_ATTENTION | FAIL (no available kernel) | -- | -- |
| **MATH** | **OK** | **OK** | **OK** |

Every fused kernel raises `CUDA error: invalid configuration argument`
regardless of autocast dtype -- this is not the previously-documented
dropout/RNG-offset 65535 limit (dropout is already off in this module for
that reason), it is a separate, harder failure: the fused kernels don't
launch at all at this flattened-batch size on this GPU/driver/torch
combination. `MATH` (the unfused, no-special-kernel fallback) is the only
backend that handles it.

## Fix

`TemporalAttention3d.forward` now wraps the `nn.MultiheadAttention` call in
`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`, forcing the correct
backend unconditionally rather than only on failure.

## Status

Confirmed via the isolated repro above and via `scripts/train.py --epochs 1
--limit-volumes 2` running to completion after the fix, on an L40S. Not
re-verified against V100/H200 nodes on this cluster, or against Kaggle's
GPUs -- if fused backends work fine there, forcing MATH is a (likely small,
unmeasured) throughput cost on those, in exchange for portability. Revisit
if training speed becomes a concern on hardware where the fused kernels
would have worked.
