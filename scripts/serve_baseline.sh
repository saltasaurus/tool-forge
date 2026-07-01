#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"

# WSL hard-won pins:
#  - native sampler (no FlashInfer JIT/nvcc)
#  - NO fp8 KV: fp8 forces the FlashInfer attention backend, which JIT-compiles
#    with nvcc we don't have. bf16 KV → prebuilt FLASH_ATTN, no nvcc.
#  - 0.90 util + 12288 ctx: bf16 KV pool is ~14k tokens here; 12288 fits one
#    request and clears BFCL's prompt + 4096-output reservation for live/most
#    multi_turn (longest multi_turn_long_context prompts may still overflow —
#    a genuine 12 GB ceiling, documented and applied to every model compared).
# CUDA runtime libs ship inside the venv (pip nvidia-* wheels). After a WSL/shell
# restart LD_LIBRARY_PATH is empty, so vllm's _C can't find libcudart.so.13.
# Point the linker at every bundled nvidia/*/lib dir (resolved by soname, so the
# cu12/cu13 copies don't clash). This must run before `vllm` is invoked.
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv-serve"
export LD_LIBRARY_PATH="$(echo "$VENV"/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve "$MODEL" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 12288 \
  --enforce-eager
