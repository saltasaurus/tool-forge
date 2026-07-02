#!/usr/bin/env bash
set -euo pipefail

# Phase A of the fast eval: run generate_vllm.py in the isolated .venv-serve.
# Env block copied verbatim from serve_baseline.sh — these WSL pins are load-bearing:
#  - LD_LIBRARY_PATH: after a shell restart it's empty, so vllm's _C can't find
#    libcudart.so.13. Point the linker at every bundled nvidia/*/lib dir (resolved
#    by soname, so cu12/cu13 copies don't clash). Must run before vllm is imported.
#  - VLLM_USE_FLASHINFER_SAMPLER=0: native sampler, no FlashInfer JIT (no nvcc).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-serve"
export LD_LIBRARY_PATH="$(echo "$VENV"/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$VENV/bin/python" "$ROOT/scripts/generate_vllm.py" "$@"
