#!/usr/bin/env bash
set -euo pipefail

# Build the isolated vLLM serving env (.venv-serve). Kept separate from the primary
# .venv because vLLM 0.21.0 is a WSL-fragile, out-of-lockfile install; a repair here
# must not touch the training env. Runtime WSL pins live in serve_baseline.sh.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-serve"
PY="$VENV/bin/python"

uv venv "$VENV" --python 3.12.11

# vLLM 0.21.0 ships a CUDA-13 wheel: its _C extension links libcudart.so.13 and a
# cu13-built libtorch, so torch must also be cu13. --torch-backend auto resolves a
# cu128 torch that cannot load it, and uv 0.8.0 exposes no cu130 backend; pull the
# matching cu13 torch (torch 2.11.0+cu130, nvidia-cuda-runtime 13.x) from PyTorch's
# cu130 index instead. unsafe-best-match lets uv prefer the +cu130 build over PyPI.
uv pip install --python "$PY" "vllm==0.21.0" \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match

"$PY" -c "import vllm; print('vllm', vllm.__version__)"
echo "OK: .venv-serve ready. Serve with: ./scripts/serve_baseline.sh"
