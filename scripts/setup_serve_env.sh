#!/usr/bin/env bash
set -euo pipefail

# Build the isolated vLLM serving env (.venv-serve). Kept separate from the primary
# .venv because vLLM 0.21.0 is a WSL-fragile, out-of-lockfile install; a repair here
# must not touch the training env. Runtime WSL pins live in serve_baseline.sh.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-serve"
PY="$VENV/bin/python"

uv venv "$VENV" --python 3.12.11

# vLLM pulls its own torch/transformers; --torch-backend auto picks the cu128 wheel.
uv pip install --python "$PY" "vllm==0.21.0" --torch-backend auto

"$PY" -c "import vllm; print('vllm', vllm.__version__)"
echo "OK: .venv-serve ready. Serve with: ./scripts/serve_baseline.sh"
