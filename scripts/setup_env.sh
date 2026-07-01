#!/usr/bin/env bash
set -euo pipefail

# Build the project's PRIMARY environment (.venv): core + data pipeline +
# training stack (SFT/DPO/GRPO) + dev tools. This is the default env — tests,
# tooling, and `python -m tool_forge.*` all run here.
#
# The vLLM *serving* stack lives in a separate .venv-serve (see setup_serve_env.sh)
# because it is an imperative, out-of-lockfile, WSL-fragile install (exact 0.21.0
# pin, orphan EngineCore, LD_LIBRARY_PATH dance). Keeping it apart means a vLLM
# repair can't clobber training and vice versa. (The stacks co-resolve today; this
# is blast-radius isolation, not a version conflict.)
#
# Pins below are what resolved on this machine (RTX 4070, CUDA 12.8, WSL2).
#
# WSL2 note: after a shell restart LD_LIBRARY_PATH is empty and a bare
# `import bitsandbytes` may fail to find libcudart. torch is imported first in the
# training path, which loads the CUDA runtime, so bnb resolves; if you import bnb
# standalone and it errors, point LD_LIBRARY_PATH at
# .venv/lib/python3.12/site-packages/nvidia/*/lib (see serve_baseline.sh).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

# Fresh interpreter on the project's pinned Python.
uv venv "$VENV" --python 3.12.11

# torch first, alone: the keystone every other package pins against.
# --torch-backend auto detects CUDA 12.8 and selects the +cu128 wheel.
uv pip install --python "$PY" torch==2.11.0 --torch-backend auto

# Training stack, resolved against that torch.
uv pip install --python "$PY" \
  transformers==5.12.1 \
  trl==1.7.0 \
  peft==0.19.1 \
  bitsandbytes==0.49.2 \
  accelerate==1.14.0

# Our package (editable) + its declared deps (datasets, pydantic, scikit-learn, ...).
uv pip install --python "$PY" -e "$ROOT"

# Dev toolchain (the `dev` dependency-group — not pulled by `-e .`).
uv pip install --python "$PY" pytest mypy ruff types-jsonschema

# Gate: clean imports + CUDA visible. Exit nonzero if the env is broken so a failed
# setup never looks like success.
"$PY" - <<'EOF'
import torch, transformers, trl, peft, accelerate, bitsandbytes as bnb
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| bnb", bnb.__version__)
assert torch.cuda.is_available(), "CUDA not visible — training would run on CPU"
EOF

echo "OK: .venv ready (dev + training). Activate with: source .venv/bin/activate"
