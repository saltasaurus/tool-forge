#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from the script's own location → absolute result-dir,
# which escapes BFCL's "relative-to-package-root" trap regardless of CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_HANDLER="Qwen/Qwen3-4B-Instruct-2507-FC"
RUN="${RUN:-instruct}"           # which run these results belong to; BFCL nests <model-handler>/ under it
CATEGORY="${1:-simple_python}"   # simplest; pass single_turn,multi_turn for the full run

export LOCAL_SERVER_ENDPOINT=localhost LOCAL_SERVER_PORT=8000

# Ensure scripts/serve_baseline.sh is running first
bfcl generate \
  --model "$MODEL_HANDLER" \
  --test-category "$CATEGORY" \
  --temperature 0.0 \
  --backend vllm \
  --skip-server-setup \
  --result-dir "$ROOT/runs/$RUN/bfcl/results" \
  --num-threads 4
  