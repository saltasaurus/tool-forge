#!/usr/bin/env bash
# Self-healing baseline runner: reruns `bfcl generate` until a pass completes
# cleanly, leaning on BFCL's incremental resume (it skips already-generated
# cases) to get past the WSL client-side hang on the last few stragglers.
#
# NOTE: deliberately NO `set -e` here — the loop depends on `timeout` returning
# non-zero when it kills a hung pass; `set -e` would abort on that expected
# failure. `set -uo pipefail` is still safe and worth keeping.
#
# Prereq: scripts/serve_baseline.sh must already be running in another terminal
# (this script only generates) and `nvidia-smi` must be clear of orphan engines.
set -uo pipefail

CATEGORY="simple_python"   # default = smoke; pass --category=single_turn,multi_turn for the full run
for arg in "$@"; do
  case "$arg" in
    --category=*) CATEGORY="${arg#--category=}" ;;
    *)            CATEGORY="$arg" ;;   # also accept a bare positional category
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for i in $(seq 1 20); do
  echo "=== pass $i: generating '$CATEGORY' ==="
  if timeout 3600 "$ROOT/run_baseline.sh" "$CATEGORY"; then
    echo "✅ completed on pass $i"
    exit 0
  fi
  echo "pass $i timed out or was interrupted; resuming missing cases..."
done

echo "⚠️  WARNING: did not complete after 20 passes — check the server and results."
exit 1
