# tool-forge

Post-train a small instruction-tuned LLM for reliable multi-step **tool use**, evaluate it on the
public **Berkeley Function Calling Leaderboard (BFCL-V4)**, and serve it behind an OpenAI-compatible
API with a minimal ReAct loop — **all on a single 12 GB RTX 4070**.

Base model: **Qwen3-4B**. The résumé artifact is a `base → SFT → aligned` BFCL accuracy table.

## The constraint drives everything

One consumer GPU. On a desktop-driving 4070, ~2 GB goes to the display, leaving **~10 GB usable**.
Every choice (QLoRA 4-bit, adapter-disabled reference model, short context, batch-size-1 + grad
accumulation) exists to fit that budget. Measured: bf16 Qwen3-4B serving uses **7.56 GiB weights +
1.38 GiB KV cache** at `gpu_memory_utilization=0.80`.

## Pipeline (do not reorder)

1. **Baseline** — serve the *untouched* base, run BFCL-V4, record per-category accuracy *before* any training.
2. **Data** — normalize xLAM/ToolACE to one schema, render to the chat template, validate every gold call through the verifier.
3. **SFT** — QLoRA, train-on-completions (mask prompt tokens).
4. **Alignment** — DPO (mine near-miss preference pairs) → GRPO (verifiable reward) as a stretch.
5. **Eval** — BFCL per-category + custom metrics (JSON-validity %, schema-compliance %, hallucinated-tool rate).
6. **Serve + agent** — merge adapter, serve via vLLM, run a ReAct loop.

## Results (to fill)

| Model | BFCL-V4 overall | Simple | Multiple | Parallel | Multi-turn |
|-------|----------------:|-------:|---------:|---------:|-----------:|
| base (Qwen3-4B) | — | — | — | — | — |
| + SFT           | — | — | — | — | — |
| + aligned       | — | — | — | — | — |

## Status

- ✅ Scaffold: `uv` (Python 3.12.11), `ruff` + `mypy --strict` + `pytest`, all green.
- ✅ Pure core: `schema.py` (`ToolSpec` / `ToolCall` / `PreferencePair`), `verify.py` (JSON-Schema verifier), full tests.
- ✅ GPU verified, vLLM serving Qwen3-4B confirmed on WSL2.
- ⬜ Baseline BFCL number → next.

## Setup

```bash
uv sync                 # core deps + dev tools (ruff, mypy, pytest)
uv run ruff check . && uv run mypy && uv run pytest
```

## Serving (WSL2 — separate from the training env)

vLLM is installed imperatively (not in the lockfile) because serving and training have conflicting
torch pins. **Pin vLLM to 0.21.0** — 0.23 + CUDA 13 crashes on WSL2 (UVA unavailable).

```bash
uv pip install "vllm==0.21.0" --torch-backend auto
export VLLM_USE_FLASHINFER_SAMPLER=0     # FlashInfer's JIT sampler needs nvcc, which we don't install
uv run vllm serve Qwen/Qwen3-4B --gpu-memory-utilization 0.80 --max-model-len 4096
```

## Stack

Python 3.12 · `uv` · `transformers` + `trl` + `peft` + `bitsandbytes` (Unsloth backend) · `vllm` ·
`bfcl-eval` · `pydantic` v2 · `jsonschema`. Quality gates: `ruff`, `mypy --strict`, `pytest`.
