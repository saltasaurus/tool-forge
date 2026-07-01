# tool-forge

Teach reliable multi-step **tool use** to a raw pretrained LLM via post-training (SFT → alignment),
evaluate on the public **Berkeley Function Calling Leaderboard (BFCL-V4)**, and serve behind an
OpenAI-compatible API with a minimal ReAct loop — **all on a single 12 GB RTX 4070**.

Training target: **Qwen3-4B-Base** (pretrained only — no instruction or tool-use tuning). Reference
ceiling: **Qwen3-4B-Instruct-2507**, the same base after the vendor's own post-training. The result is
a `base → SFT → aligned` BFCL accuracy table measuring how much of the instruction-tuned ceiling
post-training recovers on a single GPU.

## The constraint drives everything

One consumer GPU. On a desktop-driving 4070, ~2 GB goes to the display, leaving **~10 GB usable**.
Every choice (QLoRA 4-bit, adapter-disabled reference model, short context, batch-size-1 + grad
accumulation) exists to fit that budget. Measured: bf16 Qwen3-4B-Instruct-2507 serving uses
**~7.64 GiB weights + ~2.5 GiB KV cache** (18.5k tokens) at `gpu_memory_utilization=0.90`,
`--max-model-len 12288`.

## Pipeline (do not reorder)

1. **Anchors** — measure both BFCL-V4 references before training: the **Instruct-2507 ceiling** (the vendor's post-trained model) and the **Qwen3-4B-Base floor** (pretrained only).
2. **Data** — normalize xLAM/ToolACE to one schema, render to the chat template, validate every gold call through the verifier.
3. **SFT** — QLoRA on the base model, completion-only loss (prompt tokens masked to `-100`).
4. **Alignment** — DPO (mine near-miss preference pairs) → GRPO (verifiable reward) as a stretch.
5. **Eval** — BFCL per-category + custom metrics (JSON-validity %, schema-compliance %, hallucinated-tool rate).
6. **Serve + agent** — merge adapter, serve via vLLM, run a ReAct loop.

## Architecture

The spine is `schema.py` — every module speaks its types; nothing speaks "xLAM-ese" past
`normalize`. The pure core (`schema` / `verify` / `normalize` / `split` / `format`) is I/O-free and
unit-tested; the edge (HF load, W&B, vLLM, trainers) is where side effects live.

**Data flow** (✅ built · 🔨 in progress · ⬜ planned):

```mermaid
flowchart LR
    XLAM[("xLAM rows<br/>HF, gated")]
    NORM["normalize.py ✅<br/>row → Conversation"]
    VERIFY["verify.py ✅<br/>validate gold calls"]
    SPLIT["split.py ✅<br/>seeded → train/dev/test"]
    FORMAT["format.py ✅<br/>→ chat template"]
    WANDB[("W&B artifact")]
    SCHEMA["schema.py ✅<br/>shared types"]

    XLAM --> NORM --> VERIFY --> SPLIT --> FORMAT --> WANDB
    SCHEMA -. types .-> NORM
    SCHEMA -. types .-> VERIFY
    SCHEMA -. types .-> SPLIT
    SCHEMA -. types .-> FORMAT
    VERIFY -. quarantine counts<br/>converter feedback .-> NORM
```

`verify.py` is reused three times — Phase 1 (filter gold), Phase 3 (grade generated calls, *strict*),
Phase 5 (guard agent calls) — which is why it stays pure and dependency-free.

**Core types** (`schema.py`):

```mermaid
classDiagram
    class ToolSpec {
        +str name
        +str description
        +dict parameters
    }
    class ToolCall {
        +str name
        +dict arguments
        +str id
    }
    class Conversation {
        +int id
        +str query
        +dict~str,ToolSpec~ tools
        +tuple~ToolCall~ gold_calls
    }
    class PreferencePair {
        +str query
        +dict~str,ToolSpec~ tools
        +ToolCall chosen
        +ToolCall rejected
        +VerificationOutcome rejection_reason
    }
    class VerificationResult {
        +VerificationOutcome result
        +str detail
    }
    Conversation o-- ToolSpec : tools
    Conversation o-- ToolCall : gold_calls
    PreferencePair o-- ToolSpec : tools
    PreferencePair o-- ToolCall : chosen / rejected
    VerificationResult ..> VerificationOutcome
```

`Conversation` (SFT) and `PreferencePair` (DPO) are siblings — both carry `query` + `tools`; one holds
gold calls, the other a chosen/rejected pair.

## Results

Reference ceiling — `Qwen/Qwen3-4B-Instruct-2507-FC`, **untouched**, on BFCL-V4 — the post-trained
target the base-model curve is measured against. The **Qwen3-4B-Base** floor is not yet measured.
Repro: greedy (`temperature=0.0`), `bfcl-eval==2025.12.17`, vLLM 0.21.0, bf16, `--max-model-len 12288`.

| Stage | Non-Live AST | Live | Multi-turn | Overall (full V4) |
|-------|-------------:|-----:|-----------:|------------------:|
| Base (floor)              | — | — | — | — |
| + SFT                     | — | — | — | — |
| + aligned                 | — | — | — | — |
| *Instruct-2507 (ceiling)* | *88.31%* | *76.31%* | *17.50%* | _partial¹_ |

¹ BFCL's single "Overall" blends sections not yet run (multi-turn, agentic), so it is **not**
meaningful until those land — track the section columns, not the headline. Agentic (web search,
memory) is **excluded by design** (out of training distribution; non-deterministic external services).

<details><summary>Instruct-2507 ceiling — per-category (single-turn)</summary>

| Non-Live (hand-written) | Acc | Live (real prompts) | Acc |
|---|--:|---|--:|
| simple_python      | 95.25% | live_simple            | 78.68% |
| simple_java        | 64.00% | live_multiple          | 76.16% |
| simple_javascript  | 68.00% | live_parallel          | 62.50% |
| multiple           | 94.50% | live_parallel_multiple | 66.67% |
| parallel           | 93.00% | live_relevance         | 87.50% |
| parallel_multiple  | 90.00% | live_irrelevance       | 81.22% |
| irrelevance        | 89.17% | | |

**Reading:** the ceiling is strong on Python AST (90–95%) and weak on **non-Python** (`java` 64%,
`javascript` 68%) and **live parallel** (62–67%) — the accuracy shape the base-model curve is measured against.
</details>

<details><summary>Instruct-2507 ceiling — per-category (multi-turn)</summary>

| Multi-turn category | Acc |
|---|--:|
| base          | 25.50% |
| miss_func     | 21.50% |
| miss_param    | 17.50% |
| long_context  |  5.50% |
| **overall**   | **17.50%** |

**Reading:** the expected difficulty gradient — `base` highest, `long_context` lowest. Stateful
multi-step execution is the hardest BFCL section for a 4B; the clean spread (not a flat near-zero)
confirms the harness scores real behavior. Multi-turn lies outside the single-turn training
distribution used here — reported for context, not a training target.
</details>

## Status

- ✅ Scaffold: `uv` (Python 3.12.11), `ruff` + `mypy --strict` + `pytest`, all green.
- ✅ Pure core: `schema.py` (`ToolSpec` / `ToolCall` / `PreferencePair`), `verify.py` (JSON-Schema verifier), full tests.
- ✅ GPU verified, vLLM serving Qwen3-4B-Instruct-2507 confirmed on WSL2.
- ✅ Instruct ceiling (single-turn): Non-Live AST **88.31%**, Live **76.31%** (`Qwen3-4B-Instruct-2507-FC`, greedy).
- ✅ Instruct ceiling (multi-turn): Overall **17.50%** (base 25.50% / miss_func 21.50% / miss_param 17.50% / long_context 5.50%).
- ✅ Phase 1 (data pipeline): `normalize` → `Conversation`, `verify`, `split`, `format`, W&B artifact — **done**. Full-corpus gold-call validity **98.43%** (1.57% xLAM noise quarantined via `verify`).
- ✅ Environments: primary `.venv` (core + data + QLoRA training + dev tools) and isolated `.venv-serve` (vLLM), both script-built.
- ✅ QLoRA verified: 4-bit base + LoRA (**0.81%** trainable, ~3.5 GB resident); SFT smoke run passes with token-level completion-masking confirmed, ~5 GB peak.
- 🔨 Phase 2 (SFT): `models.py` (`ModelSpec` + loaders) and `dataset.render_prompt_completion` (+ tests) **done**; base-model SFT run next.
- ⬜ Base floor eval · alignment (DPO → GRPO) · serve + ReAct agent.

## Setup

The primary environment `.venv` holds the core, data pipeline, CUDA training stack, and dev tools. The
training stack is installed imperatively (not in the lockfile), so **`uv sync` / `uv run` must not be
used** — they would prune it. Invoke through `.venv/bin/python` or an activated shell.

```bash
./scripts/setup_env.sh                          # build .venv (core + data + training + dev tools)
.venv/bin/python -m pytest                      # run the suite (ruff, mypy also installed)
```

## Serving + baseline eval (WSL2 — runs in a separate `.venv-serve`)

vLLM lives in its own `.venv-serve`, imperatively installed and isolated from the primary `.venv`
because it is a WSL-fragile, out-of-lockfile install whose repair must not clobber training. Build it
once, then serve + eval:

```bash
./scripts/setup_serve_env.sh                 # build .venv-serve (vllm==0.21.0)
./scripts/serve_baseline.sh                  # terminal A: serve, wait for "Application startup complete"
./scripts/run_baseline_auto.sh single_turn   # terminal B: generate (resume loop survives WSL hangs)
./scripts/eval_bfcl.sh single_turn           # score → bfcl_scores/ (CPU-only; reads sealed gold)
```

Hard-won WSL2 pins, all encoded in `serve_baseline.sh`:

- **`vllm==0.21.0`** — 0.23 + CUDA 13 crashes on WSL2 (UVA unavailable).
- **`VLLM_USE_FLASHINFER_SAMPLER=0` and no `--kv-cache-dtype fp8`** — anything that makes vLLM
  JIT-compile a FlashInfer kernel needs `nvcc`, which isn't installed. fp8 forces FlashInfer; bf16
  KV stays on prebuilt `FLASH_ATTN`.
- **`--max-model-len 12288`** (not 4096) — BFCL reserves up to 4096 output tokens against the model's
  *native* 262K context, so the served window must hold `prompt + 4096`; 4096 overflows real prompts.
- **`--gpu-memory-utilization 0.90 --enforce-eager`** — fits bf16 weights + KV in ~10 GB usable.
- **`LD_LIBRARY_PATH` → venv `nvidia/*/lib`** — CUDA libs ship as pip wheels in the venv; after a
  WSL/shell restart the linker can't find `libcudart.so.13` until this is re-exported.
- **Reap orphans after a crash** — `vllm serve` spawns an `EngineCore` child that Ctrl-C does *not*
  kill (it keeps holding VRAM): `pkill -9 -f 'EngineCore'` (also `bin/vllm`, `bin/bfcl`).

Eval is pinned to `bfcl-eval==2025.12.17`. `--skip-server-setup` points BFCL at the running server;
greedy decoding (`--temperature 0.0`) for reproducibility. Result/score dirs use **absolute** paths
because BFCL resolves relative `--result-dir`/`--score-dir` against its *package* root, not the invocation CWD.
**Agentic categories (web search, memory) are excluded** — they need live external services.

## Stack

Python 3.12 · `uv` · `transformers` + `trl` + `peft` + `bitsandbytes` · `vllm` ·
`bfcl-eval` · `pydantic` v2 · `jsonschema`. Quality gates: `ruff`, `mypy --strict`, `pytest`.
