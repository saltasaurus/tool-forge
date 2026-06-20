# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active guided course — resume here

This project is being built as a **guided course**: the user writes all project code; Claude acts as teacher/advisor (Socratic + lecture, **full rigor**, **small steps**). **If you are a new session, read [COURSE_LOG.md](COURSE_LOG.md) before doing anything** — it holds the course settings, the teaching contract, and the exact next action. Do not scaffold or write project code on the user's behalf; guide them to write it.

## Project status: greenfield

This repo currently contains **only a design document** — [LLM_Project-Function-Calling-Post-Training.md](LLM_Project-Function-Calling-Post-Training.md) — and no code, no `pyproject.toml`, and no commits yet. That document is the **authoritative spec**; when building, follow its Section 6 (repository layout) and Section 7 (phase plan). Update the design doc if a decision changes; don't let code and spec drift.

## What this is

`tool-forge`: take a small instruction-tuned LLM that fails at multi-step tool use, post-train it (SFT → preference alignment), evaluate on the public **Berkeley Function Calling Leaderboard (BFCL-V4)**, and serve it behind an OpenAI-compatible API with a minimal ReAct agent loop — **all on a single 12 GB RTX 4070**. The resume artifact is a `base → SFT → aligned` BFCL accuracy table.

The recommended base model is **Qwen3-4B** (fallback: Qwen2.5-3B-Instruct).

## The 12 GB VRAM constraint drives the architecture

Almost every non-obvious design choice exists to fit on one consumer GPU. Don't "simplify" these away:

- **QLoRA (4-bit NF4) everywhere** for training; `adamw_8bit`, gradient checkpointing, short `max_length`, `per_device_batch_size=1` + gradient accumulation.
- **Reference model via adapter-disabling**, not a second model in VRAM — DPO/GRPO get π_ref for free by turning the LoRA adapters off.
- **DPO is the tightest training step** (it processes chosen *and* rejected per step, ~2× activations). **GRPO is tighter still** (samples G generations per prompt). If either OOMs at 4B, drop the alignment base to 1.5–2B — *the method is the résumé signal, not the parameter count.*
- Always run a 30-step smoke test and read peak `nvidia-smi` before committing to a `(model, context, batch)` config.

## The verifier is the load-bearing component

`src/tool_forge/align/verifier.py` (parse + validate a tool call against its JSON Schema) powers **three** things: preference-pair mining, the GRPO reward, *and* the custom eval suite. It must be a **pure function** (no I/O, no global state) and gets the most test coverage. Same purity rule for `data/format.py`.

## Pipeline / phase order (do not reorder)

1. **Baseline first.** Serve the *untouched* base model with vLLM and run BFCL-V4 before any training. This "before" number is half the story — never skip it.
2. **Data** — normalize xLAM / ToolACE rows to one internal schema, render to the base model's exact **chat template**, validate every gold call through the verifier.
3. **SFT** (QLoRA, train-on-completions — mask prompt tokens from loss). Expect the biggest BFCL jump here.
4. **Preference alignment** — pick one path: **DPO** (Path A, recommended, lower risk; mine near-miss pairs from the SFT model's own samples) or **GRPO** (Path B, stretch; verifiable reward from the verifier, no reward model).
5. **Eval** — BFCL-V4 per-category + custom format metrics (JSON-validity %, schema-compliance %, hallucinated-tool rate, exact-call accuracy).
6. **Serve + agent** — merge adapter, optionally AWQ/GPTQ quantize, serve via vLLM, measure tokens/sec and p50/p95 latency, run the ReAct loop.

## Sealed-split discipline

The test split is **never looked at until the final eval**. Splits are deterministic, hash-based, and seeded (`data/split.py`) so they're reproducible. **BFCL is held out entirely** as the external test set — never train or tune on it.

## Toolchain & intended commands

Python 3.12 with `uv`. Standing quality gates: `ruff`, `mypy --strict`, `pytest`. Training via HF `transformers` + `trl` (SFTTrainer / DPOTrainer / GRPOTrainer) + `peft` + `bitsandbytes`, with **Unsloth** as the low-VRAM backend. Eval via `bfcl-eval` (pin the exact version). Serving via `vllm`. Tracking via Weights & Biases; CI via GitHub Actions (ruff, mypy --strict, unit tests on push).

Planned entrypoints live in `scripts/` (`run_sft.py`, `run_dpo.py`, `run_eval.py`, `serve.py`) and will be run with `uv run`. These do not exist yet — create them per the spec.

## Code standards (from the design doc + user profile)

- One responsibility per file; small, single-purpose modules; type hints on every signature.
- Pydantic v2 for config and schemas (`ToolSpec`, `ToolCall`, `PreferencePair`).
- Specific exceptions only (`json.JSONDecodeError`, `pydantic.ValidationError`) — never bare `except`. Use `logging`, not `print`.
- Pure functions (`verifier.py`, `format.py`) get the most tests; also test split determinism and the reward function.

## Scope discipline (YAGNI)

Cut unless proven necessary: multiple base models (pick one, go deep), a hand-rolled training loop (`trl` + Unsloth is the right abstraction), a fancy agent UI (a clean README transcript is enough). An honest negative result (alignment doesn't beat SFT) is still a valid, reportable outcome — analyze *why*.
