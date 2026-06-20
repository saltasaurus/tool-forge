# LLM_Project — Function-Calling Post-Training & Serving

**One-line pitch:** Take a small instruction-tuned LLM that *fails* at multi-step tool use, post-train it (SFT → preference alignment), evaluate it on the public Berkeley Function Calling Leaderboard (BFCL-V4), and serve it behind an OpenAI-compatible API with a minimal agent loop — all on a single 12 GB RTX 4070.

**Working title:** `tool-forge` (a small model fine-tuned into a reliable tool-caller)

---

## 1. Why this project (strategic rationale)

This project is chosen to close the two gaps you have flagged in your job search — **agentic frameworks** and **RLHF / post-training** — while staying distinct from your existing portfolio:

- **Outcrop** demonstrates retrieval + structured extraction on a *custom domain*. Its weakness as a portfolio piece is that the results are not externally verifiable.
- **Foraminifera LoRA** demonstrates *vision* transfer learning.
- **This project** demonstrates the *text post-training stack* (the thing Anthropic Post-Training, OpenAI, and agent teams hire for) and lands on a **public, reproducible benchmark**, so a reviewer can trust the number.

The measurable claim that makes it resume-worthy:

> "Fine-tuned a 4B model from a baseline BFCL-V4 overall accuracy of *X* to *Y*, closing most of the gap to models 3–8× its size, runnable on a single consumer GPU."

This is credible because recent evaluations confirm the baseline failure mode: small general-purpose models without explicit tool-call training emit malformed calls on anything past trivial single-step tasks (PromptQuorum, May 2026). You are fixing a *known, documented* deficiency and proving it with a standard metric.

---

## 2. Terminology (defined on first use)

- **Function calling (a.k.a. tool calling):** an LLM capability where, instead of replying in prose, the model emits a *structured* call to a named external function with typed arguments (e.g. `get_weather(city="Redmond", unit="celsius")`). The function signatures are supplied to the model as a **tool schema** (typically JSON Schema).
- **Chat template:** the model-specific string format that wraps a multi-turn conversation (system/user/assistant/tool roles) into the exact token sequence the model was trained on. Using the wrong template silently degrades quality.
- **SFT (Supervised Fine-Tuning):** training the model on `(prompt, ideal_response)` pairs via standard next-token cross-entropy loss. Teaches the *format and behavior*.
- **LoRA (Low-Rank Adaptation):** a parameter-efficient fine-tuning method that freezes the base weights and trains small low-rank adapter matrices added to selected layers. Drastically reduces trainable parameters and memory.
- **QLoRA (Quantized LoRA):** LoRA applied on top of a base model quantized to 4-bit (NF4). Roughly 4× less VRAM than 16-bit LoRA, at a marginal accuracy cost.
- **DPO (Direct Preference Optimization):** an *offline* preference-alignment method. Given triples `(prompt, chosen, rejected)`, it directly optimizes the policy to prefer `chosen` over `rejected` without training a separate reward model. Loss (β = strength/temperature, σ = sigmoid, π_θ = policy, π_ref = frozen reference):
  ```
  L_DPO = −E[ log σ( β·( log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x)) ) ) ]
  ```
  where `y_w` is the chosen (winning) response and `y_l` the rejected (losing) one.
- **Reference model (π_ref):** a frozen copy of the model used to anchor DPO/GRPO so the policy does not drift arbitrarily. With LoRA, the reference is obtained for free by *disabling the adapters* — no second model in VRAM.
- **GRPO (Group Relative Policy Optimization):** an *online* RL method (the lineage used in modern RLHF/reasoning training). For each prompt it samples a *group* of G responses, scores each with a **reward function**, and computes a **group-relative advantage** `A_i = (r_i − mean(r)) / std(r)`, then updates the policy with a PPO-style clipped objective plus a KL penalty toward π_ref. Tool-call correctness is *programmatically verifiable*, which gives a clean automatic reward with no human labels or reward model — an ideal fit for GRPO.
- **AST evaluation:** BFCL parses the model's emitted call into an **Abstract Syntax Tree** (a structural representation of code/call syntax) and compares it to the gold call structurally, rather than by brittle string match. This is how function name + argument correctness is scored.
- **vLLM:** a high-throughput inference/serving engine featuring **PagedAttention** (efficient **KV-cache** memory management — the cache of past keys/values that avoids recomputing attention each step) and continuous batching. Exposes an **OpenAI-compatible API** so any OpenAI client works unchanged.
- **ReAct:** an agent pattern that interleaves reasoning ("thought") and tool calls ("action") in a loop until the task is solved.

---

## 3. Skills demonstrated (mapped to mid-level LLM-engineer competencies)

| Competency | Where it shows up |
|---|---|
| Data engineering for LLMs | Phase 1: schema normalization, chat-template formatting, dedup, train/dev/test sealing |
| Supervised fine-tuning (PEFT/QLoRA) | Phase 2 |
| Preference alignment (DPO) and/or online RL (GRPO) | Phase 3 |
| Verifiable reward design | Phase 3 (Path B) — programmatic AST/schema verifier as reward |
| Rigorous evaluation | Phase 4: public benchmark + custom format-compliance metrics, sealed splits |
| Inference optimization & serving | Phase 5: vLLM, quantization, KV-cache, throughput/latency measurement |
| Agentic systems | Phase 5: ReAct-style multi-step tool loop |
| MLOps / reproducibility | Phase 0: `uv`, `ruff`, `mypy --strict`, W&B, CI, frozen seeds |

---

## 4. Hardware feasibility (the 12 GB constraint)

Honest assessment, with measurement flagged where I cannot assert exact VRAM:

- **SFT (QLoRA, 4-bit) on a 3–4B model:** comfortably fits. Community runs do 4-bit QLoRA on 2B models at 4096 context on 12 GB cards; a 3–4B QLoRA at 2048–4096 context with `optim="adamw_8bit"` and gradient checkpointing is in range.
- **DPO (LoRA, reference via adapter-disabling):** feasible but **the tightest training step** — DPO processes chosen *and* rejected per step (≈2× activation memory). Mitigations: shorter `max_length`, `per_device_batch_size=1` with gradient accumulation, smaller base model.
- **GRPO (Path B):** also tight — it generates G samples per prompt during training. Feasible on a small (1.5–2B) model with short generations; treat as a stretch.
- **Serving (vLLM):** a 4B model (optionally AWQ/GPTQ 4-bit quantized) serves within 12 GB, leaving room for KV cache.

> **Action item before committing:** run a 30-step smoke test at your target `(model, context_length, batch_size)` and read peak `nvidia-smi` memory. If DPO OOMs at 4B, drop the base to ~1.5–2B for the alignment phase — the *method* is what the résumé showcases, not the parameter count.

**Recommended base model:** `Qwen3-4B` (Apache-2.0, strong tool-call priors, excellent ecosystem). **Conservative fallback:** `Qwen2.5-3B-Instruct`. Both are well-supported by Unsloth and vLLM.

---

## 5. Tech stack

- **Language/env:** Python 3.12, `uv`, `ruff`, `mypy --strict`, Pydantic v2 (consistent with your standing toolchain)
- **Training:** PyTorch + Hugging Face `transformers`, `trl` (SFTTrainer / DPOTrainer / GRPOTrainer), `peft`, `bitsandbytes`; **Unsloth** as the memory-efficient backend (advertised ~2× faster, ~70% less VRAM; supports DPO/GRPO/KTO and exports to GGUF/vLLM/HF)
- **Data:** an open function-calling SFT corpus — **Salesforce `xLAM`** (HF cookbook exists for QLoRA fine-tuning on it) and/or **ToolACE** (BFCL-strong); **Glaive function-calling** as a backup source
- **Eval:** `bfcl-eval` (`pip install bfcl-eval`) for BFCL-V4 (AST + executable scoring across simple / multiple / parallel / multi-turn / agentic categories) + a small custom format-compliance harness
- **Serving:** `vllm` (OpenAI-compatible server), optional `autoawq`/GPTQ for 4-bit serving
- **Tracking/CI:** Weights & Biases (free tier), GitHub Actions

---

## 6. Repository layout

Follows your Python file-layout convention (one responsibility per file). Each module is small and single-purpose.

```
tool-forge/
  pyproject.toml            # uv + ruff + mypy config, pinned deps
  README.md                 # the story + final results table (the resume artifact)
  .github/workflows/ci.yml  # ruff, mypy --strict, unit tests on push

  src/tool_forge/
    config.py               # Pydantic settings: model id, paths, hyperparameters, seeds
    schema.py               # Pydantic models: ToolSpec, ToolCall, PreferencePair
    data/
      load.py               # pull + cache raw function-calling datasets
      format.py             # apply chat template, render tool schemas -> training rows
      split.py              # frozen, sealed train/dev/test split (hash-based, seeded)
    train/
      sft.py                # SFT (QLoRA) loop only
      dpo.py                # DPO loop only
      grpo.py               # GRPO loop only (Path B / stretch)
    align/
      verifier.py           # parse + validate a tool call against its schema (PURE)
      pairs.py              # build near-miss preference pairs from verifier failures
      reward.py             # map verifier result -> scalar reward (for GRPO)
    eval/
      bfcl_runner.py        # thin wrapper that runs bfcl-eval on a served endpoint
      format_metrics.py     # custom: JSON-validity %, schema-compliance %, hallucinated-tool %
    serve/
      server.py             # launch vLLM OpenAI-compatible server with the merged adapter
    agent/
      loop.py               # minimal ReAct-style multi-step tool loop
      tools.py              # 3–5 toy tools (calculator, weather stub, web fetch stub, ...)

  tests/                    # pytest: verifier, formatter, split determinism, reward fn
  scripts/                  # entrypoints: run_sft.py, run_dpo.py, run_eval.py, serve.py
```

**Design notes (per your standards):** `verifier.py` and `format.py` are **pure functions** (no I/O, no global state) so they are trivially unit-testable; the verifier is the load-bearing component (it powers preference-pair mining, the GRPO reward, *and* the custom eval), so it gets the most tests. Use specific exceptions (e.g. `json.JSONDecodeError`, `pydantic.ValidationError`), never bare `except`. Use `logging`, not `print`. Type hints on every signature.

---

## 7. Phase plan (end-to-end)

### Phase 0 — Scaffolding & baseline (day 0–1)
1. `uv` project, `ruff` + `mypy --strict`, CI on push, W&B project created.
2. **Establish the baseline number first.** Serve the *untouched* base model with vLLM and run BFCL-V4. Record per-category accuracy. This is your "before" — do not skip it; it is half the story.

**Deliverable:** baseline results table in W&B + README.

### Phase 1 — Data pipeline (day 1–3)
1. Load xLAM (and/or ToolACE). Normalize every row to a single internal schema (`schema.py`): system prompt, available `ToolSpec`s, conversation, gold `ToolCall`(s).
2. Render to the base model's **chat template** with tool schemas in the system/tool section. Validate every gold call parses and matches its schema (reuse `verifier.py`).
3. **Seal the splits:** deterministic, hash-based train/dev/test split (`split.py`); the test split is *never* looked at until the end. Keep BFCL fully held out — it is the external test set.

**Deliverable:** a versioned dataset artifact in W&B; unit test proving split determinism.

### Phase 2 — Supervised fine-tuning (day 3–5)
1. QLoRA SFT with Unsloth on the formatted data. LoRA on attention + MLP projections; start `r=16, alpha=16/32, dropout=0`; `adamw_8bit`; 1–3 epochs; train-on-completions (mask the prompt tokens from the loss).
2. Log loss curves, sample generations, and dev-set format-compliance to W&B.
3. Re-run BFCL-V4 on the SFT model.

**Deliverable:** SFT checkpoint + BFCL delta vs baseline (expect the biggest jump here).

### Phase 3 — Preference alignment (day 5–9)

The goal here is the *last mile*: SFT fixes gross formatting; alignment fixes the subtle near-misses (wrong argument, missing required field, slightly-off function choice). **Pick one path; both are résumé-valid.**

**Path A — DPO (recommended starting point, lower risk):**
1. **Mine near-miss preference pairs** (`pairs.py`): sample N completions from the SFT model on dev prompts; run each through `verifier.py`. Use a *correct* call as `chosen` and a *plausible-but-wrong* call (malformed JSON, hallucinated tool, wrong/missing arg) as `rejected`. Optionally augment with programmatic corruptions of gold calls.
2. DPO with `trl`'s `DPOTrainer` over LoRA; reference model via adapter-disabling (no second model in VRAM); `β≈0.1`; short `max_length`.
3. Re-run BFCL-V4.

**Path B — GRPO (higher ceiling, directly demonstrates RLHF; stretch on 12 GB):**
1. Reward = `reward.py` mapping the verifier result to a scalar (e.g. +1 valid-and-correct, partial credit for valid-but-wrong-arg, −1 unparseable/hallucinated tool). This is a **verifiable reward** — no human labels, no reward model.
2. GRPO with `trl`'s `GRPOTrainer` on a smaller base (1.5–2B) with short generations to fit VRAM.
3. Re-run BFCL-V4.

**Deliverable:** aligned checkpoint + BFCL delta vs SFT; a short write-up of *which failure categories* improved (the analysis is what signals seniority).

### Phase 4 — Evaluation (day 9–11)
1. **Public:** BFCL-V4 via `bfcl-eval`, full per-category breakdown (simple / multiple / parallel / multi-turn / agentic).
2. **Custom (`format_metrics.py`):** JSON-validity %, schema-compliance %, hallucinated-tool rate, exact-call accuracy on your sealed test split — these isolate *why* the model improved, which BFCL alone won't tell you.
3. Build a clean comparison table: **base → SFT → aligned**, plus 1–2 published reference points for context.

**Deliverable:** the results table that anchors the README and résumé bullet.

### Phase 5 — Serving + agent harness (day 11–14)
1. Merge the adapter, optionally quantize (AWQ/GPTQ 4-bit), serve with vLLM (OpenAI-compatible endpoint). **Measure tokens/sec and p50/p95 latency** at a couple of batch sizes — serving metrics are a mid-level differentiator most candidates skip.
2. `agent/loop.py`: a minimal ReAct loop with 3–5 real toy tools, showing the *aligned* model completing a multi-step task end-to-end against the served endpoint.

**Deliverable:** `serve.py` + a recorded multi-step agent transcript in the README.

---

## 8. Evaluation metrics (definitions)

- **BFCL Overall Accuracy:** unweighted mean across BFCL sub-categories (AST + executable scoring). The headline external number.
- **JSON-validity %:** fraction of emitted calls that parse as valid JSON.
- **Schema-compliance %:** of the valid-JSON calls, fraction matching the tool's JSON Schema (required fields present, types correct).
- **Hallucinated-tool rate:** fraction of calls naming a function not in the provided toolset (should trend to ~0 after alignment).
- **Exact-call accuracy:** strict match of function name + all arguments on the sealed test split.
- **Serving:** throughput (tokens/sec) and latency (p50/p95) under vLLM.

---

## 9. Résumé bullets (fill in your numbers)

Paste-ready, quantified, action-first:

- *Post-trained a 4B LLM into a reliable tool-caller (SFT → DPO with self-generated near-miss preference pairs), raising Berkeley Function Calling Leaderboard (BFCL-V4) overall accuracy from **X%** to **Y%** on a single 12 GB consumer GPU.*
- *Designed a programmatic schema/AST verifier that powered preference-pair mining, a verifiable RL reward (GRPO), and a custom eval suite (JSON-validity, schema-compliance, hallucinated-tool rate).*
- *Served the model via vLLM behind an OpenAI-compatible API at **Z** tokens/sec (p95 **W** ms) and built a ReAct agent loop demonstrating reliable multi-step tool use.*
- *Built the full pipeline with sealed train/dev/test splits, W&B tracking, and CI (ruff, mypy --strict) for reproducibility.*

---

## 10. Scope discipline (YAGNI) & risks

**Cut these unless proven necessary:**
- Multiple base models — pick one, go deep.
- A custom training loop — `trl` + Unsloth is the correct abstraction; hand-rolling it adds risk, not signal.
- A fancy agent UI — a clean transcript in the README is enough.

**Top risks & mitigations:**
1. *DPO/GRPO OOM at 4B* → drop the alignment base to 1.5–2B; shorten context; batch=1 + grad-accum.
2. *Alignment doesn't beat SFT* → that is still a valid, reportable result; analyze *why* (likely SFT already saturated the easy categories). Honest negative results read as senior.
3. *BFCL setup friction* → pin `bfcl-eval==<exact version>` and serve via the OpenAI-compatible endpoint so the harness treats your model like any API model.
4. *Data licensing* → confirm dataset licenses (xLAM / ToolACE / Glaive) permit your use before publishing.

---

## 11. Stretch goals (only after the core ships)
- Add **constrained decoding** (grammar/JSON-Schema-guided generation) at serve time and quantify the format-compliance gain — ties directly to your prior study of logit processors and grammar-constrained decoding.
- Compare **DPO vs GRPO** head-to-head on the same base (a genuinely interesting, publishable-quality ablation).
- Add **speculative decoding** in vLLM and report the latency improvement.

---

## 12. References (current as of June 2026)
- Berkeley Function Calling Leaderboard V4 — gorilla.cs.berkeley.edu/leaderboard.html ; `pip install bfcl-eval`
- Salesforce xLAM function-calling fine-tuning cookbook — huggingface.co/learn/cookbook/en/function_calling_fine_tuning_llms_on_xlam
- Unsloth (QLoRA/DPO/GRPO, low-VRAM fine-tuning) — unsloth.ai/docs
- TRL (SFTTrainer / DPOTrainer / GRPOTrainer) — Hugging Face
- DPO: Rafailov et al., "Direct Preference Optimization" (2023)
- Tool-calling reliability of small models — promptquorum.com (May 2026), as motivation for the baseline failure mode

---

### Suggested filename for the repo's design doc
`docs/DESIGN.md` (this file). Mirror Section 6's layout in the repo and keep the final results table in both `README.md` and your Obsidian project note.
