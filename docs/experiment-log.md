# Experiment log

A running lab notebook of what was tried, what worked, and — deliberately — what
didn't. The README appendix carries the polished per-checkpoint results; this file
carries the reasoning and the dead-ends.

Base model: `Qwen/Qwen3-4B-Base`. Reference ceiling: `Qwen/Qwen3-4B-Instruct-2507`.
Method: QLoRA SFT on xLAM tool-calling data, single 12 GB RTX 4070.

Metrics (custom protocol, loosest → strictest): `emits_json`, `schema_valid`,
`tool_name`, `name_and_args`, `protocol` (emits the `<tool_call>` wrapper),
`strict` (wrapped **and** correct). Base floor and Instruct ceiling are scored with
the identical harness so deltas are attributable to the fine-tune.

---

## v1 — first SFT run (discarded)

**Config:** LoRA on `all-linear`, LR 2e-4, no warmup, grad-accum 4.

**Observed:** training loss noisy and stuck ~0.65–0.78; generations degenerated into
rare-token spew (`取`, `NdrFc`, leading `�`). Stopped early at step 3500.

**Eval:** `protocol` 0%, `strict` 0%; content metrics *below* the untuned base.

---

## v2 — stabilized schedule (discarded)

**Change:** LR 2e-4 → 1e-4, `warmup_ratio` 0.03, cosine schedule, grad-accum 4 → 16,
added a held-out `eval_loss` curve.

**Observed:** loss curve became smooth and monotonic — the instability was gone.

**Eval:** metrics **byte-for-byte identical to v1** (`protocol` 0%, `strict` 0%,
`name_and_args` ~42%).

> **Learning: stability was not the bug.** A cleaner optimization trajectory changed
> nothing about the outcome. Two different schedules landing on the identical eval is
> the tell that the ceiling is structural, not optimization.

---

## Diagnosis — the frozen tied head

The model emitted its calls as bare JSON with a garbage byte (`�`) where the
`<tool_call>` token should be. Root cause, confirmed directly:

- Qwen3-4B-**Base** is pretraining-only and never learned the tool-call special
  tokens. Their rows in the embedding matrix sit at initialization: `<tool_call>`
  (id 151657) norm **0.3566 vs a 1.16 median** — identical across all tool tokens,
  i.e. untouched.
- `tie_word_embeddings=True` — the input embedding and output head are the *same*
  matrix, so that untrained row is also the output projection used to emit the token.
- `target_modules="all-linear"` **excludes** `embed_tokens`/`lm_head`. The LoRA
  reshaped hidden states but could not raise the logit for a token whose output row
  it was not allowed to touch.

This also explains the ~0.65 loss floor in v1/v2: the model paid full cross-entropy on
the `<tool_call>` token that opened *every* target and could not produce it.

---

## v3 — train the tied head (current)

**Change:** add `lm_head` and `embed_tokens` to the LoRA `target_modules`. Forced
companion: `loss_type="nll"` (TRL's default `chunked_nll` reads `lm_head` directly and
errors once it is PEFT-wrapped).

**Smoke run (200 steps):** train loss dropped from the stuck ~0.65 to **0.10**;
`protocol` 0% → **22.5%**, `strict` 0% → **10%**. Confirmed the mechanism before
committing to a full run.

**Serving note:** vLLM does not reliably apply LoRA to `lm_head`/`embed_tokens`, so a
served adapter would silently drop the fix. The adapter is therefore **merged**
(`tool_forge.merge`) into a standalone bf16 model before eval. Merging also removes the
QLoRA 4-bit-train / bf16-serve mismatch.

### v3 — checkpoint 400 (0.14 epoch)

Faithful eval — the adapter applied un-merged to the 4-bit base it was trained against
(HF generate, 300-row dev subset). This checkpoint beats the base floor on every metric
**and** adds the protocol the base cannot produce:

| Metric | Base floor | v3-ckpt400 (faithful) | v3-ckpt400 (merged — artifact) |
|---|---|---|---|
| `protocol` | 0.00% | 100.00% | 73.41% |
| `strict` | 0.00% | 83.67% | 47.61% |
| `name_and_args` | 67.08% | 83.67% | 47.61% |
| `tool_name` | 87.31% | 99.67% | 61.03% |
| `emits_json` | 97.23% | 100.00% | 73.09% |
| `schema_valid` | 93.80% | 100.00% | 69.21% |
| `hallucinated` | 0.10% | 0.00% | 3.38% |

The gap between the two columns is a serving bug, investigated next.

---

## Dead-end 1 — the content drop was not truncation

The *merged* eval showed ~34% of outputs failing to close the `<tool_call>` wrapper.

**Hypothesis:** multi-call rows exceed the 256-token cap and get cut off; the scorer
(which needs a closing `</tool_call>`) then drops the whole row.

**Test:** raised `--max-new-tokens` 256 → 512 and regenerated.

**Result:** truncation rate unchanged at **33.9%**; raising the cap made the unclosed
completions *longer*, not fewer — the opposite of truncation. The unclosed rows were
**degenerate repetition loops** (median 4260 chars vs 150; endings like `_sp_sp…`,
`444…`; `<tool_call>` opened up to 25×), not cut-off calls.

> **Learning:** it wasn't max-length. `--max-new-tokens` was kept at 512 (legit 3–5-call
> rows need it), but it was not the fix. Also: teacher-forced eval (`eval_loss` 0.02,
> token-acc 0.99) is blind to this — it never lets the model free-generate and spiral.

## Dead-end 2 — the degeneration was the merge, not the model

The "34% degeneration" was first read as an **undertraining** tail. That was wrong.

**Test:** a free-generation probe on the un-merged 4-bit path read `hit_cap_rate` **0%**.
Suspicious, so the exact rows that degenerated in the *merged* eval were regenerated
three ways:

| inference path | those rows |
|---|---|
| 4-bit base + adapter, **un-merged** (= training regime) | 0% degenerate, 100% closed |
| merged into **bf16** base (`merge.py` as written) | degenerate |
| merged into **4-bit** base (re-quantized) | 100% degenerate |

**Cause:** a QLoRA adapter is trained against the dequantized 4-bit base. Folding it into
a fresh bf16 base (`quantized=False`) gives it the wrong base weights — the quantization
gap tips ~34% of inputs into repetition. Re-quantizing on merge is worse. A likely
compounding factor: `tie_word_embeddings=True` with a tied layer in the adapter — PEFT
warns this mishandles merging (`ensure_weight_tying`, issue #2777).

> **Learning: a QLoRA adapter must be served on the base it was trained against.** The
> faithful path (un-merged 4-bit + adapter) has no degeneration and scores `strict` 83.7%.
> The model was never the problem; the serving/merge path was. A metric that looks bad
> only after a serving transform is a serving bug until proven otherwise.

---

## Open threads

- **Serving:** apply the adapter faithfully *and* fast. vLLM won't apply the head LoRA
  and merging degrades — leads: PEFT `ensure_weight_tying`, dequantize-then-merge to
  bf16, or plain (non-quantized) LoRA so merges are lossless.
- Lock the final numbers with a full-dev faithful eval (the 300-row subset above is a
  fast read).
- Run BFCL on the SFT model for an external, comparable number (Instruct-2507 anchor
  is 30.23% overall).
- The `gen/hit_cap_rate` probe is wired in but monitors the (clean) training regime; its
  value is catching a *future* training-path regression, not this serving issue.
