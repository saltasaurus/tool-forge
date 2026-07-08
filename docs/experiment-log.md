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

| Metric | Base floor | v3-ckpt400 |
|---|---|---|
| `protocol` | 0.00% | 73.41% |
| `strict` | 0.00% | 47.61% |
| `name_and_args` | 67.08% | 47.61% |
| `tool_name` | 87.31% | 61.03% |
| `emits_json` | 97.23% | 73.09% |
| `schema_valid` | 93.80% | 69.21% |
| `hallucinated` | 0.10% | 3.38% |

`strict` off 0% is the thesis working. The content metrics regressed vs base — the
investigation of *why* is the next section.

---

## Dead-end — the content drop was not truncation

**Hypothesis:** multi-call rows emit several verbose `<tool_call>` blocks, exceed the
256-token generation cap, get cut off, and the scorer (which needs a closing
`</tool_call>`) drops the whole row — tanking every content metric at once.

**Test:** raised `--max-new-tokens` 256 → 512 and regenerated.

**Result:** truncation rate unchanged at **33.9%**; metrics essentially identical.
Raising the cap made the "unclosed" completions *longer*, not fewer — the opposite of
what truncation predicts.

**Actual cause:** the unclosed completions are **degenerate repetition loops** (median
4260 chars vs 150 for good ones; endings like `_sp_sp_sp…`, `444444…`; `<tool_call>`
opened up to 25×). On ~34% of prompts the model fails to produce one call and stop, and
rambles to the cap. 0% are pure junk — every output is a real attempt — so this is an
**undertraining** tail on a 0.14-epoch checkpoint, not a decoding-length problem.

> **Learning 1: it wasn't max-length.** The 512 experiment disproved it cleanly.
> `--max-new-tokens` was kept at 512 anyway (legit 3–5-call rows need it), but it was
> not the fix.
>
> **Learning 2: teacher-forced eval cannot see degeneration.** `eval_loss` read 0.02
> and `eval_mean_token_accuracy` 0.99 while a third of *free* generations were garbage —
> because teacher-forced scoring always feeds the gold prefix and never lets the model
> spiral. Catching this during training needs a free-generation probe (e.g. logging the
> fraction of sampled generations that hit the token cap without emitting EOS).

---

## Open threads

- Resume v3 from checkpoint-400 and train further; expect the degeneration tail to
  shrink and `strict` to climb.
- Add a generation-probe metric (`gen/hit_cap_rate`) to the training loop so
  degeneration is visible live rather than only at full-eval time.
- Run BFCL on the SFT model for an external, comparable number (Instruct-2507 anchor
  is 30.23% overall).
