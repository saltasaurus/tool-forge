# Course Log — Guided build of `tool-forge`

> **New session: read this file first, then [CLAUDE.md](CLAUDE.md) and the design doc
> ([LLM_Project-Function-Calling-Post-Training.md](LLM_Project-Function-Calling-Post-Training.md)),
> then resume at "Current state / next action." This is a guided course — do NOT scaffold or
> write project code on the user's behalf; guide them to write it.**

## Course settings (chosen by the user)

- **Theory depth: FULL RIGOR** — derive objectives (DPO/GRPO loss) from first principles, name
  every symbol, read the primary papers. The user reads formal math and targets senior ML/Research roles.
- **Pace: SMALL STEPS** — one concept or one file per lesson, with a checkpoint before moving on.
  Stop and wait at each checkpoint; do not run ahead.
- **Skills in play:** `ml-research-assistant` (post-training domain), `software-engineer-profile`
  (code standards), `pytorch-training` (when training code starts).

## The teaching contract

- **The user writes every line of project code.** Claude never hands over a finished `pyproject.toml`,
  verifier, or trainer. For each lesson: give the concept + *why it exists* + acceptance criteria,
  then review what the user wrote and iterate.
- **Two voices:** lecture (frame the concept, define terms on first use, write the objective in real
  math, name the failure mode) and Socratic (make the user predict/derive/decide before confirming).
- **Two rails never skipped:**
  1. **12 GB VRAM feasibility** stated for every training step (or explicitly flagged as not fitting,
     with the QLoRA/offload path named).
  2. **Sealed evaluation** — baseline measured *before* any change; the test split is never inspected
     until the end; BFCL is held out entirely.

## The arc (phase → capability the user will own)

| Phase | Deliverable | Capability |
|------|-------------|-----------|
| 0 | Scaffold + **baseline BFCL number** | Reproducible setup; measure-before-you-change discipline |
| 1 | Data pipeline | Chat templates, schema normalization, **sealed splits** |
| 2 | SFT checkpoint | QLoRA/PEFT, loss masking (train-on-completions), the TRL loop |
| 3 | Aligned checkpoint | **Preference alignment / RLHF** — DPO then GRPO (user's gap #1) |
| 4 | Results table | Public benchmark + custom metrics, honest reading of results |
| 5 | Served model + agent | vLLM serving, KV-cache, **ReAct loop** (user's gap #2) |

## Environment

- **Moving to WSL2 Ubuntu, native ext4** (`~/code/LLM_structured_response`), **not** `/mnt/c`
  (9p cross-OS filesystem is 10–50× slower for git/uv/imports/HF cache).
- Reason for WSL2: **Unsloth, bitsandbytes (4-bit QLoRA), Triton, vLLM are Linux-first**; native
  Windows support is fragile/absent.
- User confirmed present: `uv`, Python 3.12, GPU stack verified, W&B + HF accounts.
  **Pending:** re-verify the GPU *inside WSL* (the env that now matters).

## Current state / next action  ← UPDATE THIS EACH SESSION

**LESSON 0 nearly closed.** Socratic baseline checkpoint **PASSED**; Python version **decided**.
One small item still owed by the user before Lesson 0 is officially done:

- **Pin + GPU check (owed):** `uv python pin 3.12.11`, then confirm `nvidia-smi` sees the 4070
  *inside WSL*. (The `torch.cuda.is_available()` probe is deferred — `torch` doesn't exist until
  Lesson 1's `uv add`; don't ask for it yet.)

**Resolved this session:**
- **Python = 3.12.11** (uv-managed, hermetic). Rule taught: interpreter version is set by the
  most-fragile dependency tier (bitsandbytes/Unsloth/Triton/flash-attn), not language features.
  `requires-python = ">=3.12,<3.13"` to be encoded in `pyproject.toml` next lesson.
- **Baseline Socratic — passed.** User gave 2 of ≥3 distinct failures (comparability/regression;
  attribution). Supplied the two missed: **harness-validation** (baseline smoke-tests the eval
  pipeline — can't tell bad model from broken plumbing) and **per-category diagnosis**. The
  size/cost point the user offered was flagged valid-but-off-target.

**NEXT, once the pin + nvidia-smi land:** give the scaffolding spec for **Lesson 1** — `uv` project + `src/tool_forge/`
skeleton + `ruff`/`mypy --strict` green on an empty package + first commit (design doc tracked).
**Then Lesson 2:** build the load-bearing pure core — `schema.py` (Pydantic `ToolSpec`/`ToolCall`/
`PreferencePair`) and `align/verifier.py` (the verifier reused by pair-mining, the GRPO reward, *and*
eval) — with their tests, since they're pure, GPU-free, and unblock everything.

## Session log (newest last)

- **2026-06-19** — Created `CLAUDE.md` from the design doc. Established the course (full rigor, small
  steps). Decided to move the project to WSL2 ext4 for the Linux-first training stack. Issued Lesson 0;
  awaiting the Socratic answer + WSL diagnostics. Created this handoff log + a pointer in `CLAUDE.md`.
- **2026-06-19 (cont.)** — Lesson 0 Socratic session. Decided **Python 3.12.11** (uv-managed; rule:
  fragile-dependency tier sets the ceiling). Baseline checkpoint **passed** (user gave 2/≥3 failures;
  supplied harness-validation + per-category). `/reflect` written → vault project resolved to
  **ToolCaller** (folder ambiguity; user confirmed). Wrote devlog + decision (3.12.11) +
  ml-research-assistant learnings. Only `uv python pin 3.12.11` + WSL `nvidia-smi` still owed.
