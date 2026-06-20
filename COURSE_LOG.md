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

**LESSON 1 COMPLETE** (commit `da7584a`). Reproducible scaffold is in place:
- `uv` packaged project, **src-layout** (`src/tool_forge/`), interpreter pinned **3.12.11**.
- `requires-python = ">=3.12, <3.13"` (minor-level contract; lockfile is the exact pin).
- `ruff` (E,F,I,UP,B) + `mypy strict=true` (config-pinned, src-aware) — **both green** on the empty package.
- First commit tracks design doc + `CLAUDE.md` + `COURSE_LOG.md` + `pyproject.toml` + `uv.lock` +
  `.python-version`. No `__pycache__`/`.venv` leaked.

**Still owed from Lesson 0 (deferred, not blocking):** confirm `nvidia-smi` sees the 4070 *inside WSL*;
the `torch.cuda.is_available()` probe waits until `torch` is added (Phase 2/SFT). Don't ask yet.

**NEXT — LESSON 2: the load-bearing pure core.** Build `schema.py` (Pydantic v2 `ToolSpec` /
`ToolCall` / `PreferencePair`) and `align/verifier.py` (parse + validate a tool call against its
JSON Schema). These are **pure, GPU-free, and unblock everything** — the verifier powers pair-mining,
the GRPO reward, *and* eval, so it gets the most tests. Suggested small-step order: (a) design the
three schema types + *why each field exists*; (b) write `ToolSpec`/`ToolCall` with a test; (c) the
verifier as a pure function with its failure-mode taxonomy; (d) `PreferencePair`. Keep `verifier.py`
and `format.py` pure (no I/O, no global state); specific exceptions only.

## Session log (newest last)

- **2026-06-19** — Created `CLAUDE.md` from the design doc. Established the course (full rigor, small
  steps). Decided to move the project to WSL2 ext4 for the Linux-first training stack. Issued Lesson 0;
  awaiting the Socratic answer + WSL diagnostics. Created this handoff log + a pointer in `CLAUDE.md`.
- **2026-06-19 (cont.)** — Lesson 0 Socratic session. Decided **Python 3.12.11** (uv-managed; rule:
  fragile-dependency tier sets the ceiling). Baseline checkpoint **passed** (user gave 2/≥3 failures;
  supplied harness-validation + per-category). `/reflect` written → vault project resolved to
  **ToolCaller** (folder ambiguity; user confirmed). Wrote devlog + decision (3.12.11) +
  ml-research-assistant learnings. Only `uv python pin 3.12.11` + WSL `nvidia-smi` still owed.
- **2026-06-20** — **Lesson 1 complete** (commit `da7584a`). Built the uv src-layout scaffold; taught
  src-layout (import shadowing / source-vs-installed divergence), the two pin mechanisms
  (`requires-python` minor contract vs `.python-version`/`uv.lock` exact), strict-from-zero ratchet,
  and that `.gitignore` is not retroactive (caught a staged `.pyc` via read-before-commit). ruff+mypy
  green. Promoted the 4 baseline failure-modes to a vault concept note (`Evaluation-Baseline`).
  Next: Lesson 2 pure core (`schema.py` + `verifier.py`).
