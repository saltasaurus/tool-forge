"""Eval driver: generate tool calls on the dev split and score them.

Produces the base **floor** number (untouched Qwen3-4B-Base) and, with
`--adapter`, any SFT checkpoint against the same metrics — build once, run on
every checkpoint. Edge module (loads the training stack); not re-exported from
the package `__init__`.

    python -m tool_forge.eval --model base --limit 500              # floor
    python -m tool_forge.eval --model base --adapter out/sft-base   # an SFT checkpoint

Metrics form a lenient->strict ladder (means over rows). The content rungs read
calls wrapper-agnostically (a raw base model emits bare JSON, never the tags);
`protocol`/`strict` measure the invocation format on top of correct content. The
gap between `name_and_args` and `strict` is the SFT thesis: base can reason, SFT
teaches the protocol.
    emits_json    - produced >=1 well-formed call (bare JSON or wrapped)
    tool_name     - call names match gold (order-insensitive)
    name_and_args - names AND arguments match gold  (the reasoning floor)
    schema_valid  - every call validates against its tool's schema (via verify())
    protocol      - emitted the <tool_call> wrapper Qwen/BFCL expect
    strict        - wrapped AND correct: production/BFCL-usable  (the real accuracy)
    hallucinated  - called a tool not in the registry  (diagnostic)
"""

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict, ValidationError
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tool_forge import models
from tool_forge.dataset import load_dataset
from tool_forge.schema import ToolCall, ToolSpec, VerificationOutcome
from tool_forge.verify import verify

# Qwen3 wraps each call in literal <tool_call>{json}</tool_call> text (not a special token).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_JSON = json.JSONDecoder()


class RowScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    emits_json: bool
    tool_name: bool
    name_and_args: bool
    schema_valid: bool
    protocol: bool
    strict: bool
    hallucinated: bool


def row_registry(row: dict[str, Any]) -> dict[str, ToolSpec]:
    """dev-row `tools` -> the name->ToolSpec registry verify() expects."""
    return {t["function"]["name"]: ToolSpec(**t["function"]) for t in row["tools"]}


def gold_calls(row: dict[str, Any]) -> list[ToolCall]:
    """The assistant turn's gold tool calls as ToolCall objects."""
    calls = row["messages"][1]["tool_calls"]
    return [ToolCall(name=c["function"]["name"], arguments=c["function"]["arguments"]) for c in calls]


def _leading_json_objects(text: str) -> list[dict[str, Any]]:
    """Consecutive JSON objects from the first '{'; stop at the first non-JSON.

    ponytail: leading-run only. A base model emits its call(s) as bare JSON then
    degenerates into garbage (no EOS) -> grab the good prefix, stop at the junk.
    """
    i = text.find("{")
    if i < 0:
        return []
    out: list[dict[str, Any]] = []
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n or text[i] != "{":
            break
        try:
            obj, i = _JSON.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_calls(text: str) -> tuple[list[ToolCall], bool, bool]:
    """(calls, emits_json, wrapped).

    `wrapped` = the model emitted the <tool_call> protocol; when so, parse only
    those blocks, else fall back to leading bare JSON. `emits_json` = >=1 candidate
    and every candidate was a well-formed name+arguments call.
    """
    blocks = _TOOL_CALL_RE.findall(text)
    wrapped = bool(blocks)
    candidates: list[Any] = blocks if wrapped else _leading_json_objects(text)
    calls: list[ToolCall] = []
    ok = bool(candidates)
    for cand in candidates:
        try:
            obj = json.loads(cand) if isinstance(cand, str) else cand
            calls.append(ToolCall(name=obj["name"], arguments=obj["arguments"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            ok = False
    return calls, ok, wrapped


def score_row(
    pred: Sequence[ToolCall],
    gold: Sequence[ToolCall],
    registry: dict[str, ToolSpec],
    *,
    emits_json: bool,
    wrapped: bool,
) -> RowScore:
    verdicts = [verify(c, registry) for c in pred]
    content_exact = list(pred) == list(gold)  # ToolCall eq covers name + arguments
    return RowScore(
        emits_json=emits_json,
        tool_name=sorted(c.name for c in pred) == sorted(c.name for c in gold),
        name_and_args=content_exact,
        schema_valid=bool(pred) and all(v.result is VerificationOutcome.VALID for v in verdicts),
        protocol=wrapped,
        strict=wrapped and content_exact,
        hallucinated=any(v.result is VerificationOutcome.UNKNOWN_TOOL for v in verdicts),
    )


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str], max_new_tokens: int
) -> list[str]:
    # Left-pad so every prompt ends at the same right boundary -> new tokens start
    # at one uniform offset for the whole batch (set padding_side in evaluate()).
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(  # type: ignore[operator]
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        # Base's generation_config stops only on <|endoftext|>, never on the <|im_end|>
        # that ends a chat turn -> it runs to the cap every row. Stop on the turn-ender.
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _prompt(row: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> str:
    return cast(
        str,
        tokenizer.apply_chat_template(
            [row["messages"][0]], tools=row["tools"], add_generation_prompt=True, tokenize=False
        ),
    )


def _score_one(row: dict[str, Any], completion: str) -> RowScore:
    pred, emits_json, wrapped = parse_calls(completion)
    return score_row(pred, gold_calls(row), row_registry(row), emits_json=emits_json, wrapped=wrapped)


def _metrics(scores: Sequence[RowScore]) -> dict[str, float]:
    if not scores:
        return {}
    return {k: sum(getattr(s, k) for s in scores) / len(scores) for k in RowScore.model_fields}


def score_completions(data: Path, completions: Path) -> tuple[list[RowScore], dict[str, float]]:
    """Score a vLLM (or any offline) generation dump against the gold split.

    The dump is JSONL of {"i": row_index, "completion": text}; rows align by `i`,
    so a partial dump scores only what it covers. Engine-agnostic — same scorer
    the HF path uses, so metrics are directly comparable.
    """
    rows = load_dataset(data)
    by_index = {r["i"]: r["completion"] for r in load_dataset(completions)}
    scores = [_score_one(rows[i], by_index[i]) for i in sorted(by_index)]
    return scores, _metrics(scores)


def evaluate(
    model_arg: str, adapter: Path | None, data: Path, *, limit: int, max_new_tokens: int, batch_size: int
) -> tuple[list[RowScore], dict[str, float]]:
    spec = models.resolve(model_arg)
    tokenizer = models.load_tokenizer(spec)
    tokenizer.padding_side = "left"  # required for correct batched generation slicing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = models.load_model(spec)
    if adapter is not None:
        from peft import PeftModel

        model = cast(PreTrainedModel, PeftModel.from_pretrained(model, str(adapter)))
    model.eval()  # type: ignore[no-untyped-call]

    rows = load_dataset(data)
    if limit:
        rows = rows[:limit]

    scores: list[RowScore] = []
    bar = tqdm(range(0, len(rows), batch_size), unit="batch")
    for start in bar:
        batch = rows[start : start + batch_size]
        texts = generate_batch(model, tokenizer, [_prompt(r, tokenizer) for r in batch], max_new_tokens)
        scores.extend(_score_one(row, text) for row, text in zip(batch, texts, strict=True))
        n = len(scores)
        bar.set_postfix(
            content=f"{sum(s.name_and_args for s in scores) / n:.1%}",
            strict=f"{sum(s.strict for s in scores) / n:.1%}",
        )

    return scores, _metrics(scores)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="base", help="registry name (base/instruct) or a HF id / path")
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir to load onto the base (SFT checkpoint)")
    p.add_argument("--data", type=Path, default=Path("data/dev.jsonl"))
    p.add_argument("--limit", type=int, default=0, help="0 = all rows; >0 samples the first N")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16, help="prompts generated at once; raise to use GPU headroom")
    p.add_argument(
        "--completions",
        type=Path,
        default=None,
        help="score a generation dump (scripts/generate_vllm.py) instead of generating here",
    )
    p.add_argument("--out", type=Path, default=None, help="write metrics JSON here")
    args = p.parse_args()

    if args.completions is not None:
        _, metrics = score_completions(args.data, args.completions)
    else:
        _, metrics = evaluate(
            args.model,
            args.adapter,
            args.data,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    print("\n" + "\n".join(f"{k:>13}: {v:6.2%}" for k, v in metrics.items()))
    if args.out is not None:
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
