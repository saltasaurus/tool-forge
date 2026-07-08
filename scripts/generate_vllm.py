"""Phase A of the fast eval: generate tool-call completions with vLLM.

Runs in `.venv-serve` (the isolated vLLM env) and imports NO `tool_forge` — its
heavy deps would collide with vLLM's pinned stack. Reads the dev split, renders
the SAME chat-template prompt `tool_forge.eval` uses, and dumps one
{"i", "completion"} per line. Phase B (`python -m tool_forge.eval --completions`)
scores the dump in `.venv` with the shared, pure scorer.

vLLM's continuous batching + per-sequence `stop` is the speedup: finished rows
free their slot immediately instead of waiting on the batch's slowest straggler.

    ./scripts/generate_vllm.sh --model base --out runs/base/eval/dev.gen.jsonl
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Mirror tool_forge.const / models: base weights are decoded with the Instruct
# tokenizer (the base ships no chat template). Kept literal to avoid importing tool_forge.
WEIGHTS = {"base": "Qwen/Qwen3-4B-Base", "instruct": "Qwen/Qwen3-4B-Instruct-2507"}
TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def render_prompts(data: Path, tokenizer_id: str) -> list[str]:
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    prompts: list[str] = []
    with open(data) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(
                tok.apply_chat_template(
                    [row["messages"][0]], tools=row["tools"], add_generation_prompt=True, tokenize=False
                )
            )
    return prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="base", help="base/instruct, or a HF id / local path")
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir (SFT checkpoint)")
    p.add_argument("--data", type=Path, default=Path("data/dev.jsonl"))
    p.add_argument("--out", type=Path, required=True, help="JSONL dump of {i, completion}")
    p.add_argument("--max-new-tokens", type=int, default=512)  # wrapped multi-call rows exceed 256 and truncate
    p.add_argument("--gpu-util", type=float, default=0.90)  # matches serve_baseline.sh
    p.add_argument("--max-model-len", type=int, default=4096)  # dev prompts are short; leaves KV for batching
    args = p.parse_args()

    weights = WEIGHTS.get(args.model, args.model)
    prompts = render_prompts(args.data, TOKENIZER)

    llm = LLM(
        model=weights,
        tokenizer=TOKENIZER,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # WSL: no FlashInfer JIT (no nvcc)
        enable_lora=args.adapter is not None,
        max_lora_rank=16,
    )
    # Stop on the chat turn-ender so ragged base output doesn't run to the cap.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop=["<|im_end|>"])
    lora = LoRARequest("sft", 1, str(args.adapter)) if args.adapter is not None else None

    outputs = llm.generate(prompts, sampling, lora_request=lora)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, out in enumerate(outputs):
            f.write(json.dumps({"i": i, "completion": out.outputs[0].text}) + "\n")
    print(f"wrote {len(outputs)} completions -> {args.out}")


if __name__ == "__main__":
    main()
