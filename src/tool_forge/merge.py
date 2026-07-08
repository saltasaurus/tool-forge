"""Merge a LoRA/QLoRA adapter into the base weights -> standalone bf16 model.

Serving path for adapters that touch lm_head/embed_tokens: vLLM's runtime LoRA does not
reliably apply head/embedding adapters, so the emitted <tool_call> wrapper would silently
vanish at serve time. Folding the adapter into the base (`merge_and_unload`) sidesteps
that — the merged model needs no runtime LoRA. Merging into the bf16 base (not the 4-bit
training base) also removes the QLoRA train/serve quantization mismatch.

    python -m tool_forge.merge --adapter runs/sft-base-v2/train --out runs/sft-base-v2/merged
"""

import argparse
from pathlib import Path
from typing import cast

from peft import PeftModel
from transformers import PreTrainedModel

from tool_forge import models


def merge(adapter: Path, out: Path, model_arg: str) -> None:
    """Fold `adapter` into the bf16 base and save a standalone model + tokenizer to `out`."""
    spec = models.resolve(model_arg)
    base = models.load_model(spec, quantized=False)  # bf16, not 4-bit: the merge target
    merged = cast(PreTrainedModel, PeftModel.from_pretrained(base, str(adapter)).merge_and_unload())
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    models.load_tokenizer(spec).save_pretrained(str(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir (SFT checkpoint)")
    p.add_argument("--out", type=Path, required=True, help="output dir for the merged bf16 model")
    p.add_argument("--model", default="base", help="registry name (base/instruct) or HF id / path")
    args = p.parse_args()
    merge(args.adapter, args.out, args.model)


if __name__ == "__main__":
    main()
