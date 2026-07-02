"""SFT driver: QLoRA fine-tune a base model on the tool-call dataset.

python -m tool_forge.train --model base --out out/sft-base
"""

import argparse
from pathlib import Path
from typing import cast

import datasets
from peft import LoraConfig, PeftModel, get_peft_model
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from tool_forge import models
from tool_forge.dataset import load_dataset, render_prompt_completion

LORA_CONFIG = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    target_modules="all-linear",
    lora_alpha=32,
    lora_dropout=0.05,
)


def train(
    model_arg: str,
    out: Path,
    data: Path,
    *,
    epochs: float,
    max_steps: int,
    limit: int,
    report_to: str,
) -> None:
    spec = models.resolve(model_arg)
    tokenizer = models.load_tokenizer(spec)

    rows = load_dataset(data)
    if limit:
        rows = rows[:limit]
    # Pre-render to two text columns: Arrow stores strings trivially, dodging
    # schema inference on the heterogeneous nested `tools`.
    dataset = datasets.Dataset.from_list([render_prompt_completion(r, tokenizer) for r in rows])

    # load_model(for_training) already runs prepare_model_for_kbit_training; wrap LoRA here.
    # cast: single LoraConfig -> PeftModel, never the PeftMixedModel arm of the union.
    model = cast(PeftModel, get_peft_model(models.load_model(spec, for_training=True), LORA_CONFIG))

    config = SFTConfig(
        str(out),
        max_length=1024,  # length probe: >1024 truncates ~3%
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        max_steps=max_steps,  # -1 = run full epochs
        logging_steps=10,
        report_to=report_to,
        run_name=out.name,
    )

    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(out))  # adapter only


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="base", help="registry name (base/instruct) or a HF id / path")
    p.add_argument("--out", type=Path, required=True, help="adapter output dir")
    p.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = full epochs; >0 caps steps (smoke runs)")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows; >0 truncates the dataset")
    p.add_argument("--wandb", action="store_true", help="report to W&B (default: no reporting)")
    args = p.parse_args()

    train(
        args.model,
        args.out,
        args.data,
        epochs=args.epochs,
        max_steps=args.max_steps,
        limit=args.limit,
        report_to="wandb" if args.wandb else "none",
    )


if __name__ == "__main__":
    main()
