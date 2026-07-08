"""SFT driver: QLoRA fine-tune a base model on the tool-call dataset.

python -m tool_forge.train --model base --out runs/sft-base/train
"""

import argparse
import os
from pathlib import Path
from typing import cast

import datasets
from peft import LoraConfig, PeftModel, get_peft_model
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from tool_forge import models
from tool_forge.dataset import load_dataset, render_prompt_completion

# target_modules is all-linear's attention/MLP projections PLUS the tied embed/head.
# Qwen3-4B-Base is pretraining-only, so the tool-call special tokens sit at init in the
# tied embed/lm_head matrix (<tool_call> row norm 0.31x median) — and all-linear excludes
# embed_tokens/lm_head. Without these two the model cannot emit the <tool_call> protocol
# wrapper no matter how long it trains. lm_head -> emit the token; embed_tokens -> a usable
# input representation of the token once emitted (so the JSON after it doesn't degrade).
LORA_CONFIG = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head", "embed_tokens",
    ],
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
    eval_limit: int,
    resume: bool,
    report_to: str,
) -> None:
    spec = models.resolve(model_arg)
    tokenizer = models.load_tokenizer(spec)

    # Label the run by its folder (runs/<run>/train -> "<run>"), not the literal "train".
    run_label = out.parent.name if out.name == "train" else out.name
    if report_to == "wandb":
        # One W&B run per output dir, stitched across resumes. The id lives in
        # <out>/wandb_run_id.txt: --resume reads it back, a fresh run mints run_label
        # and records it. Either way the id is stable, so resumes append to the same
        # curve with zero env vars. (setdefault still lets an explicit env override win.)
        id_file = out / "wandb_run_id.txt"
        run_id = id_file.read_text().strip() if resume and id_file.exists() else run_label
        # Group runs under this repo's project; without it TRL defaults to "huggingface".
        os.environ.setdefault("WANDB_PROJECT", "tool-forge")
        os.environ.setdefault("WANDB_RUN_ID", run_id)
        os.environ.setdefault("WANDB_RESUME", "allow")
        out.mkdir(parents=True, exist_ok=True)
        id_file.write_text(os.environ["WANDB_RUN_ID"])

    rows = load_dataset(data)
    if limit:
        rows = rows[:limit]
    # Pre-render to two text columns: Arrow stores strings trivially, dodging
    # schema inference on the heterogeneous nested `tools`.
    dataset = datasets.Dataset.from_list([render_prompt_completion(r, tokenizer) for r in rows])
    # Held-out eval_loss curve so early-stop is read off dev, not the noisy train loss.
    # A subset keeps each eval cheap; dev.jsonl sits beside the train split.
    dev_rows = load_dataset(data.parent / "dev.jsonl")[:eval_limit]
    eval_dataset = datasets.Dataset.from_list([render_prompt_completion(r, tokenizer) for r in dev_rows])

    # load_model(for_training) already runs prepare_model_for_kbit_training; wrap LoRA here.
    # cast: single LoraConfig -> PeftModel, never the PeftMixedModel arm of the union.
    model = cast(PeftModel, get_peft_model(models.load_model(spec, for_training=True), LORA_CONFIG))

    config = SFTConfig(
        str(out),
        max_length=1024,  # length probe: >1024 truncates ~3%
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # eff. batch 16: smoother gradient than the old 4
        gradient_checkpointing=True,
        bf16=True,
        # Standard cross-entropy. TRL's default chunked_nll reads lm_head weights directly
        # and errors once lm_head is PEFT-wrapped (which we now do to learn the tool tokens).
        loss_type="nll",
        learning_rate=1e-4,  # was 2e-4; the old run spiked into token-degeneracy
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,  # was 0: peak LR on step 1 destabilized the QLoRA run
        num_train_epochs=epochs,
        max_steps=max_steps,  # -1 = run full epochs
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        # nll loss materializes the full 151,936-vocab logits, so eval memory scales
        # hard with batch. 2 keeps headroom for the load_best_model_at_end reload that
        # OOM'd at finalization with an lm_head/embed-inclusive (large) adapter resident.
        per_device_eval_batch_size=2,
        # Keep the lowest-eval_loss checkpoint and reload it at the end, so `out` is
        # the best model, not the last. save_steps must be a multiple of eval_steps.
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,  # best is always retained even if outside the window
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        run_name=run_label,
    )

    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset, eval_dataset=eval_dataset, processing_class=tokenizer
    )
    # resume=True auto-finds the latest checkpoint in `out` and restores optimizer,
    # scheduler, RNG and step count — the run continues, it does not restart.
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out))  # adapter only


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="base", help="registry name (base/instruct) or a HF id / path")
    p.add_argument("--out", type=Path, required=True, help="adapter output dir")
    p.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = full epochs; >0 caps steps (smoke runs)")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows; >0 truncates the dataset")
    p.add_argument("--eval-limit", type=int, default=200, help="dev rows for the held-out eval_loss curve")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --out")
    p.add_argument("--wandb", action="store_true", help="report to W&B (default: no reporting)")
    args = p.parse_args()

    train(
        args.model,
        args.out,
        args.data,
        epochs=args.epochs,
        max_steps=args.max_steps,
        limit=args.limit,
        eval_limit=args.eval_limit,
        resume=args.resume,
        report_to="wandb" if args.wandb else "none",
    )


if __name__ == "__main__":
    main()
