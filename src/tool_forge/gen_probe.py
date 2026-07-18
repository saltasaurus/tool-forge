"""Free-generation probe: a TrainerCallback that surfaces degeneration during SFT.

Teacher-forced eval (`eval_loss`, token accuracy) cannot see autoregressive
degeneration — it always feeds the gold prefix, so the model never spirals. This probe
greedily generates on a small fixed prompt set each eval and logs two numbers to W&B:

    gen/hit_cap_rate  - fraction that ran to the token cap WITHOUT emitting EOS
                        (failed to stop — the direct degeneration signal)
    gen/closed_rate   - fraction whose output closed the <tool_call> wrapper

The probe is best-effort: it never raises into the training loop (an OOM or generation
error is logged and skipped), so a long run is never killed by the monitor.
"""

import logging
from typing import Any

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class GenerationProbeCallback(TrainerCallback):
    """Log free-generation degeneration metrics after each eval."""

    def __init__(
        self,
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> None:
        self._prompts = prompts
        self._tok = tokenizer
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        self._eos = tokenizer.eos_token_id
        # Pad with a token distinct from EOS so "EOS present" reliably means the model
        # chose to stop, not that generate() right-padded a finished sequence.
        pad = tokenizer.pad_token_id
        if pad is None or pad == self._eos:
            pad = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self._pad = pad

    @torch.inference_mode()
    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model: PreTrainedModel = kwargs["model"]
        was_training = model.training
        prev_side = self._tok.padding_side
        model.eval()
        self._tok.padding_side = "left"  # so new tokens start at one uniform offset
        try:
            hit_cap, closed, n = self._run(model)
        except torch.cuda.OutOfMemoryError:
            logger.warning("gen probe OOM at step %d — skipping (lower batch_size)", state.global_step)
            torch.cuda.empty_cache()
            return
        finally:
            self._tok.padding_side = prev_side
            if was_training:
                model.train()

        metrics = {"gen/hit_cap_rate": hit_cap / n, "gen/closed_rate": closed / n}
        logger.info("gen probe @ step %d: %s", state.global_step, metrics)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass

    def _run(self, model: PreTrainedModel) -> tuple[int, int, int]:
        hit_cap = closed = n = 0
        for i in range(0, len(self._prompts), self._batch_size):
            batch = self._prompts[i : i + self._batch_size]
            inputs = self._tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                eos_token_id=self._eos,
                pad_token_id=self._pad,
            )
            new = out[:, inputs["input_ids"].shape[1] :]
            for row in new.tolist():
                if self._eos not in row:  # never stopped -> ran to the cap
                    hit_cap += 1
                if "</tool_call>" in self._tok.decode(row, skip_special_tokens=True):
                    closed += 1
                n += 1
        return hit_cap, closed, n
