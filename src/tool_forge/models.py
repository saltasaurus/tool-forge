from pathlib import Path

import torch
from peft import prepare_model_for_kbit_training
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from tool_forge.const import QWEN_4B_BASE, QWEN_4B_INSTRUCT


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    weights: str
    tokenizer: str


INSTRUCT = ModelSpec(name="instruct", weights=QWEN_4B_INSTRUCT, tokenizer=QWEN_4B_INSTRUCT)

BASE = ModelSpec(
    name="base",
    weights=QWEN_4B_BASE,
    tokenizer=QWEN_4B_INSTRUCT,
)

MODEL_REGISTRY = {m.name: m for m in (INSTRUCT, BASE)}


def resolve(arg: str) -> ModelSpec:
    return MODEL_REGISTRY.get(arg) or ModelSpec(name=Path(arg).name, weights=arg, tokenizer=INSTRUCT.tokenizer)


def load_tokenizer(spec: ModelSpec) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(spec.tokenizer)


def load_model(spec: ModelSpec, *, quantized: bool = True, for_training: bool = False) -> PreTrainedModel:

    nf4_config = _get_quantized_config() if quantized else None

    model = AutoModelForCausalLM.from_pretrained(
        spec.weights, quantization_config=nf4_config, dtype=torch.bfloat16, device_map={"": 0}
    )

    if for_training:
        model = prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call]

    return model


def _get_quantized_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
