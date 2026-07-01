import pytest
from pydantic import ValidationError

from tool_forge.const import QWEN_4B_BASE, QWEN_4B_INSTRUCT
from tool_forge.models import BASE, INSTRUCT, MODEL_REGISTRY, ModelSpec, resolve

# --- resolve --------------------------------------------------------------


def test_resolve_registered_names_return_specs() -> None:
    assert resolve("base") is BASE
    assert resolve("instruct") is INSTRUCT


def test_resolve_unknown_arg_is_a_local_checkpoint() -> None:
    spec = resolve("/models/sft-base")
    assert spec.weights == "/models/sft-base"  # the path becomes the weights
    assert spec.tokenizer == QWEN_4B_INSTRUCT  # tokenizer defaults to the tool-aware one
    assert spec.name == "sft-base"  # basename as the label


# --- spec invariants (the reason ModelSpec exists) ------------------------


def test_base_pairs_base_weights_with_instruct_tokenizer() -> None:
    assert BASE.weights == QWEN_4B_BASE
    assert BASE.tokenizer == QWEN_4B_INSTRUCT  # weights and tokenizer differ, on purpose


def test_instruct_uses_one_id_for_both() -> None:
    assert INSTRUCT.weights == INSTRUCT.tokenizer == QWEN_4B_INSTRUCT


def test_registry_is_keyed_by_name() -> None:
    assert MODEL_REGISTRY == {"instruct": INSTRUCT, "base": BASE}


# --- ModelSpec is a frozen, closed schema ---------------------------------


def test_modelspec_is_frozen() -> None:
    with pytest.raises(ValidationError):
        BASE.name = "nope"


def test_modelspec_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(name="x", weights="y", tokenizer="z", junk="w")  # type: ignore[call-arg]
