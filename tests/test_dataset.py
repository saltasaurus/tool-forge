from typing import Any, cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tool_forge import QWEN_4B_INSTRUCT
from tool_forge.dataset import render_prompt_completion

# Minimal single-turn {messages, tools} row, same shape as data/*.jsonl.
ROW: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Paris"},
                    },
                }
            ],
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ],
}


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    # Real Qwen3 tokenizer: the boundary depends on the actual chat template,
    # so a stub would prove nothing. Session-scoped -> loaded once.
    return AutoTokenizer.from_pretrained(QWEN_4B_INSTRUCT)


# --- render_prompt_completion ---------------------------------------------


def test_prompt_ends_at_assistant_opener(tokenizer: PreTrainedTokenizerBase) -> None:
    out = render_prompt_completion(ROW, tokenizer)
    assert out["prompt"].endswith("<|im_start|>assistant\n")


def test_reconstructs_full_render(tokenizer: PreTrainedTokenizerBase) -> None:
    # The split is a pure prefix cut: prompt is a true prefix of the full render
    # and the halves rejoin exactly -> nothing added or lost at the seam.
    out = render_prompt_completion(ROW, tokenizer)
    full = cast(str, tokenizer.apply_chat_template(ROW["messages"], tools=ROW["tools"], tokenize=False))
    assert full.startswith(out["prompt"] + out["completion"])  # Full - EOS token


def test_query_and_schema_in_prompt_call_in_completion(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    out = render_prompt_completion(ROW, tokenizer)
    # loss fires on the assistant call ...
    assert "get_weather" in out["completion"]
    assert "Paris" in out["completion"]
    # ... while the user's words and the tool schema sit in the masked prompt.
    assert "weather in Paris?" in out["prompt"]
    assert "weather in Paris?" not in out["completion"]
    assert "get_weather" in out["prompt"]  # schema, not the call


def test_completion_closes_the_turn(tokenizer: PreTrainedTokenizerBase) -> None:
    # Completion must not include the turn terminator, only tool call.
    out = render_prompt_completion(ROW, tokenizer)
    assert out["completion"].endswith("</tool_call>")
    assert str(tokenizer.eos_token) not in out["completion"]
