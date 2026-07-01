import json
from pathlib import Path
from typing import Any, cast

from transformers import PreTrainedTokenizerBase


def load_dataset(path: Path) -> list[dict[str, Any]]:
    jsonl: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f.readlines():
            jsonl.append(json.loads(line))
    return jsonl


def render_prompt_completion(row: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    # Render to the same prompt/completion boundary verified in the probe, but as strings.
    user, assistant = row["messages"]
    prompt = cast(str, tokenizer.apply_chat_template(
        [user], tools=row["tools"], add_generation_prompt=True, tokenize=False))
    full = cast(str, tokenizer.apply_chat_template(
        [user, assistant], tools=row["tools"], tokenize=False))
    return {"prompt": prompt, "completion": full[len(prompt):]}
