import json
from pathlib import Path
from typing import Any


def load_dataset(path: Path) -> Any:
    jsonl = []
    with open(path) as f:
        for line in f.readlines():
            jsonl.append(json.loads(line))
    return jsonl

def render_prompt_completion(row, tokenizer):
    # Render to the same prompt/completion boundary verified in the probe, but as strings.
    user, assistant = row["messages"]
    prompt = tokenizer.apply_chat_template(
        [user], tools=row["tools"], add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(
        [user, assistant], tools=row["tools"], tokenize=False)
    return {"prompt": prompt, "completion": full[len(prompt):]}