"""
Inspect BFCL's expected tool call signature to ensure model is trained with correct FORMAT.
DO NOT train or inspect all data
"""

import json
from importlib.resources import files

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

def discover_test_files(data_dir) -> None:
    """Print files in directory"""
    for file in data_dir.iterdir():
        print(file)

def load_file(file_path: str):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result

def main() -> None:
    data_dir = files("bfcl_eval") / "data"
    
    # discover_test_files(data_dir)

    # Choose basic python for simplest response format
    simple_python_jsonl = data_dir / "BFCL_v4_simple_python.json"

    data = load_file(str(simple_python_jsonl))
    record: dict = data[0]

    # View self-trained models required format
    # dict_keys(['id', 'question', 'function'])
    print(record.keys())

    # Record[:]
    messages = record["question"][0]
    tools = record["function"]

    print("Messages:", messages)
    print("Tools:", tools)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    rendered = tok.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False              # We want to read the STRING
    )
    print(rendered)

if __name__ == "__main__":
    main()