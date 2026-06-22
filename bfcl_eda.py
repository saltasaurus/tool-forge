"""
Inspect BFCL's expected tool call signature to ensure model is trained with correct FORMAT.
DO NOT train or inspect all data
"""

import json
from importlib.resources import files


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
    
    discover_test_files(data_dir)

    # Choose basic python for simplest response format
    simple_python_jsonl = data_dir / "BFCL_v4_simple_python.json"

    data = load_file(str(simple_python_jsonl))
    question: dict = data[0]

    # View self-trained models required format
    print(question.keys())


if __name__ == "__main__":
    main()