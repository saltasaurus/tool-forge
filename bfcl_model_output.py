import json


def main():

    FILE = "bfcl_results/Qwen_Qwen3-4B-Instruct-2507-FC/non_live/BFCL_v4_simple_python_result.json"
    data = []
    with open(FILE) as f:
        file = f.readlines()
        for line in file:
            data.append(json.loads(line))

    # Read first response
    print(data[0])

if __name__ == "__main__":
    main()