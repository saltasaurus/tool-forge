from datasets import load_dataset

def main():
    ds = load_dataset("Salesforce/xlam-function-calling-60k")

    print(ds[:20])

if __name__ == "__main__":
    main()