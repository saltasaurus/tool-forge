import json
import logging
import subprocess
from pathlib import Path

from datasets import load_dataset

from tool_forge.format import format_conversation
from tool_forge.normalize import normalize_row
from tool_forge.schema import Conversation
from tool_forge.split import Conversations, split
from tool_forge.verify import keep_valid

logger = logging.getLogger(__name__)

DATASET = "Salesforce/xlam-function-calling-60k"
SEED = 0
# Anchor to repo root regardless of cwd: scripts/ -> root is parents[1].
OUT = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ds = load_dataset(DATASET, split="train")

    conversations: list[Conversation] = []
    normalize_failures = 0
    for row in ds:
        try:
            conversations.append(normalize_row(row))
        except (NotImplementedError, KeyError) as exc:
            # Row carries a type string normalize can't represent -> skip + count.
            normalize_failures += 1
            logger.debug("normalize failed on id=%s: %s", row.get("id"), exc)

    survivors, quarantined = keep_valid(conversations)
    train, dev, test = split(survivors, seed=SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    export_conversations(train, "train.jsonl")
    export_conversations(dev, "dev.jsonl")
    export_conversations(test, "test.jsonl")

    kept = len(survivors)
    rate = quarantined / (kept + quarantined) if kept + quarantined else 0.0

    meta: dict[str, object] = {
        "seed": SEED,
        "dataset": DATASET,
        "fingerprint": ds._fingerprint,  # HF cache fingerprint of this exact corpus
        "git_sha": git_sha(),
        "counts": {
            "total": len(ds),
            "normalize_failures": normalize_failures,
            "quarantined": quarantined,
            "kept": kept,
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "quarantine_rate_per_conversation": rate,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"total={len(ds)} normalize_failures={normalize_failures} "
        f"quarantined={quarantined} kept={kept} "
        f"(train={len(train)} dev={len(dev)} test={len(test)}) "
        f"quarantine_rate={rate:.4%}"
    )


def git_sha() -> str:
    """HEAD sha of the repo that produced this build — artifact lineage.

    Build from a clean tree so this sha fully reproduces the artifact.
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def export_conversations(conversations: Conversations, filename: str) -> None:
    path = OUT / filename
    print(f"Writing {len(conversations)} examples to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(format_conversation(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
