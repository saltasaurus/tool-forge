"""Version the built dataset (data/) as a W&B Artifact.

Separate from build_dataset.py by design: building is deterministic and offline,
uploading needs auth + network. Run build_dataset.py first.
"""

import json
from pathlib import Path

import wandb

DATA = Path(__file__).resolve().parents[1] / "data"
PROJECT = "tool-forge"
ARTIFACT_NAME = "xlam-toolcalls"  # stable name; W&B auto-versions (:v0, :v1, ...)
FILES = ["train.jsonl", "dev.jsonl", "test.jsonl", "meta.json"]


def main() -> None:
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))

    with wandb.init(project=PROJECT, job_type="build-dataset") as run:
        artifact = wandb.Artifact(
            name=ARTIFACT_NAME,
            type="dataset",
            description=(
                "xLAM-60k normalized to schema, gold calls verified, "
                "seeded 80/10/10 split, rendered to Qwen3-FC {messages, tools}."
            ),
            metadata=meta,  # indexed by W&B: seed, git_sha, counts, quarantine rate
        )
        for name in FILES:
            artifact.add_file(str(DATA / name))
        run.log_artifact(artifact)


if __name__ == "__main__":
    main()
