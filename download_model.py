"""
Download and pin the original SmolLM2 model locally.

Run:

    python download_model.py

The model will be stored under:

    models/base/
"""

import json

from huggingface_hub import (
    HfApi,
    snapshot_download,
)

from config import (
    ROOT,
    MODEL,
)

OUTPUT_DIR = (
        ROOT
        / "models"
        / "base"
)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Checking model: {MODEL}"
    )

    api = HfApi()

    info = api.model_info(
        MODEL
    )

    revision = info.sha

    print(
        f"Pinned Hugging Face revision: "
        f"{revision}"
    )

    print()

    print(
        "Downloading model..."
    )

    snapshot_download(

        repo_id=MODEL,

        revision=revision,

        local_dir=str(
            OUTPUT_DIR
        )
    )

    metadata = {

        "model":
            MODEL,

        "revision":
            revision,
    }

    (
            OUTPUT_DIR
            / "model_source.json"
    ).write_text(

        json.dumps(
            metadata,
            indent=2
        ),

        encoding="utf-8"
    )

    print()

    print(
        "Original model downloaded to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "This exact local copy will now "
        "be used for both the BEFORE and "
        "AFTER experiment."
    )


if __name__ == "__main__":
    main()
