import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ROOT, MODEL


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = ROOT / "models" / "base"
ADAPTER_DIR = ROOT / "adapter"
MERGED_DIR = ROOT / "models" / "merged"

OLLAMA_AFTER_DIR = ROOT / "ollama" / "after"
GGUF_FILE = OLLAMA_AFTER_DIR / "democart-smollm2-135m-finetuned.gguf"

# llama.cpp is OUTSIDE ecommerce-ai
LLAMA_CPP_DIR = ROOT.parent / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"


# ---------------------------------------------------------
# Merge LoRA adapter into base model
# ---------------------------------------------------------

def merge_model():

    if not ADAPTER_DIR.exists():
        raise RuntimeError(
            "adapter directory not found. "
            "Run fine-tuning first."
        )

    experiment_file = ADAPTER_DIR / "experiment.json"

    if not experiment_file.exists():
        raise RuntimeError(
            f"Missing file: {experiment_file}"
        )

    metadata = json.loads(
        experiment_file.read_text(encoding="utf-8")
    )

    revision = metadata.get("revision")

    # Prefer local downloaded base model
    if BASE_DIR.exists() and (BASE_DIR / "config.json").exists():
        source = str(BASE_DIR)
        kwargs = {}

        print(f"Using local base model: {source}")

    else:
        source = MODEL
        kwargs = {}

        if revision:
            kwargs["revision"] = revision

        print(f"Using Hugging Face base model: {source}")

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        **kwargs
    )

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float32,
        **kwargs
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR)
    )

    print("Merging LoRA adapter into base model...")

    merged_model = model.merge_and_unload()

    MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Saving merged model to: {MERGED_DIR}")

    merged_model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True
    )

    tokenizer.save_pretrained(
        MERGED_DIR
    )

    merge_info = {
        "base_model": MODEL,
        "revision": revision,
        "adapter": str(ADAPTER_DIR),
        "merged_model": str(MERGED_DIR)
    }

    (MERGED_DIR / "merge_info.json").write_text(
        json.dumps(
            merge_info,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Model merge completed.")


# ---------------------------------------------------------
# Convert merged model to GGUF
# ---------------------------------------------------------

def convert_to_gguf():

    if not CONVERT_SCRIPT.exists():
        raise RuntimeError(
            "llama.cpp conversion script not found:\n"
            f"{CONVERT_SCRIPT}\n\n"
            "Expected llama.cpp at:\n"
            f"{LLAMA_CPP_DIR}"
        )

    if not MERGED_DIR.exists():
        raise RuntimeError(
            f"Merged model not found: {MERGED_DIR}"
        )

    OLLAMA_AFTER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Converting merged model to GGUF...")

    command = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(MERGED_DIR),
        "--outfile",
        str(GGUF_FILE),
        "--outtype",
        "f16"
    ]

    print("Running:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True
    )

    if not GGUF_FILE.exists():
        raise RuntimeError(
            "GGUF conversion finished but output file was not created."
        )

    print(f"GGUF created: {GGUF_FILE}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("DemoCart SmolLM2-135M Merge + GGUF Conversion")
    print("=" * 60)

    merge_model()

    print()

    convert_to_gguf()

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    print()
    print("Fine-tuned GGUF model:")
    print(GGUF_FILE)

    print()
    print("Next command:")
    print(
        r"ollama create democart-smollm-after "
        r"-f .\ollama\after\Modelfile"
    )


if __name__ == "__main__":
    main()