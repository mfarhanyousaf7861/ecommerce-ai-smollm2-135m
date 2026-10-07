"""
DemoCart SLM fine-tuning experiment.

Commands:

    python experiment.py validate
    python experiment.py baseline

    # BEFORE fine-tuning
    python experiment.py base

    # Fine-tune
    python experiment.py train --epochs 3 --threads 4 --batch-size 4 --log-every 10

    # AFTER fine-tuning
    python experiment.py tuned

    # Convenience: evaluate both after training
    python experiment.py evaluate

Model paths and result paths are relative to this file.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from config import (
    ROOT,
    MODEL,
    LABELS,
    classification_instructions,
)

RESULTS = ROOT / "results"

BASE_MODEL_DIR = ROOT / "models" / "base"

ADAPTER_DIR = ROOT / "adapter"


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def rows(name):
    path = ROOT / "data" / f"{name}.jsonl"

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------
# Result saving
# ---------------------------------------------------------

def save(name, obj):
    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    path = RESULTS / name

    path.write_text(
        json.dumps(
            obj,
            indent=2
        ),
        encoding="utf-8"
    )


# ---------------------------------------------------------
# Environment information
# ---------------------------------------------------------

def environment():
    packages = {}

    for package in [
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "huggingface-hub",
    ]:

        try:
            packages[package] = (
                importlib.metadata.version(package)
            )

        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    hashes = {}

    data_dir = ROOT / "data"

    for path in sorted(data_dir.glob("*")):

        if path.is_file():
            hashes[path.name] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    return {
        "python": platform.python_version(),

        "platform": platform.platform(),

        "processor": platform.processor(),

        "packages": packages,

        "data_sha256": hashes,
    }


# ---------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------

def validate():
    """Validate split integrity and report train/test similarity diagnostics.

    This command is intentionally separate from training. The ``train`` command
    does not call this function and does not read ``validation.jsonl``.
    """

    split_names = [
        "train",
        "validation",
        "test",
    ]

    split_rows = {
        split: rows(split)
        for split in split_names
    }

    normalized_by_split = {}

    for split, data in split_rows.items():
        labels_in_split = {
            row["label"]
            for row in data
        }

        assert labels_in_split == set(LABELS), (
            f"{split} does not contain exactly the expected labels.\n"
            f"Expected: {set(LABELS)}\n"
            f"Actual:   {labels_in_split}"
        )

        normalized = [
            " ".join(
                row["text"]
                .lower()
                .split()
            )
            for row in data
        ]

        assert len(normalized) == len(set(normalized)), (
            f"Exact duplicate found inside {split}."
        )

        normalized_by_split[split] = set(normalized)

    # Exact cross-split duplicate checks.
    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = (
            normalized_by_split[left]
            & normalized_by_split[right]
        )

        assert not overlap, (
            f"Exact duplicate across {left}/{right}: "
            f"{next(iter(overlap))}"
        )

    # Near-duplicate audit. This is diagnostic only; it does not fail the run.
    # TF-IDF is fit jointly only for this dataset-audit command. No labels from
    # validation/test are used for model training here.
    train_text = [
        row["text"]
        for row in split_rows["train"]
    ]

    test_text = [
        row["text"]
        for row in split_rows["test"]
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        train_text + test_text
    )

    train_matrix = matrix[:len(train_text)]
    test_matrix = matrix[len(train_text):]

    similarities = cosine_similarity(
        test_matrix,
        train_matrix,
    )

    nearest_indices = similarities.argmax(axis=1)
    nearest_scores = similarities.max(axis=1)

    same_label_nearest = sum(
        split_rows["test"][index]["label"]
        == split_rows["train"][nearest_index]["label"]
        for index, nearest_index
        in enumerate(nearest_indices)
    )

    similarity_audit = {
        "test_examples": len(test_text),
        "median_nearest_train_similarity": float(
            np.median(nearest_scores)
        ),
        "max_nearest_train_similarity": float(
            np.max(nearest_scores)
        ),
        "similarity_gte_0_80": int(
            np.sum(nearest_scores >= 0.80)
        ),
        "similarity_gte_0_90": int(
            np.sum(nearest_scores >= 0.90)
        ),
        "nearest_train_same_label": int(
            same_label_nearest
        ),
        "nearest_train_same_label_rate": float(
            same_label_nearest / len(test_text)
        ),
        "note": (
            "Near-duplicate similarity is diagnostic only. "
            "Exact duplicates are rejected."
        ),
    }

    counts = {
        split: len(rows(split))
        for split in [
            "train",
            "validation",
            "test",
            "challenge",
            "rag_test",
        ]
    }

    counts["similarity_audit"] = similarity_audit

    return counts


# ---------------------------------------------------------
# Common metrics
# ---------------------------------------------------------

def metrics(
        data,
        predictions,
        latencies,
):
    expected = [
        row["label"]
        for row in data
    ]

    return {

        "n": len(expected),

        "accuracy": float(
            accuracy_score(
                expected,
                predictions
            )
        ),

        "macro_f1": float(
            f1_score(
                expected,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),

        "labels": LABELS,

        "confusion_matrix":
            confusion_matrix(
                expected,
                predictions,
                labels=LABELS,
            ).tolist(),

        "latency_ms_median": float(
            np.median(latencies)
        ),

        "latency_ms_p95": float(
            np.percentile(
                latencies,
                95
            )
        ),

        "latency_note":
            "Sequential single-request wall time "
            "after one warm-up, excluding model "
            "loading. This is not a production "
            "load test.",

        "predictions": [

            dict(
                row,
                predicted=prediction,
                correct=(
                        row["label"]
                        == prediction
                )
            )

            for row, prediction
            in zip(
                data,
                predictions
            )
        ],
    }


# =========================================================
# CLASSICAL ML BASELINE
# =========================================================

def baseline():
    validate()

    train_data = rows("train")

    test_data = rows("test")

    pipeline = make_pipeline(

        TfidfVectorizer(
            ngram_range=(1, 2)
        ),

        LogisticRegression(
            C=4,
            max_iter=1000,
            random_state=42,
        )
    )

    pipeline.fit(

        [
            row["text"]
            for row in train_data
        ],

        [
            row["label"]
            for row in train_data
        ]
    )

    # Warm-up
    pipeline.predict([
        "Where is my order?"
    ])

    predictions = []

    latencies = []

    for row in test_data:
        start = time.perf_counter()

        prediction = pipeline.predict([
            row["text"]
        ])[0]

        elapsed = (
                          time.perf_counter()
                          - start
                  ) * 1000

        predictions.append(
            prediction
        )

        latencies.append(
            elapsed
        )

    report = metrics(
        test_data,
        predictions,
        latencies,
    )

    report["environment"] = (
        environment()
    )

    report["method"] = (
        "Word TF-IDF + logistic regression; "
        "fixed hyperparameters; training "
        "split only."
    )

    save(
        "baseline.json",
        report
    )

    print(
        json.dumps(
            {
                "accuracy":
                    report["accuracy"],

                "macro_f1":
                    report["macro_f1"],

                "latency_ms_median":
                    report[
                        "latency_ms_median"
                    ],
            },
            indent=2
        )
    )


# =========================================================
# MODEL LOADING
# =========================================================

def get_base_source():
    """
    Prefer our locally pinned model if
    download_model.py has already been run.

    Otherwise use Hugging Face directly.
    """

    if (
            BASE_MODEL_DIR.exists()
            and
            (
                    BASE_MODEL_DIR
                    / "config.json"
            ).exists()
    ):
        return str(
            BASE_MODEL_DIR
        )

    return MODEL


def get_adapter_metadata():
    metadata_path = (
            ADAPTER_DIR
            / "experiment.json"
    )

    if not metadata_path.exists():
        return {}

    return json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )


def load_model(
        adapter=False
):
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from peft import PeftModel

    source = get_base_source()

    revision = None

    # If we do not have a local copy,
    # use the exact revision recorded
    # during fine-tuning when possible.

    if adapter:

        metadata = (
            get_adapter_metadata()
        )

        if source == MODEL:
            revision = (
                metadata.get(
                    "revision"
                )
            )

    kwargs = {}

    if revision:
        kwargs["revision"] = (
            revision
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            source,
            **kwargs
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            source,
            dtype=torch.float32,
            **kwargs
        )
    )

    if adapter:

        if not ADAPTER_DIR.exists():
            raise RuntimeError(
                "Fine-tuned adapter "
                "does not exist."
            )

        model = (
            PeftModel
            .from_pretrained(
                model,
                str(
                    ADAPTER_DIR
                )
            )
        )

    return (
        tokenizer,
        model
    )


# =========================================================
# PROMPT
# =========================================================

def prompt(
        tokenizer,
        text,
):
    messages = [

        {
            "role": "system",
            "content":
                classification_instructions()
        },

        {
            "role": "user",
            "content": text
        }
    ]

    return (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


# =========================================================
# LABEL SCORING
# =========================================================

def classify(
        tokenizer,
        model,
        text,
):
    """
    Score every allowed label as a candidate completion.

    We deliberately do NOT ask the Hugging Face model
    to freely generate an answer.

    Each allowed label receives a completion
    log-likelihood score and the highest-scoring
    label is selected.
    """

    import torch

    prefix = prompt(
        tokenizer,
        text
    )

    suffixes = [

        tokenizer.encode(
            label,
            add_special_tokens=False
        )

        for label in LABELS
    ]

    sequences = [

        prefix + suffix

        for suffix
        in suffixes
    ]

    maximum = max(
        map(
            len,
            sequences
        )
    )

    device = next(
        model.parameters()
    ).device

    ids = torch.full(

        (
            len(LABELS),
            maximum
        ),

        tokenizer.pad_token_id,

        dtype=torch.long,

        device=device
    )

    mask = torch.zeros_like(
        ids
    )

    for index, sequence in enumerate(
            sequences
    ):
        ids[
            index,
            :len(sequence)
        ] = torch.tensor(
            sequence,
            device=device
        )

        mask[
            index,
            :len(sequence)
        ] = 1

    with torch.inference_mode():
        logits = model(
            input_ids=ids,
            attention_mask=mask,
        ).logits

        log_probabilities = (
            logits[:, :-1]
            .log_softmax(-1)
        )

        token_scores = (
            log_probabilities
            .gather(
                -1,
                ids[:, 1:]
                .unsqueeze(-1)
            )
            .squeeze(-1)
        )

        scores = torch.stack([

            token_scores[
                index,
                len(prefix) - 1:
                len(sequence) - 1
            ].mean()

            for index, sequence
            in enumerate(sequences)
        ])

        ordered = torch.sort(
            scores,
            descending=True
        ).values

        margin = float(
            (
                    ordered[0]
                    -
                    ordered[1]
            ).item()
        )

        winner = int(
            scores.argmax().item()
        )

    return (
        LABELS[winner],
        margin
    )



def format_duration(seconds):
    """Return a compact human-readable duration for training logs."""
    if seconds is None or not np.isfinite(seconds):
        return "unknown"

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"

# =========================================================
# FINE-TUNING
# =========================================================

def train(args):
    """
    Fine-tune the model on train.jsonl only.

    IMPORTANT:
    - No dataset validation is run here.
    - validation.jsonl is never loaded here.
    - No validation metric is calculated between epochs.
    - The adapter from the final epoch is saved.
    """
    import math
    import torch

    from torch.utils.data import DataLoader

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )

    # -----------------------------------------------------
    # Reproducibility + CPU settings
    # -----------------------------------------------------

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)

    training_started = time.perf_counter()

    print()
    print("=" * 72, flush=True)
    print("[TRAIN] DemoCart LoRA fine-tuning", flush=True)
    print("=" * 72, flush=True)
    print(f"[TRAIN] Model            : {MODEL}", flush=True)
    print(f"[TRAIN] Device           : {args.device}", flush=True)
    print(f"[TRAIN] CPU threads      : {args.threads}", flush=True)
    print(f"[TRAIN] Epochs           : {args.epochs}", flush=True)
    print(f"[TRAIN] Batch size       : {args.batch_size}", flush=True)
    print(f"[TRAIN] Log every        : {args.log_every} batch(es)", flush=True)
    print("[TRAIN] Validation       : DISABLED", flush=True)
    print("[TRAIN] Validation split : NOT LOADED", flush=True)
    print("[TRAIN] Dataset validate(): NOT CALLED", flush=True)
    print("=" * 72, flush=True)

    # -----------------------------------------------------
    # Load base model
    # -----------------------------------------------------

    print("[TRAIN] Loading tokenizer and base model...", flush=True)
    model_load_started = time.perf_counter()

    tokenizer, base_model = load_model(adapter=False)

    model_load_seconds = time.perf_counter() - model_load_started

    revision = getattr(
        base_model.config,
        "_commit_hash",
        None,
    )

    print(
        f"[TRAIN] Base model loaded in {model_load_seconds:.1f}s",
        flush=True,
    )
    print(
        f"[TRAIN] Base source       : {get_base_source()}",
        flush=True,
    )
    print(
        f"[TRAIN] Model revision    : {revision or 'not reported'}",
        flush=True,
    )

    # -----------------------------------------------------
    # Attach LoRA adapter
    # -----------------------------------------------------

    print("[TRAIN] Attaching LoRA adapter...", flush=True)

    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        ),
    )

    model.config.use_cache = False
    model.to(args.device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_percent = (
        100.0 * trainable_parameters / total_parameters
        if total_parameters
        else 0.0
    )

    print(
        f"[TRAIN] Trainable params  : {trainable_parameters:,}",
        flush=True,
    )
    print(
        f"[TRAIN] Total params      : {total_parameters:,}",
        flush=True,
    )
    print(
        f"[TRAIN] Trainable percent : {trainable_percent:.4f}%",
        flush=True,
    )

    # -----------------------------------------------------
    # Encode one training example
    # -----------------------------------------------------

    def encode(row):
        prefix = prompt(
            tokenizer,
            row["text"],
        )

        target = (
            tokenizer.encode(
                row["label"],
                add_special_tokens=False,
            )
            + [tokenizer.eos_token_id]
        )

        input_ids = prefix + target
        labels = ([-100] * len(prefix)) + target

        return input_ids, labels

    # -----------------------------------------------------
    # Pad mini-batches
    # -----------------------------------------------------

    def collate(batch):
        max_length = max(
            len(item[0])
            for item in batch
        )

        input_ids = []
        attention_masks = []
        label_rows = []

        for ids, labels in batch:
            padding = max_length - len(ids)

            input_ids.append(
                ids
                + [tokenizer.pad_token_id] * padding
            )

            attention_masks.append(
                [1] * len(ids)
                + [0] * padding
            )

            label_rows.append(
                labels
                + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(label_rows),
        }

    # -----------------------------------------------------
    # Load TRAIN split only
    # -----------------------------------------------------

    print("[TRAIN] Loading train.jsonl only...", flush=True)
    train_data = rows("train")

    print(
        f"[TRAIN] Training examples : {len(train_data):,}",
        flush=True,
    )

    print("[TRAIN] Encoding training examples...", flush=True)
    encode_started = time.perf_counter()

    encoded_training = [
        encode(row)
        for row in train_data
    ]

    encode_seconds = time.perf_counter() - encode_started

    print(
        f"[TRAIN] Encoding complete in {encode_seconds:.1f}s",
        flush=True,
    )

    loader = DataLoader(
        encoded_training,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(42),
    )

    batches_per_epoch = len(loader)
    total_training_steps = batches_per_epoch * args.epochs

    print(
        f"[TRAIN] Batches/epoch     : {batches_per_epoch:,}",
        flush=True,
    )
    print(
        f"[TRAIN] Total train steps : {total_training_steps:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=3e-4,
    )

    print("[TRAIN] Optimizer         : AdamW", flush=True)
    print("[TRAIN] Learning rate     : 0.0003", flush=True)
    print("[TRAIN] Gradient clip     : 1.0", flush=True)
    print("=" * 72, flush=True)

    history = []
    global_step = 0

    ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Epoch loop -- TRAINING ONLY, NO VALIDATION
    # -----------------------------------------------------

    optimization_started = time.perf_counter()

    for epoch_index in range(args.epochs):
        epoch_number = epoch_index + 1
        epoch_started = time.perf_counter()

        model.train()

        losses = []

        print()
        print("-" * 72, flush=True)
        print(
            f"[TRAIN][EPOCH {epoch_number}/{args.epochs}] START",
            flush=True,
        )
        print("-" * 72, flush=True)

        for batch_index, batch in enumerate(loader, start=1):
            step_started = time.perf_counter()

            batch = {
                key: value.to(args.device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            loss = model(**batch).loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )

            optimizer.step()

            loss_value = float(loss.item())
            losses.append(loss_value)
            global_step += 1

            should_log = (
                batch_index == 1
                or batch_index == batches_per_epoch
                or batch_index % args.log_every == 0
            )

            if should_log:
                now = time.perf_counter()
                epoch_elapsed = now - epoch_started
                total_elapsed = now - optimization_started

                average_loss = float(np.mean(losses))
                batches_per_second = (
                    batch_index / epoch_elapsed
                    if epoch_elapsed > 0
                    else 0.0
                )

                remaining_batches_this_epoch = (
                    batches_per_epoch - batch_index
                )

                remaining_steps_total = (
                    total_training_steps - global_step
                )

                eta_epoch_seconds = (
                    remaining_batches_this_epoch / batches_per_second
                    if batches_per_second > 0
                    else math.inf
                )

                overall_steps_per_second = (
                    global_step / total_elapsed
                    if total_elapsed > 0
                    else 0.0
                )

                eta_total_seconds = (
                    remaining_steps_total / overall_steps_per_second
                    if overall_steps_per_second > 0
                    else math.inf
                )

                step_seconds = now - step_started

                print(
                    (
                        f"[TRAIN][EPOCH {epoch_number}/{args.epochs}] "
                        f"batch={batch_index}/{batches_per_epoch} "
                        f"global_step={global_step}/{total_training_steps} "
                        f"loss={loss_value:.6f} "
                        f"avg_loss={average_loss:.6f} "
                        f"step={step_seconds:.2f}s "
                        f"epoch_elapsed={format_duration(epoch_elapsed)} "
                        f"epoch_eta={format_duration(eta_epoch_seconds)} "
                        f"total_eta={format_duration(eta_total_seconds)}"
                    ),
                    flush=True,
                )

        epoch_seconds = time.perf_counter() - epoch_started
        mean_epoch_loss = float(np.mean(losses)) if losses else float("nan")

        record = {
            "epoch": epoch_number,
            "mean_train_loss": mean_epoch_loss,
            "training_seconds": epoch_seconds,
            "batches": batches_per_epoch,
        }

        history.append(record)

        print(
            f"[TRAIN][EPOCH {epoch_number}/{args.epochs}] COMPLETE",
            flush=True,
        )
        print(
            f"[TRAIN][EPOCH {epoch_number}/{args.epochs}] "
            f"mean_loss={mean_epoch_loss:.6f} "
            f"duration={format_duration(epoch_seconds)}",
            flush=True,
        )
        print(
            f"[TRAIN][EPOCH {epoch_number}/{args.epochs}] "
            "validation=SKIPPED",
            flush=True,
        )

    # -----------------------------------------------------
    # Save FINAL epoch adapter
    # -----------------------------------------------------

    print()
    print("=" * 72, flush=True)
    print("[TRAIN] All epochs completed.", flush=True)
    print("[TRAIN] Saving FINAL epoch adapter...", flush=True)

    save_started = time.perf_counter()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    save_seconds = time.perf_counter() - save_started
    training_seconds = time.perf_counter() - training_started

    metadata = {
        "model": MODEL,
        "base_source": get_base_source(),
        "revision": revision,
        "selected_epoch": args.epochs,
        "selection_method": "final_epoch_no_validation",
        "validation_during_training": False,
        "seed": 42,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "device": args.device,
    }

    (ADAPTER_DIR / "experiment.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    report = {
        "training_seconds": training_seconds,
        "history": history,
        "validation_during_training": False,
        "selection_method": "final_epoch_no_validation",
        "saved_epoch": args.epochs,
        "model": MODEL,
        "revision": revision,
        "device": args.device,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "log_every": args.log_every,
        "train_examples": len(train_data),
        "batches_per_epoch": batches_per_epoch,
        "total_training_steps": total_training_steps,
        "seed": 42,
        "environment": environment(),
    }

    save("training.json", report)

    print(
        f"[TRAIN] Adapter saved     : {ADAPTER_DIR}",
        flush=True,
    )
    print(
        f"[TRAIN] Save duration     : {format_duration(save_seconds)}",
        flush=True,
    )
    print(
        f"[TRAIN] Total duration    : {format_duration(training_seconds)}",
        flush=True,
    )
    print(
        f"[TRAIN] Final mean loss   : {history[-1]['mean_train_loss']:.6f}",
        flush=True,
    )
    print("[TRAIN] Validation        : NOT PERFORMED", flush=True)
    print("=" * 72, flush=True)
    print()
    print("Next command:", flush=True)
    print("python experiment.py tuned", flush=True)


# =========================================================
# CONFIDENCE / MARGIN THRESHOLD
# =========================================================

def choose_threshold(
        data,
        values,
):
    """
    Validation-only heuristic.

    Select the threshold that gives the
    greatest coverage while maintaining
    >=90% accuracy among accepted
    validation predictions.

    Margin is NOT a calibrated probability.
    """

    choices = []

    candidate_thresholds = (
            [0.0]
            +
            sorted(
                set(
                    margin

                    for prediction, margin
                    in values
                )
            )
    )

    for threshold in candidate_thresholds:

        accepted = [

            index

            for index, (
                prediction,
                margin
            )
            in enumerate(values)

            if margin >= threshold
        ]

        if not accepted:
            continue

        accuracy = (

                sum(

                    values[index][0]
                    ==
                    data[index]["label"]

                    for index
                    in accepted
                )

                /
                len(accepted)
        )

        if accuracy >= 0.90:
            choices.append(

                (
                    len(accepted),
                    -threshold,
                    threshold,
                )
            )

    if not choices:
        return 1e9

    return max(
        choices
    )[2]


# =========================================================
# BASE/TUNED EVALUATION
# =========================================================

def evaluate_condition(
        args,
        tuned,
):
    import torch

    torch.set_num_threads(
        args.threads
    )

    name = (
        "tuned"
        if tuned
        else "base"
    )

    label = name.upper()
    evaluation_started = time.perf_counter()

    print()
    print("=" * 72, flush=True)
    print(f"[{label}] Starting model evaluation", flush=True)
    print("=" * 72, flush=True)
    print(f"[{label}] Condition         : {'fine-tuned adapter' if tuned else 'original base model'}", flush=True)
    print(f"[{label}] Device            : {args.device}", flush=True)
    print(f"[{label}] CPU threads       : {args.threads}", flush=True)
    print(f"[{label}] Base source       : {get_base_source()}", flush=True)
    print(f"[{label}] Allowed labels    : {len(LABELS)}", flush=True)
    print("=" * 72, flush=True)

    if tuned:
        if not (
                ADAPTER_DIR
                / "experiment.json"
        ).exists():
            raise SystemExit(
                "No trained adapter exists.\n"
                "Run first:\n"
                "python experiment.py train"
            )

    # -----------------------------------------------------
    # Load tokenizer/model
    # -----------------------------------------------------

    print(f"[{label}] Loading tokenizer and model...", flush=True)
    model_load_started = time.perf_counter()

    tokenizer, model = load_model(
        adapter=tuned
    )

    model_load_seconds = (
        time.perf_counter()
        - model_load_started
    )

    print(
        f"[{label}] Model loaded in    : {format_duration(model_load_seconds)}",
        flush=True,
    )

    if tuned:
        print(
            f"[{label}] Adapter source     : {ADAPTER_DIR}",
            flush=True,
        )

    print(
        f"[{label}] Moving model to    : {args.device}",
        flush=True,
    )

    device_started = time.perf_counter()

    model.to(
        args.device
    )

    model.eval()

    print(
        f"[{label}] Model ready in     : "
        f"{format_duration(time.perf_counter() - device_started)}",
        flush=True,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[{label}] Parameters         : {total_parameters:,}",
        flush=True,
    )

    # -----------------------------------------------------
    # Helper for per-request progress logging
    # -----------------------------------------------------

    def log_request(
            stage,
            index,
            total,
            row,
            prediction,
            margin,
            request_seconds,
            stage_started,
            correct_so_far,
    ):
        elapsed = time.perf_counter() - stage_started
        average_seconds = (
            elapsed / index
            if index > 0
            else 0.0
        )
        remaining = total - index
        eta = average_seconds * remaining
        running_accuracy = (
            100.0 * correct_so_far / index
            if index > 0
            else 0.0
        )

        request_text = " ".join(
            row["text"].split()
        )

        if len(request_text) > 120:
            request_text = (
                request_text[:117]
                + "..."
            )

        expected = row.get("label", row.get("expected"))

        print(
            (
                f"[{label}][{stage} {index}/{total}] "
                f"expected={expected} "
                f"predicted={prediction} "
                f"correct={'YES' if prediction == expected else 'NO'} "
                f"margin={margin:.6f} "
                f"time={request_seconds:.2f}s "
                f"running_accuracy={running_accuracy:.2f}% "
                f"elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(eta)}"
            ),
            flush=True,
        )
        print(
            f"[{label}][{stage} {index}/{total}] text=\"{request_text}\"",
            flush=True,
        )

    # -----------------------------------------------------
    # Choose threshold using validation only
    # -----------------------------------------------------

    validation = rows(
        "validation"
    )

    print()
    print("-" * 72, flush=True)
    print(
        f"[{label}] STAGE 1/3: Validation threshold selection",
        flush=True,
    )
    print(
        f"[{label}] Validation examples: {len(validation):,}",
        flush=True,
    )
    print(
        f"[{label}] NOTE: this is evaluation-time threshold selection only; "
        "it is NOT training validation.",
        flush=True,
    )
    print("-" * 72, flush=True)

    validation_values = []
    validation_started = time.perf_counter()
    validation_correct = 0

    for index, row in enumerate(
            validation,
            start=1,
    ):
        request_started = time.perf_counter()

        prediction, margin = classify(
            tokenizer,
            model,
            row["text"],
        )

        request_seconds = (
            time.perf_counter()
            - request_started
        )

        validation_values.append(
            (
                prediction,
                margin,
            )
        )

        if prediction == row["label"]:
            validation_correct += 1

        log_request(
            "VALIDATION",
            index,
            len(validation),
            row,
            prediction,
            margin,
            request_seconds,
            validation_started,
            validation_correct,
        )

    threshold = choose_threshold(
        validation,
        validation_values
    )

    validation_seconds = (
        time.perf_counter()
        - validation_started
    )

    print(
        f"[{label}] Validation complete : {format_duration(validation_seconds)}",
        flush=True,
    )
    print(
        f"[{label}] Selected threshold  : {threshold:.6f}",
        flush=True,
    )

    # -----------------------------------------------------
    # Warm-up - not counted in latency
    # -----------------------------------------------------

    print()
    print("-" * 72, flush=True)
    print(f"[{label}] STAGE 2/3: Warm-up", flush=True)
    print("-" * 72, flush=True)
    print(
        f"[{label}] Running one warm-up request...",
        flush=True,
    )

    warmup_started = time.perf_counter()

    warmup_prediction, warmup_margin = classify(
        tokenizer,
        model,
        "Where is my order?"
    )

    print(
        (
            f"[{label}] Warm-up complete    : "
            f"prediction={warmup_prediction} "
            f"margin={warmup_margin:.6f} "
            f"time={time.perf_counter() - warmup_started:.2f}s"
        ),
        flush=True,
    )

    # -----------------------------------------------------
    # Evaluate untouched TEST set
    # -----------------------------------------------------

    test_data = rows(
        "test"
    )

    print()
    print("-" * 72, flush=True)
    print(
        f"[{label}] STAGE 3/3: Untouched test-set evaluation",
        flush=True,
    )
    print(
        f"[{label}] Test examples      : {len(test_data):,}",
        flush=True,
    )
    print("-" * 72, flush=True)

    predictions = []
    margins = []
    latencies = []

    test_started = time.perf_counter()
    test_correct = 0

    for index, row in enumerate(
            test_data,
            start=1,
    ):
        start = time.perf_counter()

        prediction, margin = classify(
            tokenizer,
            model,
            row["text"]
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        predictions.append(
            prediction
        )

        margins.append(
            margin
        )

        latencies.append(
            elapsed_ms
        )

        if prediction == row["label"]:
            test_correct += 1

        log_request(
            "TEST",
            index,
            len(test_data),
            row,
            prediction,
            margin,
            elapsed_ms / 1000.0,
            test_started,
            test_correct,
        )

    test_seconds = (
        time.perf_counter()
        - test_started
    )

    print(
        f"[{label}] Test evaluation complete: {format_duration(test_seconds)}",
        flush=True,
    )

    report = metrics(
        test_data,
        predictions,
        latencies,
    )

    accepted = [
        index
        for index, margin
        in enumerate(margins)
        if margin >= threshold
    ]

    if accepted:
        accepted_accuracy = (
            sum(
                predictions[index]
                ==
                test_data[index]["label"]
                for index
                in accepted
            )
            /
            len(accepted)
        )
    else:
        accepted_accuracy = None

    report.update({
        "condition":
            (
                "fine_tuned"
                if tuned
                else
                "original_base_model"
            ),

        "method":
            "Mean label-completion log likelihood "
            f"across the {len(LABELS)} allowed candidates.",

        "margin_note":
            "Top-two mean log-likelihood gap. "
            "This is NOT a calibrated probability "
            "or an OOD detector.",

        "threshold":
            threshold,

        "coverage":
            (
                len(accepted)
                /
                len(test_data)
            ),

        "accepted_accuracy":
            accepted_accuracy,

        "device":
            args.device,

        "threads":
            args.threads,

        "environment":
            environment(),
    })

    for item, margin in zip(
            report["predictions"],
            margins
    ):
        item["margin"] = margin

        item["accepted"] = (
            margin
            >=
            threshold
        )

    # -----------------------------------------------------
    # Challenge cases
    # -----------------------------------------------------

    challenge_data = rows(
        "challenge"
    )

    print()
    print("-" * 72, flush=True)
    print(
        f"[{label}] CHALLENGE: Evaluating {len(challenge_data):,} cases",
        flush=True,
    )
    print("-" * 72, flush=True)

    challenge_results = []
    challenge_started = time.perf_counter()
    challenge_correct = 0

    for index, row in enumerate(
            challenge_data,
            start=1,
    ):
        request_started = time.perf_counter()

        prediction, margin = classify(
            tokenizer,
            model,
            row["text"]
        )

        request_seconds = (
            time.perf_counter()
            - request_started
        )

        if prediction == row["expected"]:
            challenge_correct += 1

        log_request(
            "CHALLENGE",
            index,
            len(challenge_data),
            row,
            prediction,
            margin,
            request_seconds,
            challenge_started,
            challenge_correct,
        )

        challenge_results.append(
            dict(
                row,
                predicted=prediction,
                margin=margin,
                accepted=(
                    margin
                    >=
                    threshold
                )
            )
        )

    report["challenge"] = (
        challenge_results
    )

    save(
        f"{name}.json",
        report
    )

    total_seconds = (
        time.perf_counter()
        - evaluation_started
    )

    print()
    print("=" * 72, flush=True)
    print(
        f"[{label}] EVALUATION COMPLETE",
        flush=True,
    )
    print("=" * 72, flush=True)

    print(
        json.dumps(
            {
                "accuracy":
                    report["accuracy"],

                "macro_f1":
                    report["macro_f1"],

                "latency_ms_median":
                    report[
                        "latency_ms_median"
                    ],

                "coverage":
                    report["coverage"],

                "accepted_accuracy":
                    report[
                        "accepted_accuracy"
                    ],

                "threshold":
                    report["threshold"],

                "total_duration":
                    format_duration(total_seconds),

                "results_file":
                    str(
                        RESULTS
                        / f"{name}.json"
                    ),
            },
            indent=2
        ),
        flush=True,
    )

    print("=" * 72, flush=True)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_base(args):
    evaluate_condition(
        args,
        tuned=False
    )


def evaluate_tuned(args):
    evaluate_condition(
        args,
        tuned=True
    )


def evaluate_both(args):
    if not (
            ADAPTER_DIR
            / "experiment.json"
    ).exists():
        raise SystemExit(
            "Train first before using "
            "'evaluate'.\n"
            "For pre-training evaluation use:\n"
            "python experiment.py base"
        )

    evaluate_base(
        args
    )

    evaluate_tuned(
        args
    )


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(

        "command",

        choices=[
            "validate",
            "baseline",
            "base",
            "train",
            "tuned",
            "evaluate",
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size. Lower this to 2 if RAM becomes tight.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print training progress every N batches. Use 1 to log every batch.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default="cpu",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.threads < 1:
        parser.error("--threads must be at least 1")

    if args.log_every < 1:
        parser.error("--log-every must be at least 1")

    if args.command == "validate":

        print(
            json.dumps(
                validate(),
                indent=2
            )
        )


    elif args.command == "baseline":

        baseline()


    elif args.command == "base":

        evaluate_base(
            args
        )


    elif args.command == "train":

        train(
            args
        )


    elif args.command == "tuned":

        evaluate_tuned(
            args
        )


    elif args.command == "evaluate":

        evaluate_both(
            args
        )


if __name__ == "__main__":
    main()
