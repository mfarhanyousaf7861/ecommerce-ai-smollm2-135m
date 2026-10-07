"""
Evaluate an Ollama model against DemoCart's
unchanged test dataset.

BEFORE:

    python ollama_eval.py \
        democart-smollm-before \
        --output ollama_before.json

AFTER:

    python ollama_eval.py \
        democart-smollm-after \
        --output ollama_after.json

Both runs use exactly the same test.jsonl.
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from config import (
    ROOT,
    LABELS,
)

OLLAMA_URL = (
    "http://localhost:11434/api/chat"
)


# ---------------------------------------------------------
# Read dataset
# ---------------------------------------------------------

def rows(split):
    path = (
            ROOT
            / "data"
            / f"{split}.jsonl"
    )

    return [

        json.loads(line)

        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()

        if line.strip()
    ]


# ---------------------------------------------------------
# Call Ollama
# ---------------------------------------------------------

def call_ollama(
        model,
        text,
):
    body = {

        "model":
            model,

        "messages": [

            {
                "role":
                    "user",

                "content":
                    text,
            }
        ],

        "stream":
            False,

        "keep_alive":
            "10m",

        "options": {

            "temperature":
                0,

            "seed":
                42,

            "num_predict":
                20,
        }
    }

    encoded = json.dumps(
        body
    ).encode(
        "utf-8"
    )

    request = urllib.request.Request(

        OLLAMA_URL,

        data=encoded,

        headers={
            "Content-Type":
                "application/json"
        },

        method="POST",
    )

    start = (
        time.perf_counter()
    )

    try:

        with urllib.request.urlopen(
                request,
                timeout=180
        ) as response:

            payload = json.loads(
                response
                .read()
                .decode(
                    "utf-8"
                )
            )


    except urllib.error.URLError as exc:

        raise RuntimeError(
            "Unable to connect to Ollama.\n"
            "Make sure Ollama is running.\n"
            f"Original error: {exc}"
        ) from exc

    elapsed_ms = (
                         time.perf_counter()
                         -
                         start
                 ) * 1000

    content = (

        payload
        .get(
            "message",
            {}
        )
        .get(
            "content",
            ""
        )
        .strip()
    )

    return (
        content,
        elapsed_ms
    )


# ---------------------------------------------------------
# Convert model output to one allowed label
# ---------------------------------------------------------

def normalize_prediction(
        response,
):
    if not response:
        return "__INVALID__"

    cleaned = (
        response
        .strip()
        .upper()
    )

    # Remove common formatting noise.

    cleaned = (
        cleaned
        .replace(
            "`",
            ""
        )
        .replace(
            "*",
            ""
        )
        .strip()
    )

    # Perfect output.

    if cleaned in LABELS:
        return cleaned

    # Sometimes tiny models return:
    #
    # "TRACK_ORDER."
    #
    # or:
    #
    # "The intent is TRACK_ORDER"
    #
    # We accept it only when exactly ONE known
    # label can be found in the response.

    matches = [

        label

        for label
        in LABELS

        if label
           in cleaned
    ]

    if len(matches) == 1:
        return matches[0]

    return "__INVALID__"


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_report(
        data,
        predictions,
        raw_responses,
        latencies,
        model,
):
    expected = [

        row["label"]

        for row
        in data
    ]

    accuracy = float(

        accuracy_score(
            expected,
            predictions
        )
    )

    macro_f1 = float(

        f1_score(

            expected,
            predictions,

            labels=LABELS,

            average="macro",

            zero_division=0,
        )
    )

    matrix = (
        confusion_matrix(

            expected,
            predictions,

            labels=LABELS
        )
        .tolist()
    )

    invalid_count = sum(

        prediction
        ==
        "__INVALID__"

        for prediction
        in predictions
    )

    details = []

    for (
            row,
            prediction,
            raw_response,
            latency
    ) in zip(
        data,
        predictions,
        raw_responses,
        latencies,
    ):
        details.append({

            "text":
                row["text"],

            "expected":
                row["label"],

            "predicted":
                prediction,

            "raw_response":
                raw_response,

            "correct":
                (
                        prediction
                        ==
                        row["label"]
                ),

            "latency_ms":
                latency,
        })

    return {

        "model":
            model,

        "n":
            len(data),

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "labels":
            LABELS,

        "confusion_matrix":
            matrix,

        "invalid_predictions":
            invalid_count,

        "latency_ms_median":
            float(
                statistics.median(
                    latencies
                )
            ),

        "latency_ms_p95":
            float(
                sorted(
                    latencies
                )[
                    min(
                        len(latencies) - 1,
                        int(
                            0.95
                            *
                            len(latencies)
                        )
                    )
                ]
            ),

        "latency_note":
            "Sequential Ollama /api/chat "
            "wall time after one warm-up. "
            "Model loading is excluded.",

        "predictions":
            details,
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def evaluate(
        model,
        split,
        output,
):
    data = rows(
        split
    )

    print(
        f"Model : {model}"
    )

    print(
        f"Split : {split}"
    )

    print(
        f"Cases : {len(data)}"
    )

    print()

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print(
        "Warming up Ollama model..."
    )

    warm_response, _ = (
        call_ollama(
            model,
            "Where is my order?"
        )
    )

    print(
        f"Warm-up response: "
        f"{warm_response}"
    )

    print()

    predictions = []

    raw_responses = []

    latencies = []

    # -----------------------------------------------------
    # Actual test cases
    # -----------------------------------------------------

    for index, row in enumerate(
            data,
            start=1
    ):

        raw_response, latency = (
            call_ollama(
                model,
                row["text"]
            )
        )

        prediction = (
            normalize_prediction(
                raw_response
            )
        )

        predictions.append(
            prediction
        )

        raw_responses.append(
            raw_response
        )

        latencies.append(
            latency
        )

        status = (
            "OK"
            if prediction
               ==
               row["label"]
            else
            "WRONG"
        )

        print(
            f"[{index:02d}/{len(data):02d}] "
            f"{status}"
        )

        print(
            f"  text      : "
            f"{row['text']}"
        )

        print(
            f"  expected  : "
            f"{row['label']}"
        )

        print(
            f"  predicted : "
            f"{prediction}"
        )

        if raw_response != prediction:
            print(
                f"  raw       : "
                f"{raw_response!r}"
            )

        print(
            f"  latency   : "
            f"{latency:.2f} ms"
        )

        print()

    report = calculate_report(

        data,
        predictions,
        raw_responses,
        latencies,
        model,
    )

    results_dir = (
            ROOT
            / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
            results_dir
            / output
    )

    output_path.write_text(

        json.dumps(
            report,
            indent=2
        ),

        encoding="utf-8"
    )

    print(
        "=" * 60
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(

        json.dumps(

            {

                "model":
                    report["model"],

                "n":
                    report["n"],

                "accuracy":
                    report["accuracy"],

                "macro_f1":
                    report["macro_f1"],

                "invalid_predictions":
                    report[
                        "invalid_predictions"
                    ],

                "latency_ms_median":
                    report[
                        "latency_ms_median"
                    ],

                "latency_ms_p95":
                    report[
                        "latency_ms_p95"
                    ],
            },

            indent=2
        )
    )

    print()

    print(
        f"Full report saved to:"
    )

    print(
        output_path
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(

        "model",

        help=(
            "Ollama model name, for example "
            "democart-smollm-before"
        )
    )

    parser.add_argument(

        "--split",

        choices=[
            "test",
            "challenge",
        ],

        default="test"
    )

    parser.add_argument(

        "--output",

        required=True,

        help=(
            "Result filename, for example "
            "ollama_before.json"
        )
    )

    args = parser.parse_args()

    evaluate(

        model=
        args.model,

        split=
        args.split,

        output=
        args.output,
    )


if __name__ == "__main__":
    main()
