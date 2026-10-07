# DemoCart — LoRA Fine-Tuning a Tiny SLM for Intent Classification

A practical proof of concept that compares **traditional machine learning** with a **LoRA fine-tuned SmolLM2-135M-Instruct** model for a narrow e-commerce intent-classification task, with a separate lightweight **RAG/policy retrieval** demo.

The purpose of this project is not to prove that an SLM always beats classical ML. It is to explore where each approach fits and to support a progressive architecture:

**Traditional ML → Fine-tuned SLM → Larger LLM / Human Review**

The design principle is simple:

> **Use the smallest, fastest, and cheapest component that can handle the request reliably. Escalate only when confidence is low.**

## Disclaimer

This is a **personal educational proof of concept** built independently using fictional data and a fictional e-commerce scenario. It is **not associated with, derived from, or representative of any employer, client, production system, proprietary implementation, or confidential dataset**.

## Key Results

Reference results from the experiment described in this repository:

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Base SmolLM2-135M-Instruct | **10.94%** | **4.62%** |
| LoRA fine-tuned SmolLM2-135M-Instruct | **81.94%** | **82.22%** |
| TF-IDF + Logistic Regression | **89.24%** | **89.12%** |

With confidence gating, the fine-tuned SLM accepted approximately **85.24%** of test requests and achieved approximately **89.21% accuracy on that accepted subset**.

> **Important:** 89.21% is **not** the SLM's overall accuracy. Overall held-out test accuracy was **81.94%**. The higher number applies only to predictions that passed the confidence gate.

The main finding is that traditional ML remained the strongest first-line option for this bounded classification task, while the fine-tuned SLM demonstrated how a small domain-adapted model can serve as a semantic fallback for harder or less predictable language.

## Dataset

The included dataset is already wired into the project:

- Train: **2,592** examples
- Validation: **288** examples
- Test: **576** examples
- 12 balanced intent labels

Intents:

`CANCEL_ITEM`, `CANCEL_ORDER`, `CHANGE_DELIVERY_ADDRESS`, `EXCHANGE_ITEM`, `PAYMENT_ISSUE`, `REFUND_REQUEST`, `REFUND_STATUS`, `RETURN_ITEM`, `RETURN_ORDER`, `RETURN_POLICY`, `SHIPPING_POLICY`, `TRACK_ORDER`.

Several deliberately similar intent pairs are kept separate, including:

- cancel one item vs. cancel the complete order
- return one item vs. return the complete order
- request a new refund vs. check an existing refund
- ask about shipping policy vs. track an existing shipment

This makes the task more realistic than classifying completely unrelated labels.

## Project Structure

```text
ecommerce-ai-smollm2-135m/
├─ config.py
├─ download_model.py
├─ experiment.py
├─ merge_adapter.py
├─ ollama_eval.py
├─ rag.py
├─ demo.py
├─ report.py
├─ requirements.txt
├─ data/
│  ├─ train.jsonl
│  ├─ validation.jsonl
│  ├─ test.jsonl
│  ├─ challenge.jsonl
│  ├─ policies.json
│  └─ rag_test.jsonl
├─ results/
│  ├─ baseline.json
│  ├─ base.json
│  ├─ tuned.json
│  └─ training.json
├─ ollama/
│  ├─ before/Modelfile
│  └─ after/Modelfile
└─ tests/
   └─ test_contracts.py
```

Reference experiment results are included under `results/`. Running the experiment again can regenerate or update these files.

Large generated artifacts such as downloaded base models, LoRA adapters, merged model files, virtual environments, and GGUF files should remain outside version control.

## 1. Create a Virtual Environment

Python **3.11 or 3.12** is recommended for the ML stack.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Validate the Dataset

```powershell
.\.venv\Scripts\python.exe experiment.py validate
```

Expected counts:

```json
{
  "train": 2592,
  "validation": 288,
  "test": 576,
  "challenge": 76,
  "rag_test": 4
}
```

Run contract tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Dataset Leakage Audit

`experiment.py validate` checks exact duplicates inside and across train/validation/test and reports TF-IDF nearest-neighbour similarity between every test example and the training set.

Exact duplicates fail validation. High similarity is reported as a diagnostic so the test set can be reviewed without artificially forcing a target score.

The validation/test examples are independently worded rather than being simple prefix/suffix rewrites of training sentences.

## 3. Run the Classical ML Baseline

This is **not** the SLM. It is a **TF-IDF + Logistic Regression** reference classifier.

```powershell
.\.venv\Scripts\python.exe experiment.py baseline
```

Output:

```text
results/baseline.json
```

The baseline is intentionally included because the goal is to compare approaches, not to force the SLM to win.

## 4. Download and Pin SmolLM2-135M

```powershell
.\.venv\Scripts\python.exe download_model.py
```

This saves the exact base model under `models/base/` so the before/after comparison starts from the same checkpoint.

## 5. Evaluate the Base Model Before Fine-Tuning

```powershell
.\.venv\Scripts\python.exe experiment.py base --threads 4 --device cpu
```

Output:

```text
results/base.json
```

The Hugging Face experiment scores all 12 allowed labels as candidate completions instead of relying on unconstrained free-text generation.

## 6. Fine-Tune with LoRA

Recommended first run on CPU:

```powershell
.\.venv\Scripts\python.exe experiment.py train --epochs 1 --threads 4 --batch-size 4 --device cpu --log-every 10
```

If memory pressure appears, reduce the batch size:

```powershell
.\.venv\Scripts\python.exe experiment.py train --epochs 1 --threads 4 --batch-size 2 --device cpu --log-every 10
```

The reference experiment intentionally used **one training epoch**. The objective was to measure how much useful domain behavior could be learned without repeatedly tuning until a headline-friendly score appeared.

Training performs **no validation during the epoch**. It does not call the dataset `validate()` function and does not evaluate `validation.jsonl` between epochs. The adapter from the final epoch is saved under `adapter/`.

### Training Log Output

During training the console reports:

- model and device settings
- number of training examples and batches per epoch
- epoch start/end
- batch/step number
- latest loss and running average loss
- elapsed time, processing rate, and estimated remaining time
- final adapter save path and total training time

`--log-every 10` prints progress every 10 batches. Use `--log-every 1` to print every batch.

## 7. Evaluate After Fine-Tuning

```powershell
.\.venv\Scripts\python.exe experiment.py tuned --threads 4 --device cpu
```

Or compare base and tuned in one command after training:

```powershell
.\.venv\Scripts\python.exe experiment.py evaluate --threads 4 --device cpu
```

Outputs:

```text
results/base.json
results/tuned.json
```

Create the short summary:

```powershell
.\.venv\Scripts\python.exe report.py
```

## 8. Ollama — Before Fine-Tuning

Pull the 135M model:

```powershell
ollama pull smollm2:135m
```

Create the constrained DemoCart model:

```powershell
ollama create democart-smollm-before -f .\ollama\before\Modelfile
```

Manual test:

```powershell
ollama run democart-smollm-before
```

Automated evaluation on the same 576-case test split:

```powershell
.\.venv\Scripts\python.exe ollama_eval.py democart-smollm-before --output ollama_before.json
```

## 9. Merge the LoRA Adapter and Create GGUF

`merge_adapter.py` expects `llama.cpp` beside the project folder, for example:

```text
C:\AI\
├─ ecommerce-ai-smollm2-135m\
└─ llama.cpp\
```

Then run:

```powershell
.\.venv\Scripts\python.exe merge_adapter.py
```

It creates:

```text
ollama\after\democart-smollm2-135m-finetuned.gguf
```

The generated GGUF is a large artifact and should not be committed to GitHub.

## 10. Ollama — After Fine-Tuning

```powershell
ollama create democart-smollm-after -f .\ollama\after\Modelfile
```

Then evaluate:

```powershell
.\.venv\Scripts\python.exe ollama_eval.py democart-smollm-after --output ollama_after.json
```

The before/after Ollama runs use the same 12 labels and the same held-out test split.

## 11. RAG Retrieval Demo

The RAG part is **independent from intent fine-tuning**. It retrieves fictional policy text from `data/policies.json`.

Run retrieval:

```powershell
.\.venv\Scripts\python.exe rag.py retrieve
```

For local generation with Ollama/Gemma:

```powershell
ollama pull gemma3:1b
.\.venv\Scripts\python.exe rag.py ask --question "What is the minimum spend for free shipping?" --model gemma3:1b
```

## Recommended Experiment Sequence

```powershell
.\.venv\Scripts\python.exe experiment.py validate
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe experiment.py baseline
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe experiment.py base --threads 4 --device cpu
.\.venv\Scripts\python.exe experiment.py train --epochs 1 --threads 4 --batch-size 4 --device cpu --log-every 10
.\.venv\Scripts\python.exe experiment.py tuned --threads 4 --device cpu
.\.venv\Scripts\python.exe report.py
```

## How to Interpret the Experiment

This project deliberately uses the small **SmolLM2-135M-Instruct** model so the POC remains practical on CPU.

The measured results should be interpreted as a comparison of architectural trade-offs:

- **Traditional ML** is fast, cheap, predictable, and strong for bounded classification.
- **Fine-tuned SLMs** can learn domain-specific behavior and provide a semantic fallback when lexical patterns become less reliable.
- **Larger LLMs** can be reserved for complex, ambiguous, multi-intent, or reasoning-heavy requests.

A production architecture should not blindly trust every prediction. Confidence gating allows a component to abstain and escalate uncertain requests to a stronger layer.

The experiment therefore supports a progressive routing design:

```text
Customer Request
      ↓
Traditional ML
      ↓ low confidence
Fine-tuned SLM
      ↓ still low confidence
Large LLM / Human Review
```

The goal is **not** to use the newest model everywhere.

The goal is to use the **smallest reliable component first** and escalate only when the problem actually requires more capability.

## Reproducibility Notes

- The test set is held out from training.
- Exact train/validation/test duplicates are rejected by validation.
- The base and fine-tuned SLM use the same starting checkpoint.
- The reference fine-tuning run used **one epoch**.
- Saved JSON results record the measured outcomes.
- Model weights, adapters, and GGUF artifacts are intentionally excluded from version control because they can be regenerated.

## Scope

This is a **proof of concept**, not a production implementation.

It does not claim that:

- SLMs always outperform traditional ML.
- a larger SLM would automatically achieve a specific accuracy target.
- confidence scores are perfectly calibrated probabilities.
- this architecture removes the need for testing, monitoring, human review, or domain-specific controls.

The repository is intended to make the experiment inspectable, reproducible, and easy to discuss as an engineering trade-off.
