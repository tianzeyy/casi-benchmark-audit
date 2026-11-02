"""Run a fully local frozen BioClinicalBERT baseline on the fixed CASI split.

The encoder is not fine-tuned. Target-centred contexts are encoded once with
BioClinicalBERT, mean pooled, and classified by one balanced logistic
regression model per abbreviation. The script records predictive performance,
runtime, memory, storage, and item-level predictions for paired comparisons.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL_SNAPSHOT = Path(os.environ.get("CLINICALBERT_MODEL_PATH", str(HERE.parent / "private/models/Bio_ClinicalBERT")))
SEED = 20260722
MAX_LENGTH = 128
BATCH_SIZE = 32
CONTEXT_CHARS_EACH_SIDE = 260


class ConstantSelector:
    def __init__(self, label: str):
        self.label = label

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.repeat(self.label, len(matrix))


def load_split(name: str) -> list[dict]:
    return json.loads((RESULTS / f"casi_{name}.json").read_text(encoding="utf-8"))


def target_centered_text(row: dict) -> str:
    snippet = row["snippet"]
    try:
        start = int(row["start"])
        end = int(row["end"])
    except (TypeError, ValueError):
        start, end = 0, min(len(snippet), len(row.get("surface", "")))
    start = min(max(start, 0), len(snippet))
    end = min(max(end, start), len(snippet))
    left = max(0, start - CONTEXT_CHARS_EACH_SIDE)
    right = min(len(snippet), end + CONTEXT_CHARS_EACH_SIDE)
    context = snippet[left:right]
    section = (row.get("section") or "").strip()
    return f"{section} [SEP] {context}" if section else context


def encode(
    rows: list[dict],
    tokenizer,
    encoder,
    *,
    label: str,
) -> tuple[np.ndarray, float]:
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[offset : offset + BATCH_SIZE]
            encoded = tokenizer(
                [target_centered_text(row) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            output = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            completed = min(offset + BATCH_SIZE, len(rows))
            if completed == len(rows) or completed % 1024 == 0:
                print(f"{label}: encoded {completed}/{len(rows)}", flush=True)
    return np.concatenate(vectors, axis=0), time.perf_counter() - started


def train_selectors(train_rows: list[dict], embeddings: np.ndarray) -> tuple[dict, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        grouped[row["sf"]].append(index)

    selectors = {}
    started = time.perf_counter()
    for sf, indices in sorted(grouped.items()):
        labels = np.asarray([train_rows[index]["sense"] for index in indices])
        unique = np.unique(labels)
        if len(unique) == 1:
            selectors[sf] = ConstantSelector(str(unique[0]))
        else:
            classifier = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2500,
                random_state=SEED,
                solver="lbfgs",
            )
            classifier.fit(embeddings[indices], labels)
            selectors[sf] = classifier
    return selectors, time.perf_counter() - started


def predict(selectors: dict, rows: list[dict], embeddings: np.ndarray) -> tuple[list[str], float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["sf"]].append(index)
    predictions: list[str | None] = [None] * len(rows)
    started = time.perf_counter()
    for sf, indices in grouped.items():
        output = selectors[sf].predict(embeddings[indices])
        for index, predicted in zip(indices, output):
            predictions[index] = str(predicted)
    assert all(prediction is not None for prediction in predictions)
    return [str(prediction) for prediction in predictions], time.perf_counter() - started


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [centre - radius, centre + radius]


def paired_mcnemar(gold: list[str], a: list[str], b: list[str]) -> dict:
    a_only = sum(x == y and x != z for x, y, z in zip(gold, a, b))
    b_only = sum(x == z and x != y for x, y, z in zip(gold, a, b))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "clinicalbert_only_correct": a_only,
        "word_tfidf_only_correct": b_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def max_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def main() -> None:
    if not MODEL_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"BioClinicalBERT is not cached at {MODEL_SNAPSHOT}. "
            "Set CLINICALBERT_MODEL_PATH to a complete local snapshot."
        )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    train_rows = load_split("train")
    test_rows = load_split("test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SNAPSHOT, local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL_SNAPSHOT, local_files_only=True)
    encoder.eval()

    # Warm the model before the timed full-split measurements.
    warm = tokenizer(
        [target_centered_text(row) for row in train_rows[:BATCH_SIZE]],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.inference_mode():
        encoder(**warm)

    train_embeddings, train_encoding_seconds = encode(
        train_rows, tokenizer, encoder, label="train"
    )
    selectors, classifier_training_seconds = train_selectors(train_rows, train_embeddings)
    test_embeddings, test_encoding_seconds = encode(test_rows, tokenizer, encoder, label="test")
    predictions, classifier_prediction_seconds = predict(selectors, test_rows, test_embeddings)

    gold = [row["sense"] for row in test_rows]
    labels = [f"{row['sf']}::{row['sense']}" for row in test_rows]
    predicted_labels = [
        f"{row['sf']}::{prediction}" for row, prediction in zip(test_rows, predictions)
    ]
    correct = [prediction == target for prediction, target in zip(predictions, gold)]
    per_sf = defaultdict(list)
    for row, is_correct in zip(test_rows, correct):
        per_sf[row["sf"]].append(is_correct)

    tfidf_records = json.loads(
        (RESULTS / "word_tfidf_test_predictions.json").read_text(encoding="utf-8")
    )
    tfidf_gold = [row["gold"] for row in tfidf_records]
    if tfidf_gold != gold:
        raise ValueError("ClinicalBERT test order does not match the fixed word-TFIDF predictions")
    tfidf_predictions = [row["prediction"] for row in tfidf_records]

    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in encoder.parameters()
    )
    classifier_bytes = len(pickle.dumps(selectors, protocol=pickle.HIGHEST_PROTOCOL))
    inference_seconds = test_encoding_seconds + classifier_prediction_seconds
    results = {
        "experiment_status": "complete",
        "model": "emilyalsentzer/Bio_ClinicalBERT",
        "model_revision": MODEL_SNAPSHOT.name,
        "access": "local cached weights; no hosted API",
        "design": "frozen encoder, attention-mask mean pooling, per-abbreviation balanced logistic regression",
        "input": "section plus target-centred context",
        "max_length_tokens": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "device": "cpu",
        "seed": SEED,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "accuracy": float(accuracy_score(gold, predictions)),
        "accuracy_95_ci_wilson": wilson_interval(sum(correct), len(correct)),
        "macro_f1_over_sf_sense": float(
            f1_score(labels, predicted_labels, average="macro", zero_division=0)
        ),
        "macro_abbreviation_accuracy": float(
            np.mean([np.mean(values) for values in per_sf.values()])
        ),
        "training_seconds": train_encoding_seconds + classifier_training_seconds,
        "train_encoding_seconds": train_encoding_seconds,
        "classifier_training_seconds": classifier_training_seconds,
        "test_inference_seconds": inference_seconds,
        "test_items_per_second": len(test_rows) / inference_seconds,
        "test_ms_per_item": 1000 * inference_seconds / len(test_rows),
        "model_parameter_bytes": model_parameter_bytes,
        "classifier_serialized_bytes": classifier_bytes,
        "total_model_bytes": model_parameter_bytes + classifier_bytes,
        "peak_process_rss_mb": max_rss_mb(),
        "mcnemar_vs_word_tfidf": paired_mcnemar(
            gold, predictions, tfidf_predictions
        ),
        "per_abbreviation": {
            sf: {
                "n": len(values),
                "accuracy": float(np.mean(values)),
            }
            for sf, values in sorted(per_sf.items())
        },
        "limitations": [
            "The transformer encoder was frozen rather than fine-tuned.",
            "Runtime was measured on CPU in one local environment and is hardware dependent.",
            "The linear probe uses the same fixed CASI train/test partition as the primary models.",
        ],
    }

    output_records = [
        {
            "sf": row["sf"],
            "gold": target,
            "prediction": prediction,
            "correct": prediction == target,
        }
        for row, target, prediction in zip(test_rows, gold, predictions)
    ]
    (RESULTS / "clinicalbert_test_predictions.json").write_text(
        json.dumps(output_records, indent=2), encoding="utf-8"
    )
    (RESULTS / "clinicalbert_baseline.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
