"""CASI preprocessing and selectors retained from the original implementation."""
from __future__ import annotations

import csv

import hashlib

import json

import math

import os

import pickle

import random

import re

import statistics

import time

from collections import Counter, defaultdict

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize_scalar

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).resolve().parent

ROOT = HERE.parent

CASI_DATA = ROOT / "private/source_data/casi.txt"

RESULTS = HERE / "results"

FIGURES = HERE / "figures"

SEED = 20260722

MIN_CORE_GROUPS = 5

VAL_FRACTION = 0.15

TEST_FRACTION = 0.15

TARGET_SELECTIVE_ACCURACY = 0.95

NOISE_LABELS = {"NAME", "GENERAL ENGLISH", "UNSURED SENSE"}

@dataclass(frozen=True)
class Example:
    sf: str
    sense: str
    surface: str
    start: str
    end: str
    section: str
    snippet: str
    normalized_snippet: str
    duplicate_count: int = 1

    @property
    def text(self) -> str:
        return f"{self.section or ''} {self.snippet}".strip()

class ConstantSelector:
    """sklearn-like selector for abbreviations with one trainable sense."""

    def __init__(self, label: str):
        self.classes_ = np.array([label], dtype=object)
        self.label = label

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        return np.array([self.label] * len(texts), dtype=object)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        return np.ones((len(texts), 1), dtype=float)

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_snippet(text: str) -> str:
    return normalize_space(text).lower()

def normalize_sense(text: str) -> str:
    return normalize_space(text).lower()

def is_noise(sense: str) -> bool:
    stripped = sense.strip()
    return stripped in NOISE_LABELS or stripped.upper().startswith("MISTAKE:")

def stable_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return SEED + int(hashlib.sha256(payload).hexdigest()[:8], 16)

def load_and_group_casi() -> tuple[list[Example], list[dict], dict]:
    raw_rows: list[dict] = []
    with CASI_DATA.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 7 or is_noise(parts[1]):
                continue
            raw_rows.append(
                {
                    "sf": parts[0].strip(),
                    "sense": normalize_sense(parts[1]),
                    "surface": parts[2].strip(),
                    "start": parts[3].strip(),
                    "end": parts[4].strip(),
                    "section": parts[5].strip(),
                    "snippet": parts[6].strip(),
                    "normalized_snippet": normalize_snippet(parts[6]),
                }
            )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["sf"], row["normalized_snippet"])].append(row)

    examples: list[Example] = []
    conflicts: list[dict] = []
    duplicate_surplus = 0
    for (sf, normalized), rows in sorted(grouped.items()):
        senses = sorted({row["sense"] for row in rows})
        duplicate_surplus += len(rows) - 1
        if len(senses) != 1:
            conflicts.append(
                {
                    "sf": sf,
                    "normalized_snippet": normalized,
                    "senses": senses,
                    "n_rows": len(rows),
                }
            )
            continue
        row = rows[0]
        examples.append(Example(**row, duplicate_count=len(rows)))

    audit = {
        "valid_raw_rows": len(raw_rows),
        "unique_nonconflicting_contexts": len(examples),
        "duplicate_surplus_rows_removed": duplicate_surplus,
        "conflicting_duplicate_groups_removed": len(conflicts),
        "abbreviations": len({e.sf for e in examples}),
        "sense_classes": len({(e.sf, e.sense) for e in examples}),
    }
    return examples, conflicts, audit

def split_examples(examples: list[Example]) -> tuple[dict[str, list[Example]], list[Example], dict]:
    by_class: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        by_class[(example.sf, example.sense)].append(example)

    split = {"train": [], "validation": [], "test": []}
    rare_audit: list[Example] = []
    class_summary: dict[str, dict] = {}

    for (sf, sense), class_examples in sorted(by_class.items()):
        shuffled = list(class_examples)
        random.Random(stable_seed(sf, sense)).shuffle(shuffled)
        key = f"{sf}::{sense}"
        if len(shuffled) < MIN_CORE_GROUPS:
            rare_audit.extend(shuffled)
            class_summary[key] = {
                "n_unique": len(shuffled),
                "partition": "rare_open_set_audit",
            }
            continue

        n = len(shuffled)
        n_val = max(1, int(round(n * VAL_FRACTION)))
        n_test = max(1, int(round(n * TEST_FRACTION)))
        while n - n_val - n_test < 2:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        n_train = n - n_val - n_test
        split["train"].extend(shuffled[:n_train])
        split["validation"].extend(shuffled[n_train : n_train + n_val])
        split["test"].extend(shuffled[n_train + n_val :])
        class_summary[key] = {
            "n_unique": n,
            "partition": "core_closed_set",
            "train": n_train,
            "validation": n_val,
            "test": n_test,
        }

    for name in split:
        random.Random(stable_seed(name)).shuffle(split[name])
    random.Random(stable_seed("rare")).shuffle(rare_audit)
    return split, rare_audit, class_summary

def make_pipeline(kind: str, seed: int) -> Pipeline:
    if kind == "word_tfidf":
        features = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    elif kind == "char_tfidf":
        features = TfidfVectorizer(
            analyzer="char_wb", sublinear_tf=True, ngram_range=(3, 5), min_df=2, max_features=60000
        )
    elif kind == "combined_tfidf":
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        sublinear_tf=True,
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=60000,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    classifier = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline([("features", features), ("classifier", classifier)])

def train_family(kind: str, train: list[Example]) -> tuple[dict[str, object], dict]:
    by_sf: dict[str, list[Example]] = defaultdict(list)
    for example in train:
        by_sf[example.sf].append(example)

    models: dict[str, object] = {}
    per_sf_seconds: dict[str, float] = {}
    started = time.perf_counter()
    for sf, sf_examples in sorted(by_sf.items()):
        labels = sorted({e.sense for e in sf_examples})
        t0 = time.perf_counter()
        if len(labels) == 1:
            model: object = ConstantSelector(labels[0])
        else:
            model = make_pipeline(kind, stable_seed(kind, sf))
            texts = [e.text for e in sf_examples]
            targets = [e.sense for e in sf_examples]
            try:
                model.fit(texts, targets)
            except ValueError as exc:
                # Very small/vocabulary-poor edge cases can empty a min_df=2
                # vocabulary.  Retry with min_df=1, recording the exception.
                if "empty vocabulary" not in str(exc).lower():
                    raise
                if kind == "word_tfidf":
                    model = Pipeline(
                        [
                            ("features", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)),
                            (
                                "classifier",
                                LogisticRegression(
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=stable_seed(kind, sf),
                                ),
                            ),
                        ]
                    )
                    model.fit(texts, targets)
                else:
                    raise
        per_sf_seconds[sf] = time.perf_counter() - t0
        models[sf] = model

    serialized_bytes = len(pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL))
    return models, {
        "training_seconds": time.perf_counter() - started,
        "per_abbreviation_training_seconds": per_sf_seconds,
        "serialized_bytes": serialized_bytes,
    }

def predict_records(models: dict[str, object], examples: list[Example]) -> list[dict]:
    records: list[dict] = []
    by_sf: dict[str, list[tuple[int, Example]]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_sf[example.sf].append((index, example))

    ordered: list[dict | None] = [None] * len(examples)
    for sf, pairs in by_sf.items():
        model = models[sf]
        texts = [example.text for _, example in pairs]
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        classes = [str(x) for x in model.classes_]
        for (index, example), prediction, probs in zip(pairs, predictions, probabilities):
            gold_index = classes.index(example.sense)
            ordered[index] = {
                "sf": sf,
                "gold": example.sense,
                "prediction": str(prediction),
                "correct": bool(prediction == example.sense),
                "classes": classes,
                "probabilities": [float(x) for x in probs],
                "gold_index": gold_index,
                "confidence": float(np.max(probs)),
            }
    return [record for record in ordered if record is not None]

def classification_metrics(records: list[dict]) -> dict:
    gold = [f"{r['sf']}::{r['gold']}" for r in records]
    pred = [f"{r['sf']}::{r['prediction']}" for r in records]
    per_sf: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        per_sf[record["sf"]].append(record["correct"])
    return {
        "n": len(records),
        "accuracy": accuracy_score(gold, pred),
        "micro_f1": f1_score(gold, pred, average="micro", zero_division=0),
        "macro_f1_sense_class": f1_score(gold, pred, average="macro", zero_division=0),
        "macro_abbreviation_accuracy": float(
            np.mean([np.mean(values) for values in per_sf.values()])
        ),
        "per_abbreviation_accuracy": {
            sf: float(np.mean(values)) for sf, values in sorted(per_sf.items())
        },
    }

def majority_predictions(train: list[Example], examples: list[Example]) -> list[dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for example in train:
        counts[example.sf][example.sense] += 1
    majority = {sf: counter.most_common(1)[0][0] for sf, counter in counts.items()}
    records = []
    for example in examples:
        prediction = majority[example.sf]
        records.append(
            {
                "sf": example.sf,
                "gold": example.sense,
                "prediction": prediction,
                "correct": prediction == example.sense,
            }
        )
    return records

def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    logits -= np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()

def fit_temperature(validation_records: list[dict]) -> float:
    def nll(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        losses = []
        for record in validation_records:
            scaled = temperature_scale(np.array(record["probabilities"]), temperature)
            losses.append(-math.log(max(float(scaled[record["gold_index"]]), 1e-12)))
        return float(np.mean(losses))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))

def apply_temperature(records: list[dict], temperature: float) -> list[dict]:
    scaled_records = []
    for record in records:
        updated = dict(record)
        scaled = temperature_scale(np.array(record["probabilities"]), temperature)
        updated["scaled_probabilities"] = [float(x) for x in scaled]
        updated["scaled_confidence"] = float(np.max(scaled))
        scaled_records.append(updated)
    return scaled_records

def calibration_metrics(records: list[dict], confidence_key: str, probability_key: str) -> dict:
    confidences = np.array([r[confidence_key] for r in records], dtype=float)
    correctness = np.array([r["correct"] for r in records], dtype=float)
    brier = []
    for record in records:
        probs = np.array(record[probability_key], dtype=float)
        truth = np.zeros_like(probs)
        truth[record["gold_index"]] = 1.0
        brier.append(float(np.sum((probs - truth) ** 2)))

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    reliability = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        n = int(mask.sum())
        if not n:
            reliability.append(
                {"lower": lower, "upper": upper, "n": 0, "mean_confidence": None, "accuracy": None}
            )
            continue
        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correctness[mask].mean())
        ece += (n / len(records)) * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "n": n,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "brier_multiclass": float(np.mean(brier)),
        "ece_10_bin": float(ece),
        "reliability": reliability,
    }

def risk_coverage_curve(records: list[dict], confidence_key: str) -> list[dict]:
    ordered = sorted(records, key=lambda r: r[confidence_key], reverse=True)
    curve = []
    for coverage in np.linspace(0.05, 1.0, 20):
        retained = max(1, int(math.ceil(len(ordered) * float(coverage))))
        selected = ordered[:retained]
        accuracy = float(np.mean([r["correct"] for r in selected]))
        curve.append(
            {
                "coverage": retained / len(ordered),
                "retained": retained,
                "accuracy": accuracy,
                "risk": 1.0 - accuracy,
            }
        )
    return curve

def choose_validation_threshold(records: list[dict], confidence_key: str) -> dict:
    ordered = sorted(records, key=lambda r: r[confidence_key], reverse=True)
    feasible = []
    for retained in range(1, len(ordered) + 1):
        accuracy = float(np.mean([r["correct"] for r in ordered[:retained]]))
        if accuracy >= TARGET_SELECTIVE_ACCURACY:
            feasible.append((retained, accuracy, ordered[retained - 1][confidence_key]))
    if feasible:
        retained, accuracy, threshold = max(feasible, key=lambda x: x[0])
        target_met = True
    else:
        candidates = []
        for retained in range(1, len(ordered) + 1):
            accuracy = float(np.mean([r["correct"] for r in ordered[:retained]]))
            candidates.append((accuracy, retained, ordered[retained - 1][confidence_key]))
        accuracy, retained, threshold = max(candidates, key=lambda x: (x[0], x[1]))
        target_met = False
    return {
        "threshold": float(threshold),
        "validation_retained": int(retained),
        "validation_coverage": retained / len(ordered),
        "validation_accuracy": accuracy,
        "target_accuracy": TARGET_SELECTIVE_ACCURACY,
        "target_met": target_met,
    }

def evaluate_threshold(records: list[dict], confidence_key: str, policy: dict) -> dict:
    retained = [r for r in records if r[confidence_key] >= policy["threshold"]]
    return {
        **policy,
        "test_retained": len(retained),
        "test_coverage": len(retained) / len(records),
        "test_accuracy": float(np.mean([r["correct"] for r in retained])) if retained else None,
    }

def prior_ranked_candidate_metrics(train: list[Example], test: list[Example]) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for example in train:
        counts[example.sf][example.sense] += 1
    ranks = []
    lookup_latencies_ns = []
    pool_sizes = []
    for example in test:
        t0 = time.perf_counter_ns()
        ranked = [sense for sense, _ in counts[example.sf].most_common()]
        lookup_latencies_ns.append(time.perf_counter_ns() - t0)
        pool_sizes.append(len(ranked))
        ranks.append(ranked.index(example.sense) + 1 if example.sense in ranked else None)

    metrics = {}
    for k in (1, 3, 5, 10):
        metrics[f"recall_at_{k}"] = float(np.mean([rank is not None and rank <= k for rank in ranks]))
    metrics.update(
        {
            "recall_at_all": float(np.mean([rank is not None for rank in ranks])),
            "mrr": float(np.mean([1.0 / rank if rank else 0.0 for rank in ranks])),
            "mean_candidate_pool_size": float(np.mean(pool_sizes)),
            "median_candidate_pool_size": float(np.median(pool_sizes)),
            "max_candidate_pool_size": int(max(pool_sizes)),
            "lookup_latency_microseconds_p50": float(np.percentile(lookup_latencies_ns, 50) / 1000),
            "lookup_latency_microseconds_p95": float(np.percentile(lookup_latencies_ns, 95) / 1000),
        }
    )
    return metrics

def benchmark_predictions(models: dict[str, object], examples: list[Example], repeats: int = 8) -> dict:
    by_sf: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_sf[example.sf].append(example)
    # Warm up caches and vectorizers.
    for sf, sf_examples in by_sf.items():
        models[sf].predict([e.text for e in sf_examples[: min(3, len(sf_examples))]])

    batch_seconds = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for sf, sf_examples in by_sf.items():
            models[sf].predict([e.text for e in sf_examples])
        batch_seconds.append(time.perf_counter() - t0)

    per_item_ns = []
    for sf, sf_examples in by_sf.items():
        model = models[sf]
        for example in sf_examples:
            t0 = time.perf_counter_ns()
            model.predict([example.text])
            per_item_ns.append(time.perf_counter_ns() - t0)

    median_batch = statistics.median(batch_seconds)
    return {
        "n_test": len(examples),
        "batch_repeats": repeats,
        "batch_seconds": batch_seconds,
        "batch_seconds_median": median_batch,
        "throughput_items_per_second": len(examples) / median_batch,
        "per_item_latency_ms_p50": float(np.percentile(per_item_ns, 50) / 1e6),
        "per_item_latency_ms_p95": float(np.percentile(per_item_ns, 95) / 1e6),
    }
