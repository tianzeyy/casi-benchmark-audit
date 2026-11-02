"""CASI preprocessing and selectors retained from the original implementation."""
from __future__ import annotations

import hashlib

import json

import math

import random

import re

import sys

from collections import Counter, defaultdict

from dataclasses import dataclass

from pathlib import Path

import numpy as np

import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent

ROOT = HERE.parent

CASI_DATA = ROOT / "private/source_data/casi.txt"

RESULTS = HERE / "results"

SEED = 20260722

MIN_CORE_GROUPS = 5

VAL_FRACTION = 0.15

TEST_FRACTION = 0.15

NOISE_LABELS = {"NAME", "GENERAL ENGLISH", "UNSURED SENSE"}

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

@dataclass(frozen=True)
class Example:
    sf: str
    sense: str
    section: str
    snippet: str
    normalized_snippet: str

    @property
    def text(self) -> str:
        return f"{self.section or ''} {self.snippet}".strip()

def load_raw_rows() -> list[Example]:
    """Every valid CASI row, duplicates retained."""
    rows: list[Example] = []
    with CASI_DATA.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 7 or is_noise(parts[1]):
                continue
            rows.append(
                Example(
                    sf=parts[0].strip(),
                    sense=normalize_sense(parts[1]),
                    section=parts[5].strip(),
                    snippet=parts[6].strip(),
                    normalized_snippet=normalize_snippet(parts[6]),
                )
            )
    return rows

def collapse_duplicates(rows: list[Example]) -> tuple[list[Example], int, int]:
    """Protocol step: group exact normalized contexts within abbreviation."""
    grouped: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for row in rows:
        grouped[(row.sf, row.normalized_snippet)].append(row)

    examples: list[Example] = []
    surplus = 0
    conflicts = 0
    for (_sf, _norm), group in sorted(grouped.items(), key=lambda kv: kv[0]):
        surplus += len(group) - 1
        senses = {row.sense for row in group}
        if len(senses) != 1:
            conflicts += 1
            continue
        examples.append(group[0])
    return examples, surplus, conflicts

def partition(examples: list[Example], min_count: int) -> dict[str, list[Example]]:
    """Stratified 70/15/15 within each (abbreviation, sense) class."""
    by_class: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        by_class[(example.sf, example.sense)].append(example)

    split: dict[str, list[Example]] = {"train": [], "validation": [], "test": []}
    for (sf, sense), class_examples in sorted(by_class.items()):
        shuffled = list(class_examples)
        random.Random(stable_seed(sf, sense)).shuffle(shuffled)
        if len(shuffled) < min_count:
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
    return split

def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500, class_weight="balanced", random_state=seed, solver="lbfgs"
                ),
            ),
        ]
    )

class ConstantSelector:
    def __init__(self, label: str):
        self.label = label

    def predict(self, texts):
        return np.array([self.label] * len(list(texts)), dtype=object)

def train_and_predict(train: list[Example], test: list[Example]) -> list[dict]:
    by_sf: dict[str, list[Example]] = defaultdict(list)
    for example in train:
        by_sf[example.sf].append(example)

    models: dict[str, object] = {}
    for sf, sf_examples in sorted(by_sf.items()):
        labels = sorted({e.sense for e in sf_examples})
        if len(labels) == 1:
            models[sf] = ConstantSelector(labels[0])
            continue
        model = make_model(stable_seed("word_tfidf", sf))
        texts = [e.text for e in sf_examples]
        targets = [e.sense for e in sf_examples]
        try:
            model.fit(texts, targets)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            model = Pipeline(
                [
                    ("features", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=stable_seed("word_tfidf", sf),
                        ),
                    ),
                ]
            )
            model.fit(texts, targets)
        models[sf] = model

    # Contexts seen verbatim during training, keyed within abbreviation.
    train_keys = {(e.sf, e.normalized_snippet) for e in train}

    by_sf_test: dict[str, list[tuple[int, Example]]] = defaultdict(list)
    for index, example in enumerate(test):
        by_sf_test[example.sf].append((index, example))

    records: list[dict | None] = [None] * len(test)
    for sf, pairs in by_sf_test.items():
        model = models.get(sf)
        if model is None:
            # Abbreviation absent from training: unrecoverable, scored wrong.
            for index, example in pairs:
                records[index] = {
                    "sf": sf,
                    "gold": example.sense,
                    "prediction": None,
                    "correct": False,
                    "leaked": (example.sf, example.normalized_snippet) in train_keys,
                }
            continue
        predictions = model.predict([example.text for _, example in pairs])
        for (index, example), prediction in zip(pairs, predictions):
            records[index] = {
                "sf": sf,
                "gold": example.sense,
                "prediction": str(prediction),
                "correct": bool(prediction == example.sense),
                "leaked": (example.sf, example.normalized_snippet) in train_keys,
            }
    return [r for r in records if r is not None]

def wilson_ci(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (center - margin, center + margin)

def summarize(records: list[dict], label: str) -> dict:
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    lo, hi = wilson_ci(correct, total)
    gold = [r["gold"] for r in records]
    pred = [r["prediction"] if r["prediction"] is not None else "<none>" for r in records]
    return {
        "arm": label,
        "n_test": total,
        "n_correct": correct,
        "accuracy": correct / total,
        "ci_low": lo,
        "ci_high": hi,
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }
