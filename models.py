"""
models.py — Milestone 1 ML module
Trains a LinearSVC ticket router on first import and caches the artefact
to `model_cache/m1.joblib`.  Re-uses the cache on subsequent starts.

Exported surface:
    route_ticket(subject, body)  → dict with category + urgency_level
    detect_urgency(text)         → int  1 = HIGH, 5 = NORMAL
    CATEGORIES                   → list of label strings
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_ROOT      = Path(__file__).parent
DATA_PATH  = _ROOT / "dataset" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
CACHE_DIR  = _ROOT / "model_cache"
CACHE_FILE = CACHE_DIR / "m1.joblib"

CATEGORIES = ["Billing", "Legal", "Technical"]

# ──────────────────────────────────────────────
# Label mapping
# ──────────────────────────────────────────────
_BILLING_QUEUES = {"billing and payments", "returns and exchanges"}
_LEGAL_QUEUES   = {"human resources"}


def _map_queue(queue: str) -> str:
    q = str(queue).lower().strip()
    if q in _BILLING_QUEUES:
        return "Billing"
    if q in _LEGAL_QUEUES:
        return "Legal"
    return "Technical"


# ──────────────────────────────────────────────
# Urgency heuristic (Milestone-1 regex)
# heap priority: 1 = HIGH (popped first), 5 = NORMAL
# ──────────────────────────────────────────────
_URGENT_RE = re.compile(
    r"\b(asap|urgent|immediately|critical|down|outage|breach|broken|failed|error)\b",
    re.IGNORECASE,
)


def detect_urgency(text: str) -> int:
    """Return 1 (HIGH) or 5 (NORMAL) based on urgent keywords."""
    return 1 if _URGENT_RE.search(text) else 5


# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────
def _train() -> tuple:
    """Train TF-IDF + LinearSVC on the real dataset."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    logger.info("Loading dataset from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.lower()
    df["label"] = df["queue"].apply(_map_queue)

    logger.info("Label distribution: %s", df["label"].value_counts().to_dict())

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=20_000,
    )
    clf = LinearSVC(class_weight="balanced")

    X = vec.fit_transform(df["text"])
    clf.fit(X, df["label"])

    logger.info("Training complete.")
    return vec, clf


def _load_or_train() -> tuple:
    """Load cached model or train and save it."""
    import joblib

    if CACHE_FILE.exists():
        logger.info("Loading cached model from %s", CACHE_FILE)
        return joblib.load(CACHE_FILE)

    vec, clf = _train()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((vec, clf), CACHE_FILE)
    logger.info("Model cached at %s", CACHE_FILE)
    return vec, clf


# ──────────────────────────────────────────────
# Module-level lazy state
# ──────────────────────────────────────────────
_vectorizer = None
_classifier = None


def _ensure_loaded() -> None:
    global _vectorizer, _classifier
    if _vectorizer is None:
        _vectorizer, _classifier = _load_or_train()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def route_ticket(subject: str, body: str) -> dict:
    """
    Classify a ticket and assess urgency.

    Returns:
        {
            "category":      "Billing" | "Legal" | "Technical",
            "urgency_level": 1 (HIGH) | 5 (NORMAL),
        }
    """
    _ensure_loaded()
    text = (subject + " " + body).lower()
    vec  = _vectorizer.transform([text])
    return {
        "category":      _classifier.predict(vec)[0],
        "urgency_level": detect_urgency(text),
    }


# ──────────────────────────────────────────────
# CLI smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        ("Invoice issue",    "Charged twice for last month subscription"),
        ("Server down",      "Production API returning 500 errors ASAP"),
        ("HR policy query",  "Need clarification on employment contract terms"),
    ]
    for subject, body in tests:
        result = route_ticket(subject, body)
        print(f"[{result['category']:10s} | urgency={result['urgency_level']}]  "
              f"{subject!r}")
