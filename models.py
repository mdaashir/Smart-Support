"""
models.py — Milestone 1 ML module
Trains a LinearSVC ticket router on first import and caches the artefact
to `model_cache/m1.joblib`.  Re-uses the cache on subsequent starts.

Exported surface:
    route_ticket(subject, body)  → dict with category + urgency_level
    detect_urgency(text)         → int  1 = HIGH, 5 = NORMAL
    get_metrics()                → dict | None  (cached training metrics)
    CATEGORIES                   → list of label strings
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_ROOT        = Path(__file__).parent
DATA_PATH    = _ROOT / "dataset" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
CACHE_DIR    = _ROOT / "model_cache"
CACHE_FILE   = CACHE_DIR / "m1.joblib"
METRICS_FILE = CACHE_DIR / "m1_metrics.json"

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
    """Train TF-IDF + LinearSVC on the real dataset and save metrics."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC

    logger.info("Loading dataset from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.lower()
    df["label"] = df["queue"].apply(_map_queue)

    logger.info("Label distribution: %s", df["label"].value_counts().to_dict())

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
    )
    clf = LinearSVC(class_weight="balanced", max_iter=2000, C=0.5)

    X = vec.fit_transform(df["text"])

    # Hold out 15% for evaluation metrics
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["label"], test_size=0.15, stratify=df["label"], random_state=42
    )
    clf.fit(X_tr, y_tr)

    y_pred  = clf.predict(X_te)
    acc     = accuracy_score(y_te, y_pred)
    report  = classification_report(y_te, y_pred, output_dict=True)
    logger.info(
        "M1 evaluation → acc=%.4f\n%s",
        acc,
        classification_report(y_te, y_pred)
    )

    # Refit on full dataset for production
    clf.fit(X, df["label"])

    # Persist metrics
    metrics = {
        "train_size":       len(df),
        "eval_size":        len(y_te),
        "accuracy":         round(acc, 4),
        "classification_report": report,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    logger.info("M1 metrics saved to %s", METRICS_FILE)

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


def get_metrics() -> dict | None:
    """Return normalized M1 training/eval metrics, or None if not yet trained."""
    if METRICS_FILE.exists():
        try:
            raw = json.loads(METRICS_FILE.read_text())
            # Flatten classification_report to top level for easy API consumption
            report = raw.get("classification_report", {})
            return {
                "accuracy":      raw.get("accuracy"),
                "weighted avg":  report.get("weighted avg"),
                "macro avg":     report.get("macro avg"),
                "Billing":       report.get("Billing"),
                "Legal":         report.get("Legal"),
                "Technical":     report.get("Technical"),
                "train_size":    raw.get("train_size"),
                "eval_size":     raw.get("eval_size"),
            }
        except Exception:
            return None
    return None


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
