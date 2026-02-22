"""Centralised configuration — single source of truth for every milestone."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "saved_models"
EVAL_DIR = ROOT_DIR / "evaluation" / "artifacts"

# ── Label mapping (queue → category) ────────────────────────────────────
LABEL_MAP: dict[str, str] = {
    "billing": "Billing",
    "payment": "Billing",
    "legal": "Legal",
    "compliance": "Legal",
}
DEFAULT_LABEL = "Technical"

# ── Urgency keywords (shared by all milestones) ─────────────────────────
URGENT_KEYWORDS: list[str] = [
    "urgent", "asap", "immediately", "critical",
    "right now", "broken", "not working",
    "failed", "down", "error", "outage", "breach",
]

# ── Synthetic data defaults ──────────────────────────────────────────────
SYNTHETIC_SAMPLES_PER_CLASS = 6_000
RANDOM_STATE = 42

# ── Model hyper-parameters ───────────────────────────────────────────────
TFIDF_LOGREG = {
    "ngram_range": (1, 2),
    "max_features": 15_000,
    "stop_words": "english",
    "max_iter": 500,
}

TFIDF_SVC = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "min_df": 5,
    "max_features": 20_000,
}

DISTILBERT = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "max_iter": 1_000,
}
