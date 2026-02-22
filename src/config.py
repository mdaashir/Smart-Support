"""Centralised configuration — single source of truth for every milestone."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "saved_models"
EVAL_DIR = ROOT_DIR / "evaluation" / "artifacts"

# ── Real dataset (HuggingFace open-source) ───────────────────────────────
HF_DATASET_ID = "Tobi-Bueck/customer-support-tickets"
HF_DATASET_SPLIT = "train"
CACHED_CSV = DATA_DIR / "customer_support_tickets.csv"

# ── Label mapping (queue → category) ────────────────────────────────────
#    Maps the 20+ queue values from the real dataset into 4 actionable
#    routing categories.
QUEUE_TO_CATEGORY: dict[str, str] = {
    # Billing / Payments
    "billing and payments": "Billing",
    "returns and exchanges": "Billing",
    "sales and pre-sales": "Billing",
    # Technical
    "technical support": "Technical",
    "product support": "Technical",
    "it support": "Technical",
    "service outages and maintenance": "Technical",
    # HR / Admin
    "human resources": "HR",
    # General / Customer-facing
    "customer service": "General",
    "general inquiry": "General",
}
DEFAULT_LABEL = "General"  # anything not in the map

CATEGORIES = ["Billing", "Technical", "HR", "General"]

# ── Priority mapping (real dataset values → numeric) ─────────────────────
PRIORITY_MAP: dict[str, int] = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "very_low": 5,
}

# ── Urgency keywords (shared by all milestones) ─────────────────────────
URGENT_KEYWORDS: list[str] = [
    "urgent", "asap", "immediately", "critical",
    "right now", "broken", "not working",
    "failed", "down", "error", "outage", "breach",
    "dringend", "sofort", "kritisch",  # German urgency words
]

# ── Synthetic data defaults ──────────────────────────────────────────────
SYNTHETIC_SAMPLES_PER_CLASS = 6_000
RANDOM_STATE = 42

# ── Model hyper-parameters ───────────────────────────────────────────────
TFIDF_LOGREG = {
    "ngram_range": (1, 2),
    "max_features": 15_000,
    "stop_words": "english",
    "max_iter": 1_000,
}

TFIDF_SVC = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "min_df": 5,
    "max_features": 20_000,
}

DISTILBERT = {
    "model_name": "distilbert-base-multilingual-cased",
    "max_length": 128,
    "batch_size": 16,
    "max_iter": 1_000,
}
