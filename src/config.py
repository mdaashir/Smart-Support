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

# ── Label mapping (queue → 3 categories per PDF: Billing, Technical, Legal)
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
    # Legal (HR → employment law/contracts, service → TOS disputes)
    "human resources": "Legal",
    "customer service": "Legal",
    "general inquiry": "Legal",
}
DEFAULT_LABEL = "Legal"  # anything not in the map

CATEGORIES = ["Billing", "Technical", "Legal"]

# ── Priority mapping (real dataset values → numeric) ─────────────────────
PRIORITY_MAP: dict[str, int] = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "very_low": 5,
}

# ── Urgency keywords (regex heuristic — Milestone 1 & fallback) ─────────
URGENT_KEYWORDS: list[str] = [
    "urgent", "asap", "immediately", "critical",
    "right now", "broken", "not working",
    "failed", "down", "error", "outage", "breach",
    "dringend", "sofort", "kritisch",  # German urgency words
]

# ── Milestone 2: Urgency regression ─────────────────────────────────────
URGENCY_WEBHOOK_THRESHOLD = 0.8   # S > 0.8 triggers Slack/Discord webhook
WEBHOOK_URL = "https://hooks.slack.com/services/MOCK/WEBHOOK"  # mock

# ── Milestone 3: Semantic deduplication ──────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.9
DEDUP_COUNT_THRESHOLD = 10        # tickets within window
DEDUP_WINDOW_SECONDS = 300        # 5 minutes
SENTENCE_MODEL = "all-MiniLM-L6-v2"

# ── Milestone 3: Circuit breaker ─────────────────────────────────────────
CIRCUIT_BREAKER_LATENCY_MS = 500  # failover if transformer > 500 ms

# ── Milestone 3: Skill-based routing ─────────────────────────────────────
AGENT_REGISTRY = {
    "Agent_A": {"skills": {"Technical": 0.9, "Billing": 0.1, "Legal": 0.0}, "capacity": 5},
    "Agent_B": {"skills": {"Billing": 0.8, "Legal": 0.2, "Technical": 0.0}, "capacity": 5},
    "Agent_C": {"skills": {"Legal": 0.7, "Technical": 0.2, "Billing": 0.1}, "capacity": 5},
    "Agent_D": {"skills": {"Technical": 0.5, "Billing": 0.3, "Legal": 0.2}, "capacity": 5},
}

# ── Training & anti-overfit ──────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS = 5  # stratified k-fold cross-validation

# ── Model hyper-parameters ───────────────────────────────────────────────
TFIDF_LOGREG = {
    "ngram_range": (1, 2),
    "max_features": 15_000,
    "stop_words": "english",
    "max_iter": 1_000,
    "C": 1.0,
}

TFIDF_SVC = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 5,
    "max_features": 20_000,
    "C": 1.0,
}

DISTILBERT = {
    "model_name": "distilbert-base-multilingual-cased",
    "max_length": 128,
    "batch_size": 64,
    "max_iter": 1_000,
    "C": 1.0,
}
