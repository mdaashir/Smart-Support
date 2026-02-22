"""Centralised configuration — single source of truth for every milestone."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = _BACKEND_DIR.parent
DATA_DIR = _BACKEND_DIR / "data"
MODEL_DIR = _BACKEND_DIR / "saved_models"
EVAL_DIR = _BACKEND_DIR / "evaluation" / "artifacts"

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

# ── Additional HuggingFace dataset for augmentation ─────────────────────
# Bitext customer-support dataset (~26 k English samples, intent labels)
BITEXT_DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BITEXT_DATASET_SPLIT = "train"
BITEXT_MERGED_CSV = DATA_DIR / "merged_tickets.csv"

# Map bitext intents → our 3 categories
BITEXT_INTENT_MAP: dict[str, str] = {
    # Billing / Payment
    "cancel_order": "Billing",
    "change_order": "Billing",
    "check_cancellation_fee": "Billing",
    "check_refund_policy": "Billing",
    "check_payment_methods": "Billing",
    "check_invoice": "Billing",
    "get_invoice": "Billing",
    "get_refund": "Billing",
    "payment_issue": "Billing",
    "place_order": "Billing",
    "track_refund": "Billing",
    # Technical
    "change_shipping_address": "Technical",
    "create_account": "Technical",
    "delete_account": "Technical",
    "delivery_options": "Technical",
    "delivery_period": "Technical",
    "edit_account": "Technical",
    "recover_password": "Technical",
    "registration_problems": "Technical",
    "set_up_shipping_address": "Technical",
    "switch_account": "Technical",
    "track_order": "Technical",
    # Legal / Service
    "complaint": "Legal",
    "contact_customer_service": "Legal",
    "contact_human_agent": "Legal",
    "newsletter_subscription": "Legal",
    "review": "Legal",
}

# ── Training & anti-overfit ──────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS = 5  # stratified k-fold cross-validation

# ── Model hyper-parameters ───────────────────────────────────────────────
TFIDF_LOGREG = {
    # Vocabulary
    "ngram_range": (1, 3),      # trigrams capture short phrases
    "max_features": 30_000,    # larger vocab → lower OOV rate
    "min_df": 2,               # drop hapax legomena (noise)
    "sublinear_tf": True,      # log(1+tf) dampens high-frequency terms
    "stop_words": "english",
    # Classifier
    "max_iter": 2_000,
    "C": 2.0,                  # slightly looser regularisation
    "solver": "saga",          # fast on large sparse matrices; handles multi-class natively
}

TFIDF_SVC = {
    # Word sub-vectorizer
    "word_ngram_range": (1, 2),
    "word_max_features": 30_000,
    "word_min_df": 2,
    "word_sublinear_tf": True,
    # Char sub-vectorizer (multilingual edge n-grams)
    "char_ngram_range": (3, 5),
    "char_max_features": 30_000,
    "char_min_df": 3,
    # LinearSVC
    "C": 1.5,
}

DISTILBERT = {
    "model_name": "distilbert-base-multilingual-cased",
    "max_length": 128,
    "batch_size": 64,
    "pooling": "mean",         # mean-pool all tokens > [CLS]-only
    # LogReg head
    "max_iter": 2_000,
    "C": 5.0,                  # embeddings already normalised → looser reg
    # Training sample caps
    "max_train": 20_000,       # doubled from previous 10 k
    "max_test": 5_000,
}
