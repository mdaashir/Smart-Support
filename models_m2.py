"""
models_m2.py — Milestone 2 ML module

DistilBERT-based ticket classifier + continuous urgency scorer.

Uses:
  - DistilBertModel [CLS] embeddings
  - LogisticRegression for category (Billing / Legal / Technical)
  - Ridge regression for urgency score S ∈ [0, 1]

Caches trained classifiers to model_cache/m2.joblib.
DistilBERT weights are loaded from HuggingFace cache.

Training samples 5 000 rows so embedding computation
stays under ~5 min on CPU.

Public API:
    ensure_loaded()                      → triggers train/cache load
    m2_ready()                           → bool
    predict_ticket_m2(subject, body)     → {category, urgency_score}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent
DATA_PATH  = _ROOT / "dataset" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
CACHE_DIR  = _ROOT / "model_cache"
CACHE_FILE = CACHE_DIR / "m2.joblib"

# Use ~5k samples so CPU training stays practical (full set ≈ 28k rows)
TRAIN_SAMPLE = 5_000

# ── label helpers ─────────────────────────────────────────────────────────────
_BILLING_QUEUES = {"billing and payments", "returns and exchanges"}
_LEGAL_QUEUES   = {"human resources"}
_PRIORITY_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}


def _map_queue(q: str) -> str:
    q = str(q).lower().strip()
    if q in _BILLING_QUEUES:
        return "Billing"
    if q in _LEGAL_QUEUES:
        return "Legal"
    return "Technical"


# ── lazy module-level state ───────────────────────────────────────────────────
_tokenizer   = None
_bert        = None
_queue_clf   = None
_urgency_reg = None
_queue_enc   = None
_loaded      = False


# ── DistilBERT loading ────────────────────────────────────────────────────────
def _get_bert():
    global _tokenizer, _bert
    if _tokenizer is None:
        from transformers import AutoTokenizer, DistilBertModel

        logger.info("Loading DistilBERT …")
        # Use AutoTokenizer for better transformers v4/v5 compatibility
        _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        _bert      = DistilBertModel.from_pretrained("distilbert-base-uncased")
        _bert.eval()
        logger.info("DistilBERT ready.")
    return _tokenizer, _bert


# ── embedding ─────────────────────────────────────────────────────────────────
def _embed_texts(texts, batch_size: int = 32) -> np.ndarray:
    import torch
    from tqdm import tqdm

    tok, bert = _get_bert()
    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    out_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        # Explicit str cast — guards against NaN/None and strict tokenizers v5
        batch  = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            out = bert(**inputs)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            out_list.append(cls)
    return np.vstack(out_list)


# ── training ──────────────────────────────────────────────────────────────────
def _train():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    logger.info("Loading dataset from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["text"]          = df["subject"].astype(str) + " " + df["body"].astype(str)
    df["category"]      = df["queue"].apply(_map_queue)
    df["urgency_score"] = df["priority"].str.lower().map(_PRIORITY_SCORE).fillna(0.5)

    # Sample for fast CPU training
    if len(df) > TRAIN_SAMPLE:
        df = df.sample(TRAIN_SAMPLE, random_state=42)
        logger.info("Using %d-sample subset for embedding (CPU-friendly).", TRAIN_SAMPLE)

    qenc = LabelEncoder()
    df["queue_label"] = qenc.fit_transform(df["category"])

    X_tr, X_te, yq_tr, yq_te, yu_tr, yu_te = train_test_split(
        df["text"], df["queue_label"], df["urgency_score"],
        test_size=0.2, stratify=df["queue_label"], random_state=42,
    )

    logger.info("Computing embeddings for %d train + %d test texts …", len(X_tr), len(X_te))
    X_tr_emb = _embed_texts(X_tr)
    X_te_emb = _embed_texts(X_te)

    # — category classifier
    logger.info("Training LogisticRegression classifier …")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr_emb, yq_tr)
    acc = accuracy_score(yq_te, clf.predict(X_te_emb))

    # — urgency regressor
    logger.info("Training Ridge urgency regressor …")
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr_emb, yu_tr)
    preds_urg = np.clip(reg.predict(X_te_emb), 0.0, 1.0)
    rmse = mean_squared_error(yu_te, preds_urg) ** 0.5

    logger.info("M2 training done — category acc=%.4f | urgency RMSE=%.4f", acc, rmse)
    return clf, reg, qenc


# ── cache load / train ────────────────────────────────────────────────────────
def _load_or_train() -> None:
    global _queue_clf, _urgency_reg, _queue_enc
    import joblib

    if CACHE_FILE.exists():
        logger.info("Loading cached M2 model from %s", CACHE_FILE)
        _queue_clf, _urgency_reg, _queue_enc = joblib.load(CACHE_FILE)
        return

    _queue_clf, _urgency_reg, _queue_enc = _train()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((_queue_clf, _urgency_reg, _queue_enc), CACHE_FILE)
    logger.info("M2 model cached at %s", CACHE_FILE)


# ── public API ────────────────────────────────────────────────────────────────
def ensure_loaded() -> None:
    """Load / train the M2 model (idempotent)."""
    global _loaded
    if not _loaded:
        _load_or_train()
        _loaded = True


def m2_ready() -> bool:
    """Return True if the M2 model is loaded and ready for inference."""
    return _loaded


def predict_ticket_m2(subject: str, body: str) -> dict:
    """
    Classify a ticket and return a continuous urgency score.

    Returns:
        {
            "category":      "Billing" | "Legal" | "Technical",
            "urgency_score": float  (0.0 = low, 1.0 = critical),
        }
    """
    ensure_loaded()
    emb   = _embed_texts([subject + " " + body])
    cat   = _queue_enc.inverse_transform(_queue_clf.predict(emb))[0]
    score = float(np.clip(_urgency_reg.predict(emb)[0], 0.0, 1.0))
    return {"category": cat, "urgency_score": round(score, 4)}


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        ("Server down ASAP",     "Production API returning 500 errors immediately"),
        ("Invoice issue",        "Charged twice for last month subscription"),
        ("HR policy query",      "Employment contract terms clarification"),
        ("Login not working",    "Users cannot authenticate since last deploy"),
    ]
    for subj, body in tests:
        r = predict_ticket_m2(subj, body)
        flag = " ⚠️  WEBHOOK TRIGGER" if r["urgency_score"] > 0.8 else ""
        print(f"[{r['category']:10s}  S={r['urgency_score']:.3f}]{flag}  ← {subj!r}")
