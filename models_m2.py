"""
models_m2.py — Milestone 2 ML module

DistilBERT-based ticket classifier + continuous urgency scorer.

Uses:
  - DistilBertModel [CLS] embeddings (distilbert-base-uncased)
  - LogisticRegression (C=5) for category — Billing / Legal / Technical
  - Ridge regression for urgency score S ∈ [0, 1]

Urgency label derivation (richer than 3-level):
  - Priority base:  low → 0.10 | medium → 0.45 | high → 0.80
  - Keyword boost:  +0.20 if urgency keywords found in text (capped 1.0)
  Results in 6 distinguishable bands instead of 3 discrete values.

Caches trained classifiers to model_cache/m2.joblib.
Saves eval metrics    to model_cache/m2_metrics.json.

Public API:
    ensure_loaded()                          → triggers train/cache load
    m2_ready()                               → bool
    predict_ticket_m2(subject, body)         → {category, urgency_score, confidence}
    get_metrics()                            → dict | None
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent
DATA_PATH    = _ROOT / "dataset" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
CACHE_DIR    = _ROOT / "model_cache"
CACHE_FILE   = CACHE_DIR / "m2.joblib"
METRICS_FILE = CACHE_DIR / "m2_metrics.json"

# Increase sample size for better accuracy — override with env var M2_TRAIN_SAMPLE
import os as _os
TRAIN_SAMPLE: int = int(_os.getenv("M2_TRAIN_SAMPLE", "12000"))

# ── label helpers ─────────────────────────────────────────────────────────────
_BILLING_QUEUES = {"billing and payments", "returns and exchanges"}
_LEGAL_QUEUES   = {"human resources"}

# Richer priority scores — avoid hard extremes so regression has room to learn
_PRIORITY_BASE  = {"low": 0.10, "medium": 0.45, "high": 0.80}
_PRIORITY_SCORE = _PRIORITY_BASE   # backward-compat alias

# Urgency keyword regex (same as M1 heuristic — applied at label-build time)
_URGENT_RE = re.compile(
    r"\b(asap|urgent|immediately|critical|down|outage|breach|broken|failed|error|emergency)\b",
    re.IGNORECASE,
)


def _map_queue(q: str) -> str:
    q = str(q).lower().strip()
    if q in _BILLING_QUEUES:
        return "Billing"
    if q in _LEGAL_QUEUES:
        return "Legal"
    return "Technical"


def _urgency_score(priority: str, text: str) -> float:
    """
    Derive a 6-band continuous urgency score from priority label + keyword signal.

    Bands:
        low    + no keyword  → 0.10
        low    + keyword     → 0.30
        medium + no keyword  → 0.45
        medium + keyword     → 0.65
        high   + no keyword  → 0.80
        high   + keyword     → 1.00
    """
    base    = _PRIORITY_BASE.get(str(priority).lower().strip(), 0.45)
    boost   = 0.20 if _URGENT_RE.search(str(text)) else 0.0
    return min(1.0, base + boost)


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
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        mean_absolute_error,
        mean_squared_error,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    logger.info("Loading dataset from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["text"]          = df["subject"].astype(str) + " " + df["body"].astype(str)
    df["category"]      = df["queue"].apply(_map_queue)

    # Richer urgency signal: priority base + keyword boost
    df["urgency_score"] = df.apply(
        lambda r: _urgency_score(r["priority"], r["text"]), axis=1
    )

    logger.info(
        "Urgency distribution:\n%s",
        df["urgency_score"].describe().to_string()
    )

    # Stratified sample — keep class proportions intact
    if len(df) > TRAIN_SAMPLE:
        sampled_parts = []
        for cat, grp in df.groupby("category"):
            n = max(1, int(TRAIN_SAMPLE * len(grp) / len(df)) + 1)
            sampled_parts.append(grp.sample(min(len(grp), n), random_state=42))
        df = (
            pd.concat(sampled_parts, ignore_index=True)
            .sample(frac=1, random_state=42)
            .head(TRAIN_SAMPLE)
            .reset_index(drop=True)
        )
        logger.info("Stratified sample: %d rows  %s",
                    len(df), df["category"].value_counts().to_dict())

    qenc = LabelEncoder()
    df["queue_label"] = qenc.fit_transform(df["category"])

    X_tr, X_te, yq_tr, yq_te, yu_tr, yu_te = train_test_split(
        df["text"], df["queue_label"], df["urgency_score"],
        test_size=0.2, stratify=df["queue_label"], random_state=42,
    )

    logger.info("Computing embeddings for %d train + %d test texts …", len(X_tr), len(X_te))
    X_tr_emb = _embed_texts(X_tr)
    X_te_emb = _embed_texts(X_te)

    # — category classifier (C=5 gives better margin than default C=1)
    logger.info("Training LogisticRegression classifier (C=5) …")
    clf = LogisticRegression(C=5, max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr_emb, yq_tr)
    yq_pred = clf.predict(X_te_emb)
    acc     = accuracy_score(yq_te, yq_pred)
    report  = classification_report(
        yq_te, yq_pred,
        target_names=list(qenc.classes_),
        output_dict=True,
    )
    logger.info(
        "Category classifier → acc=%.4f\n%s",
        acc,
        classification_report(yq_te, yq_pred, target_names=list(qenc.classes_))
    )

    # — urgency regressor (Ridge, alpha tuned)
    logger.info("Training Ridge urgency regressor …")
    reg        = Ridge(alpha=0.5)
    reg.fit(X_tr_emb, yu_tr)
    preds_urg  = np.clip(reg.predict(X_te_emb), 0.0, 1.0)
    rmse       = mean_squared_error(yu_te, preds_urg) ** 0.5
    mae        = mean_absolute_error(yu_te, preds_urg)
    logger.info("Urgency regressor → RMSE=%.4f  MAE=%.4f", rmse, mae)

    # Save metrics for API /metrics endpoint
    metrics = {
        "train_sample":     len(X_tr),
        "test_sample":      len(X_te),
        "category_accuracy": round(acc, 4),
        "category_report":  report,
        "urgency_rmse":     round(rmse, 4),
        "urgency_mae":      round(mae, 4),
        "label_classes":    list(qenc.classes_),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    logger.info("M2 metrics saved to %s", METRICS_FILE)

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
    Classify a ticket and return a continuous urgency score + confidence.

    Returns:
        {
            "category":      "Billing" | "Legal" | "Technical",
            "urgency_score": float  (0.0 = low, 1.0 = critical),
            "confidence":    float  (category prediction probability 0–1),
        }
    """
    ensure_loaded()
    emb        = _embed_texts([subject + " " + body])
    label_idx  = _queue_clf.predict(emb)[0]
    cat        = _queue_enc.inverse_transform([label_idx])[0]
    proba      = _queue_clf.predict_proba(emb)[0]
    confidence = float(proba[label_idx])
    score      = float(np.clip(_urgency_reg.predict(emb)[0], 0.0, 1.0))
    return {
        "category":      cat,
        "urgency_score": round(score, 4),
        "confidence":    round(confidence, 4),
    }


def get_metrics() -> dict | None:
    """Return normalized M2 training metrics, or None if not yet trained."""
    if METRICS_FILE.exists():
        try:
            raw = json.loads(METRICS_FILE.read_text())
            report = raw.get("category_report", {})
            return {
                "accuracy":      raw.get("category_accuracy"),
                "rmse":          raw.get("urgency_rmse"),
                "mae":           raw.get("urgency_mae"),
                "weighted avg":  report.get("weighted avg"),
                "macro avg":     report.get("macro avg"),
                "Billing":       report.get("Billing"),
                "Legal":         report.get("Legal"),
                "Technical":     report.get("Technical"),
                "train_sample":  raw.get("train_sample"),
                "test_sample":   raw.get("test_sample"),
            }
        except Exception:
            return None
    return None


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        ("Server down ASAP",        "Production API returning 500 errors immediately"),
        ("Critical outage",          "All services down, emergency response needed"),
        ("Invoice issue",            "Charged twice for last month subscription"),
        ("HR policy query",          "Employment contract terms clarification"),
        ("Login not working",        "Users cannot authenticate since last deploy"),
        ("Routine maintenance",      "Scheduled downtime notification next week"),
    ]
    print("\n{'category':10s}  S=score  conf=confidence  subject")
    print("-" * 70)
    for subj, body in tests:
        r    = predict_ticket_m2(subj, body)
        flag = "  ⚠️  WEBHOOK" if r["urgency_score"] > 0.8 else ""
        print(
            f"[{r['category']:10s}  S={r['urgency_score']:.3f}"
            f"  conf={r['confidence']:.2f}]{flag}  ← {subj!r}"
        )
    m = get_metrics()
    if m:
        print(f"\nModel metrics: acc={m['category_accuracy']}  "
              f"RMSE={m['urgency_rmse']}  MAE={m['urgency_mae']}")
