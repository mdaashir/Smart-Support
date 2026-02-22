#!/usr/bin/env python
"""Train all models — real open-source data only (no synthetic templates).

Usage
-----
    # Milestone 1: real data + LogReg  (cross-validated)
    python -m scripts.train --milestone 1

    # Milestone 2: real data + LinearSVC + urgency regressor
    python -m scripts.train --milestone 2

    # Milestone 3: real data + DistilBERT
    python -m scripts.train --milestone 3

    # All milestones at once
    python -m scripts.train --milestone all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("train")


# ─── Helpers ─────────────────────────────────────────────────────────────

def _timer(label: str):
    """Context manager that logs elapsed time."""
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            logger.info("%s finished in %.1f s", label, time.perf_counter() - self.t0)
    return _T()


def _cross_validate(pipeline_or_clf, X, y, *, cv: int, model_name: str) -> dict:
    """Run stratified k-fold CV and log per-fold metrics."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_or_clf, X, y, cv=skf, scoring="f1_macro")

    logger.info(
        "%s  CV(%d) macro-F1: %.4f ± %.4f  [%s]",
        model_name, cv, scores.mean(), scores.std(),
        ", ".join(f"{s:.4f}" for s in scores),
    )
    return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}


# ─── Milestone 1 — Real Data + TF-IDF + LogReg ──────────────────────────

def train_milestone_1():
    from backend.config import CV_FOLDS, MODEL_DIR
    from backend.data.dataset_loader import load_dataset, split_dataset
    from backend.models.tfidf_logreg import TfidfLogRegClassifier
    from backend.evaluation.evaluator import evaluate_and_save

    logger.info("═══ Milestone 1: Real Data + LogReg ═══")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te = split_dataset(df)

    clf = TfidfLogRegClassifier()

    # Cross-validation on training set to check for overfitting
    _cross_validate(clf.pipeline, X_tr, y_tr, cv=CV_FOLDS, model_name="LogReg")

    with _timer("LogReg training"):
        clf.fit(X_tr, y_tr)

    metrics = clf.evaluate(X_te, y_te)
    evaluate_and_save(y_te, clf.predict(X_te), model_name="tfidf_logreg")
    clf.save(MODEL_DIR / "tfidf_logreg.joblib")
    logger.info("Milestone 1 accuracy: %.4f", metrics["accuracy"])
    return metrics


# ─── Milestone 2 — Real Data + LinearSVC + Urgency Regressor ────────────

def train_milestone_2():
    from backend.config import MODEL_DIR
    from backend.data.dataset_loader import load_dataset, split_dataset
    from backend.models.tfidf_svc import TfidfSVCClassifier
    from backend.routing.urgency_regressor import UrgencyRegressor
    from backend.evaluation.evaluator import evaluate_and_save

    logger.info("═══ Milestone 2: Real Data + LinearSVC + Urgency ═══")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te = split_dataset(df)

    # ── Classification ───────────────────────────────────────────────
    clf = TfidfSVCClassifier()
    with _timer("SVC training"):
        clf.fit(X_tr, y_tr)

    metrics = clf.evaluate(X_te, y_te)
    evaluate_and_save(y_te, clf.predict(X_te), model_name="tfidf_svc")
    clf.save(MODEL_DIR / "tfidf_svc.joblib")
    logger.info("Milestone 2 classification accuracy: %.4f", metrics["accuracy"])

    # ── Urgency regressor ────────────────────────────────────────────
    if "priority_num" in df.columns:
        logger.info("Training urgency regressor …")
        train_df = df.loc[X_tr.index]
        urg = UrgencyRegressor()
        urg.fit(train_df["text"].tolist(), train_df["priority_num"].tolist())
        urg.save(MODEL_DIR / "urgency_regressor.joblib")

        # Quick sanity check
        sample_scores = [urg.predict_score(t) for t in X_te.head(5)]
        logger.info("Sample urgency scores: %s", [round(s, 3) for s in sample_scores])
    else:
        logger.warning("No priority_num column — skipping urgency regressor")

    return metrics


# ─── Milestone 3 — Real Data + DistilBERT ───────────────────────────────

def train_milestone_3():
    from backend.config import MODEL_DIR
    from backend.data.dataset_loader import load_dataset, split_dataset
    from backend.models.distilbert_classifier import DistilBertTicketClassifier
    from backend.evaluation.evaluator import evaluate_and_save

    logger.info("═══ Milestone 3: Real Data + DistilBERT ═══")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te = split_dataset(df)

    # Use a subset for DistilBERT to avoid excessive training time
    from sklearn.model_selection import train_test_split as _tts

    max_train = 10_000
    if len(X_tr) > max_train:
        X_tr, _, y_tr, _ = _tts(
            X_tr, y_tr, train_size=max_train,
            stratify=y_tr, random_state=42,
        )
        logger.info("Sub-sampled training set to %d for DistilBERT", max_train)

    max_test = 3_000
    if len(X_te) > max_test:
        X_te, _, y_te, _ = _tts(
            X_te, y_te, train_size=max_test,
            stratify=y_te, random_state=42,
        )
        logger.info("Sub-sampled test set to %d for DistilBERT eval", max_test)

    clf = DistilBertTicketClassifier()
    with _timer("DistilBERT training"):
        clf.fit(X_tr.tolist(), y_tr)

    metrics = clf.evaluate(X_te.tolist(), y_te)
    evaluate_and_save(y_te, clf.predict(X_te.tolist()), model_name="distilbert")
    clf.save(MODEL_DIR / "distilbert_head.joblib")
    logger.info("Milestone 3 accuracy: %.4f", metrics["accuracy"])
    return metrics


# ─── CLI entry point ────────────────────────────────────────────────────

MILESTONES = {
    "1": train_milestone_1,
    "2": train_milestone_2,
    "3": train_milestone_3,
}


def main():
    parser = argparse.ArgumentParser(description="Smart-Support model training")
    parser.add_argument(
        "--milestone", "-m",
        default="all",
        help="Which milestone to train: 1, 2, 3, or 'all' (default: all)",
    )
    args = parser.parse_args()

    targets = list(MILESTONES.keys()) if args.milestone == "all" else [args.milestone]

    results = {}
    for t in targets:
        if t not in MILESTONES:
            logger.error("Unknown milestone: %s", t)
            sys.exit(1)
        results[t] = MILESTONES[t]()

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║          Training Summary                ║")
    logger.info("╠══════════════════════════════════════════╣")
    for m, metrics in results.items():
        logger.info(
            "║  Milestone %s — acc=%.4f  F1=%.4f  ║",
            m, metrics["accuracy"], metrics["weighted_f1"],
        )
    logger.info("╚══════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
