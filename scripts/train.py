#!/usr/bin/env python
"""Train all models — works with both synthetic and real data.

Usage
-----
    # Milestone 1: synthetic data + LogReg
    python -m scripts.train --milestone 1

    # Milestone 2: real HuggingFace data + LinearSVC
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


# ─── Milestone 1 — Synthetic TF-IDF + LogReg ────────────────────────────

def train_milestone_1():
    from src.config import MODEL_DIR
    from src.data.synthetic_generator import generate_dataset
    from src.models.tfidf_logreg import TfidfLogRegClassifier
    from evaluation.evaluator import evaluate_and_save

    logger.info("═══ Milestone 1: Synthetic LogReg ═══")
    df = generate_dataset(n_per_class=6_000)
    logger.info("Generated %d synthetic tickets", len(df))

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["category"], test_size=0.2,
        stratify=df["category"], random_state=42,
    )

    clf = TfidfLogRegClassifier()
    with _timer("LogReg training"):
        clf.fit(X_tr, y_tr)

    metrics = clf.evaluate(X_te, y_te)
    evaluate_and_save(y_te, clf.predict(X_te), model_name="tfidf_logreg")
    clf.save(MODEL_DIR / "tfidf_logreg.joblib")
    logger.info("Milestone 1 accuracy: %.4f", metrics["accuracy"])
    return metrics


# ─── Milestone 2 — Real Data + LinearSVC ────────────────────────────────

def train_milestone_2():
    from src.config import MODEL_DIR
    from src.data.dataset_loader import load_dataset, split_dataset
    from src.models.tfidf_svc import TfidfSVCClassifier
    from evaluation.evaluator import evaluate_and_save

    logger.info("═══ Milestone 2: Real Data + LinearSVC ═══")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te = split_dataset(df)

    clf = TfidfSVCClassifier()
    with _timer("SVC training"):
        clf.fit(X_tr, y_tr)

    metrics = clf.evaluate(X_te, y_te)
    evaluate_and_save(y_te, clf.predict(X_te), model_name="tfidf_svc")
    clf.save(MODEL_DIR / "tfidf_svc.joblib")
    logger.info("Milestone 2 accuracy: %.4f", metrics["accuracy"])
    return metrics


# ─── Milestone 3 — Real Data + DistilBERT ───────────────────────────────

def train_milestone_3():
    from src.config import MODEL_DIR
    from src.data.dataset_loader import load_dataset, split_dataset
    from src.models.distilbert_classifier import DistilBertTicketClassifier
    from evaluation.evaluator import evaluate_and_save

    logger.info("═══ Milestone 3: Real Data + DistilBERT ═══")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te = split_dataset(df)

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
