"""Model evaluation & artifact generation — shared across milestones."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from backend.config import EVAL_DIR

logger = logging.getLogger(__name__)


def evaluate_and_save(
    y_true,
    y_pred,
    *,
    model_name: str,
    labels: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Compute metrics, save JSON report + confusion-matrix PNG.

    Returns the metrics dict.
    """
    out = Path(output_dir) if output_dir else EVAL_DIR / model_name
    out.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "classification_report": report,
    }

    # ── JSON report ──────────────────────────────────────────────────
    report_path = out / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved → %s", report_path)

    # ── Confusion matrix PNG ─────────────────────────────────────────
    unique_labels = labels or sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    cm_path = out / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", cm_path)

    # ── Console summary ──────────────────────────────────────────────
    print(
        f"\n{'=' * 50}\n"
        f"  {model_name}\n"
        f"  Accuracy   : {acc:.4f}\n"
        f"  Macro F1   : {macro:.4f}\n"
        f"  Weighted F1: {weighted:.4f}\n"
        f"{'=' * 50}\n"
    )
    print(classification_report(y_true, y_pred))

    return metrics
