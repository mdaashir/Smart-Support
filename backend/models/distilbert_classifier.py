"""Milestone 2 — Transformer-based classifier: DistilBERT embeddings + Logistic Regression head.

Replaces the baseline TF-IDF + LogReg (M1) with a pretrained multilingual
Transformer that produces contextual 768-d embeddings.  A lightweight
Logistic Regression head maps embeddings to the three ticket categories.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizerFast

from backend.config import DISTILBERT

logger = logging.getLogger(__name__)


class DistilBertEmbedder:
    """Produce mean-pooled embeddings from a multilingual DistilBERT model."""

    def __init__(
        self,
        model_name: str = DISTILBERT["model_name"],
        max_length: int = DISTILBERT["max_length"],
        batch_size: int = DISTILBERT.get("batch_size", 64),
        pooling: str = DISTILBERT.get("pooling", "mean"),
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s ...", model_name, self.device)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), 768) embeddings using mean-pooling over non-pad tokens.

        Mean-pooling uses the full sequence representation rather than the
        single [CLS] token, which consistently improves classification quality.
        """
        all_embs: list[np.ndarray] = []
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Embedding",
            disable=len(texts) < 50,
        ):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.pooling == "mean":
                # Mask padding tokens before averaging
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                token_emb = outputs.last_hidden_state  # (B, T, 768)
                summed = (token_emb * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = (summed / counts).cpu().numpy()
            else:
                # CLS-only fallback
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embs.append(emb)
        return np.vstack(all_embs)


class DistilBertTicketClassifier:
    """DistilBERT embedder → LabelEncoder → LogisticRegression."""

    def __init__(self) -> None:
        self.embedder = DistilBertEmbedder()
        self.label_encoder = LabelEncoder()
        self.clf = LogisticRegression(
            max_iter=DISTILBERT["max_iter"],
            C=DISTILBERT["C"],
            solver="saga",
            class_weight="balanced",
        )

    # ── Training ─────────────────────────────────────────────────────
    def fit(self, X: list[str], y) -> "DistilBertTicketClassifier":
        logger.info("Training DistilBERT + LogReg on %d samples …", len(X))
        y_enc = self.label_encoder.fit_transform(y)
        X_emb = self.embedder.embed(list(X))
        self.clf.fit(X_emb, y_enc)
        return self

    # ── Inference ────────────────────────────────────────────────────
    def predict(self, X: list[str]):
        X_emb = self.embedder.embed(list(X))
        y_enc = self.clf.predict(X_emb)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    @property
    def classes_(self):
        return self.label_encoder.classes_

    # ── Evaluation ───────────────────────────────────────────────────
    def evaluate(self, X_test: list[str], y_test) -> dict:
        y_pred = self.predict(list(X_test))
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "report": report,
        }
        logger.info(
            "Evaluation — acc=%.4f  macro-F1=%.4f  weighted-F1=%.4f",
            metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"],
        )
        return metrics

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"label_encoder": self.label_encoder, "clf": self.clf},
            path,
        )
        logger.info("Head saved → %s  (embedder weights from HuggingFace cache)", path)

    def load(self, path: str | Path) -> "DistilBertTicketClassifier":
        bundle = joblib.load(path)
        self.label_encoder = bundle["label_encoder"]
        self.clf = bundle["clf"]
        logger.info("Head loaded ← %s", path)
        return self
