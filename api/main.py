"""FastAPI application — ticket routing REST API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR
from src.routing.router import route_ticket

logger = logging.getLogger(__name__)

# ── Request / Response schemas ───────────────────────────────────────────

class TicketRequest(BaseModel):
    subject: str = Field(..., min_length=1, examples=["Invoice issue"])
    body: str = Field(..., min_length=1, examples=["Charged twice for last month subscription"])


class TicketResponse(BaseModel):
    category: str
    urgency: str
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_variant: str


# ── Global state ─────────────────────────────────────────────────────────

_state: dict = {
    "classifier": None,
    "variant": "none",
}


def _load_model(variant: str):
    """Load the requested model variant from ``saved_models/``."""
    if variant == "logreg":
        from src.models.tfidf_logreg import TfidfLogRegClassifier
        clf = TfidfLogRegClassifier()
        path = MODEL_DIR / "tfidf_logreg.joblib"
        if path.exists():
            clf.load(path)
            return clf
        # If no saved model, train a quick synthetic one
        logger.info("No saved LogReg model — training on synthetic data …")
        from src.data.synthetic_generator import generate_dataset
        df = generate_dataset(n_per_class=2_000)
        clf.fit(df["text"], df["category"])
        clf.save(path)
        return clf

    elif variant == "svc":
        from src.models.tfidf_svc import TfidfSVCClassifier
        clf = TfidfSVCClassifier()
        path = MODEL_DIR / "tfidf_svc.joblib"
        if path.exists():
            clf.load(path)
            return clf
        # Fallback: train on synthetic data
        logger.info("No saved SVC model — training on synthetic data …")
        from src.data.synthetic_generator import generate_dataset
        df = generate_dataset(n_per_class=2_000)
        clf.fit(df["text"], df["category"])
        clf.save(path)
        return clf

    elif variant == "distilbert":
        from src.models.distilbert_classifier import DistilBertTicketClassifier
        clf = DistilBertTicketClassifier()
        path = MODEL_DIR / "distilbert_head.joblib"
        if path.exists():
            clf.load(path)
            return clf
        # Fallback: train on synthetic data (small)
        logger.info("No saved DistilBERT head — training on synthetic data …")
        from src.data.synthetic_generator import generate_dataset
        df = generate_dataset(n_per_class=200)
        clf.fit(df["text"].tolist(), df["category"])
        clf.save(path)
        return clf

    else:
        raise ValueError(f"Unknown model variant: {variant!r}")


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    variant = os.getenv("MODEL_VARIANT", "logreg")
    logger.info("Starting Smart-Support API with model=%s", variant)
    _state["classifier"] = _load_model(variant)
    _state["variant"] = variant
    yield
    _state["classifier"] = None


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart-Support",
    description="AI-powered customer support ticket routing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state["classifier"] is not None,
        model_variant=_state["variant"],
    )


@app.post("/route", response_model=TicketResponse)
async def route(req: TicketRequest):
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = route_ticket(req.subject, req.body, classifier=_state["classifier"])
    return TicketResponse(
        category=result["category"],
        urgency=result["urgency"],
        model_used=_state["variant"],
    )
