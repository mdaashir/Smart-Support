"""FastAPI application — ticket routing REST API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CATEGORIES, MODEL_DIR
from src.routing.router import route_ticket

logger = logging.getLogger(__name__)

# ── Request / Response schemas ───────────────────────────────────────────


class TicketRequest(BaseModel):
    subject: str = Field(..., min_length=1, examples=["Invoice issue"])
    body: str = Field(
        ..., min_length=1, examples=["Charged twice for last month subscription"]
    )


class TicketResponse(BaseModel):
    category: str
    urgency: str
    model_used: str


class BatchTicketRequest(BaseModel):
    tickets: list[TicketRequest] = Field(..., min_length=1, max_length=100)


class BatchTicketResponse(BaseModel):
    results: list[TicketResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_variant: str
    categories: list[str]
    uptime_seconds: float


class StatsResponse(BaseModel):
    tickets_routed: int
    category_counts: dict[str, int]
    urgent_count: int


# ── Global state ─────────────────────────────────────────────────────────

_state: dict = {
    "classifier": None,
    "variant": "none",
    "started_at": None,
    "tickets_routed": 0,
    "category_counts": {c: 0 for c in CATEGORIES},
    "urgent_count": 0,
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
    _state["started_at"] = datetime.now(timezone.utc)
    yield
    _state["classifier"] = None


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart-Support",
    description=(
        "AI-powered multilingual customer support ticket routing system. "
        "Classifies tickets into Billing / Technical / HR / General and "
        "detects urgency in English and German."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    now = datetime.now(timezone.utc)
    uptime = (
        (now - _state["started_at"]).total_seconds() if _state["started_at"] else 0
    )
    return HealthResponse(
        status="ok",
        model_loaded=_state["classifier"] is not None,
        model_variant=_state["variant"],
        categories=CATEGORIES,
        uptime_seconds=round(uptime, 1),
    )


@app.post("/route", response_model=TicketResponse)
async def route(req: TicketRequest):
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = route_ticket(req.subject, req.body, classifier=_state["classifier"])
    # Track stats
    _state["tickets_routed"] += 1
    cat = result["category"]
    if cat in _state["category_counts"]:
        _state["category_counts"][cat] += 1
    if result["urgency"] == "1(HIGH)":
        _state["urgent_count"] += 1
    return TicketResponse(
        category=cat,
        urgency=result["urgency"],
        model_used=_state["variant"],
    )


@app.post("/route/batch", response_model=BatchTicketResponse)
async def route_batch(req: BatchTicketRequest):
    """Route up to 100 tickets in a single request — ideal for bulk triage."""
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for t in req.tickets:
        r = route_ticket(t.subject, t.body, classifier=_state["classifier"])
        _state["tickets_routed"] += 1
        cat = r["category"]
        if cat in _state["category_counts"]:
            _state["category_counts"][cat] += 1
        if r["urgency"] == "1(HIGH)":
            _state["urgent_count"] += 1
        results.append(
            TicketResponse(
                category=cat,
                urgency=r["urgency"],
                model_used=_state["variant"],
            )
        )
    return BatchTicketResponse(results=results, count=len(results))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Live routing statistics — great for dashboard demos."""
    return StatsResponse(
        tickets_routed=_state["tickets_routed"],
        category_counts=_state["category_counts"],
        urgent_count=_state["urgent_count"],
    )
