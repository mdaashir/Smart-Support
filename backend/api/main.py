"""FastAPI application — Smart-Support ticket routing REST API.

Implements all three milestones:
  M1 — Basic classification + regex urgency + priority queue
  M2 — 202 Accepted async pattern, urgency regression, webhook
  M3 — Semantic deduplication, circuit breaker, skill-based routing
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import CATEGORIES, MODEL_DIR, URGENCY_WEBHOOK_THRESHOLD

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
    urgency_score: float = 0.0
    model_used: str
    agent: str | None = None
    dedup: dict | None = None


class AsyncTicketResponse(BaseModel):
    job_id: str
    status: str = "accepted"
    message: str = "Ticket queued for processing"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None


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
    circuit_breaker: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    tickets_routed: int
    category_counts: dict[str, int]
    urgent_count: int
    webhook_fires: int
    master_incidents: int
    agent_status: dict | None = None


# ── Global state ─────────────────────────────────────────────────────────

_state: dict = {
    "classifier": None,
    "variant": "none",
    "started_at": None,
    "tickets_routed": 0,
    "category_counts": {c: 0 for c in CATEGORIES},
    "urgent_count": 0,
    "webhook_fires": 0,
    # M2/M3 components
    "urgency_regressor": None,
    "broker": None,
    "deduplicator": None,
    "circuit_breaker": None,
    "skill_router": None,
}


def _load_model(variant: str):
    """Load the requested model variant from ``saved_models/``."""
    if variant == "logreg":
        from backend.models.tfidf_logreg import TfidfLogRegClassifier
        clf = TfidfLogRegClassifier()
        path = MODEL_DIR / "tfidf_logreg.joblib"
        if path.exists():
            clf.load(path)
            return clf
        # Train on real data as fallback
        logger.info("No saved LogReg model — training on real data …")
        from backend.data.dataset_loader import load_dataset, split_dataset
        df = load_dataset()
        X_tr, _, y_tr, _ = split_dataset(df)
        clf.fit(X_tr, y_tr)
        clf.save(path)
        return clf

    elif variant == "svc":
        from backend.models.tfidf_svc import TfidfSVCClassifier
        clf = TfidfSVCClassifier()
        path = MODEL_DIR / "tfidf_svc.joblib"
        if path.exists():
            clf.load(path)
            return clf
        logger.info("No saved SVC model — training on real data …")
        from backend.data.dataset_loader import load_dataset, split_dataset
        df = load_dataset()
        X_tr, _, y_tr, _ = split_dataset(df)
        clf.fit(X_tr, y_tr)
        clf.save(path)
        return clf

    elif variant == "distilbert":
        from backend.models.distilbert_classifier import DistilBertTicketClassifier
        clf = DistilBertTicketClassifier()
        path = MODEL_DIR / "distilbert_head.joblib"
        if path.exists():
            clf.load(path)
            return clf
        logger.info("No saved DistilBERT head — training on real data …")
        from backend.data.dataset_loader import load_dataset, split_dataset
        df = load_dataset()
        X_tr, _, y_tr, _ = split_dataset(df)
        clf.fit(X_tr.tolist()[:5000], y_tr[:5000])
        clf.save(path)
        return clf

    else:
        raise ValueError(f"Unknown model variant: {variant!r}")


def _load_urgency_regressor():
    """Load trained urgency regressor if available."""
    from backend.routing.urgency_regressor import UrgencyRegressor
    urg = UrgencyRegressor()
    path = MODEL_DIR / "urgency_regressor.joblib"
    if path.exists():
        urg.load(path)
    return urg


# ── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    variant = os.getenv("MODEL_VARIANT", "logreg")
    logger.info("Starting Smart-Support API with model=%s", variant)
    _state["classifier"] = _load_model(variant)
    _state["variant"] = variant
    _state["started_at"] = datetime.now(timezone.utc)

    # M2: urgency regressor
    _state["urgency_regressor"] = _load_urgency_regressor()

    # M2: async broker (Redis-backed with in-memory fallback)
    from backend.routing.redis_broker import RedisBroker
    broker = RedisBroker()
    _state["broker"] = broker

    async def _process_ticket(subject: str, body: str) -> dict:
        return _route_ticket_full(subject, body)

    broker.register_processor(_process_ticket)
    await broker.start_workers(n=4)

    # M3: skill router
    from backend.routing.skill_router import SkillRouter
    _state["skill_router"] = SkillRouter()

    # M3: deduplicator (lazy — sentence model loaded on first use)
    from backend.routing.deduplicator import SemanticDeduplicator
    _state["deduplicator"] = SemanticDeduplicator()

    # M3: circuit breaker (if we have both primary and fallback)
    if variant == "distilbert":
        try:
            from backend.models.tfidf_logreg import TfidfLogRegClassifier
            from backend.routing.circuit_breaker import CircuitBreaker
            fallback = TfidfLogRegClassifier()
            fb_path = MODEL_DIR / "tfidf_logreg.joblib"
            if fb_path.exists():
                fallback.load(fb_path)
                _state["circuit_breaker"] = CircuitBreaker(
                    primary=_state["classifier"], fallback=fallback,
                )
                logger.info("Circuit breaker enabled (DistilBERT → LogReg fallback)")
        except Exception:
            logger.warning("Could not set up circuit breaker")

    yield

    # Shutdown
    if _state["broker"]:
        await _state["broker"].stop_workers()
    _state["classifier"] = None


def _route_ticket_full(subject: str, body: str) -> dict:
    """Core routing logic — classification + urgency + skill routing + dedup."""
    from backend.preprocessing.text_cleaner import combine_fields

    ticket_id = str(uuid.uuid4())
    text = combine_fields(subject, body)

    # Classification (with circuit breaker if available)
    clf = _state["circuit_breaker"] or _state["classifier"]
    category = clf.predict([text])[0]

    # Urgency
    from backend.routing.urgency import urgency_label
    urgency = urgency_label(text)
    urgency_score = 0.0
    urg_reg = _state.get("urgency_regressor")
    if urg_reg and urg_reg.is_trained:
        urgency_score = urg_reg.predict_score(text)

    # Skill-based routing
    agent_info = None
    skill_router = _state.get("skill_router")
    if skill_router:
        agent_info = skill_router.assign(category, urgency_score)

    # Deduplication
    dedup_info = None
    deduplicator = _state.get("deduplicator")
    if deduplicator:
        try:
            dedup_info = deduplicator.check(ticket_id, text)
        except Exception:
            logger.debug("Dedup check skipped (model not ready)")

    # Stats tracking
    _state["tickets_routed"] += 1
    if category in _state["category_counts"]:
        _state["category_counts"][category] += 1
    if urgency == "1(HIGH)":
        _state["urgent_count"] += 1

    result = {
        "category": category,
        "urgency": urgency,
        "urgency_score": round(urgency_score, 4),
        "model_used": _state["variant"],
        "agent": agent_info.get("agent") if agent_info else None,
        "dedup": dedup_info,
    }

    # Webhook for high urgency
    if urgency_score > URGENCY_WEBHOOK_THRESHOLD:
        _state["webhook_fires"] += 1
        # Fire webhook asynchronously (best-effort)
        try:
            from backend.routing.webhook import fire_webhook
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(fire_webhook(
                    ticket_id=ticket_id,
                    category=category,
                    urgency_score=urgency_score,
                    text_preview=text[:200],
                ))
        except Exception:
            logger.debug("Webhook fire skipped")

    return result


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart-Support",
    description=(
        "AI-powered multilingual customer support ticket routing system. "
        "Classifies tickets into Billing / Technical / Legal, detects "
        "urgency (regression score S ∈ [0,1]), routes via skill-based "
        "assignment, and detects duplicate spikes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the dashboard to call the API from any origin ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    now = datetime.now(timezone.utc)
    uptime = (
        (now - _state["started_at"]).total_seconds() if _state["started_at"] else 0
    )
    cb = _state.get("circuit_breaker")
    return HealthResponse(
        status="ok",
        model_loaded=_state["classifier"] is not None,
        model_variant=_state["variant"],
        categories=CATEGORIES,
        circuit_breaker=cb.current_state if cb else "N/A",
        uptime_seconds=round(uptime, 1),
    )


@app.post("/route", response_model=TicketResponse)
async def route(req: TicketRequest):
    """Synchronous route — returns classification immediately."""
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _route_ticket_full(req.subject, req.body)
    return TicketResponse(**result)


@app.post("/route/async", response_model=AsyncTicketResponse, status_code=202)
async def route_async(req: TicketRequest):
    """Milestone 2 — 202 Accepted pattern.  Queues ticket for background processing."""
    broker = _state.get("broker")
    if broker is None:
        raise HTTPException(status_code=503, detail="Broker not initialised")
    job_id = await broker.submit(req.subject, req.body)
    return AsyncTicketResponse(job_id=job_id)


@app.get("/route/async/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the result of an async ticket routing job."""
    broker = _state.get("broker")
    if broker is None:
        raise HTTPException(status_code=503, detail="Broker not initialised")
    job = await broker.get_result(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
    )


@app.post("/route/batch", response_model=BatchTicketResponse)
async def route_batch(req: BatchTicketRequest):
    """Route up to 100 tickets using Hungarian-algorithm batch assignment.

    Classifies all tickets in one pass, then runs constraint-optimised
    skill routing via ``SkillRouter.batch_assign()`` for globally optimal
    agent assignment before assembling results.
    """
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from backend.preprocessing.text_cleaner import combine_fields
    from backend.routing.urgency import urgency_label

    # 1. Vectorise texts and classify in one batch
    clf = _state["circuit_breaker"] or _state["classifier"]
    urg_reg = _state.get("urgency_regressor")
    deduplicator = _state.get("deduplicator")
    skill_router = _state.get("skill_router")

    texts = [combine_fields(t.subject, t.body) for t in req.tickets]
    categories = clf.predict(texts)
    urgencies = [urgency_label(t) for t in texts]
    urgency_scores = [
        urg_reg.predict_score(t) if (urg_reg and urg_reg.is_trained) else 0.0
        for t in texts
    ]

    # 2. Constraint-optimised batch agent assignment (Hungarian algorithm)
    ticket_ids = [str(uuid.uuid4()) for _ in req.tickets]
    if skill_router:
        batch_items = [
            {"id": tid, "category": cat, "urgency": score}
            for tid, cat, score in zip(ticket_ids, categories, urgency_scores)
        ]
        assignments = skill_router.batch_assign(batch_items)
    else:
        assignments = [
            {"ticket_id": tid, "agent": None, "affinity": 0.0, "load": 0}
            for tid in ticket_ids
        ]

    # 3. Dedup check + stats + assemble
    results = []
    for tid, text, cat, urg, score, assign in zip(
        ticket_ids, texts, categories, urgencies, urgency_scores, assignments
    ):
        dedup_info = None
        if deduplicator:
            try:
                dedup_info = deduplicator.check(tid, text)
            except Exception:
                pass

        _state["tickets_routed"] += 1
        if cat in _state["category_counts"]:
            _state["category_counts"][cat] += 1
        if urg == "1(HIGH)":
            _state["urgent_count"] += 1

        results.append(TicketResponse(
            category=cat,
            urgency=urg,
            urgency_score=round(score, 4),
            model_used=_state["variant"],
            agent=assign.get("agent"),
            dedup=dedup_info,
        ))

    return BatchTicketResponse(results=results, count=len(results))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Live routing statistics — great for dashboard demos."""
    dedup = _state.get("deduplicator")
    skill = _state.get("skill_router")
    return StatsResponse(
        tickets_routed=_state["tickets_routed"],
        category_counts=_state["category_counts"],
        urgent_count=_state["urgent_count"],
        webhook_fires=_state.get("webhook_fires", 0),
        master_incidents=len(dedup.get_incidents()) if dedup else 0,
        agent_status=skill.status() if skill else None,
    )


@app.get("/incidents")
async def incidents():
    """List all master incidents created by semantic deduplication."""
    dedup = _state.get("deduplicator")
    if not dedup:
        return {"incidents": []}
    return {
        "incidents": [
            {
                "incident_id": i.incident_id,
                "representative_text": i.representative_text,
                "ticket_count": len(i.ticket_ids),
                "created_at": i.created_at,
            }
            for i in dedup.get_incidents()
        ]
    }


@app.get("/agents")
async def agents_status():
    """Current agent load and utilisation."""
    skill = _state.get("skill_router")
    if not skill:
        return {"agents": {}}
    return {"agents": skill.status()}
