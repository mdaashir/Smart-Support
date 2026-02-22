"""
api/main.py — Smart-Support FastAPI  (Milestone 1 + 2)

Milestone 1 endpoints (sync, in-memory heapq):
    POST   /v1/tickets            → 201  TicketOut  (M1 model, immediate)
    GET    /v1/tickets/next       → pop highest-priority from in-memory queue
    GET    /v1/tickets/peek       → peek without dequeuing
    GET    /v1/tickets/status     → in-memory queue depth

Milestone 2 endpoints (async, Redis broker):
    POST   /v2/tickets            → 202  TicketAccepted  (enqueues to arq)
    GET    /v2/tickets/{id}       → TicketResult (pending | processed)
    GET    /v2/tickets/next       → ZPOPMAX from processed sorted set
    GET    /v2/tickets/status     → pending + processed counts

Shared:
    GET    /health                → liveness + model / Redis status

Run API:      uv run uvicorn api.main:app --reload
Run worker:   uv run arq api.worker.WorkerSettings

Set optional env vars in .env:
    REDIS_URL=redis://localhost:6379
    WEBHOOK_URL=https://hooks.slack.com/...   (empty = mock log)
    WEBHOOK_THRESHOLD=0.8
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# ── local ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
load_dotenv()

import models as ml_m1
from api import queue_store
from api.schemas import (
    ModelMetrics,
    QueueStatus,
    TicketAccepted,
    TicketIn,
    TicketOut,
    TicketResult,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis key helpers (mirrors worker.py)
def _ticket_key(tid: str) -> str:  return f"ticket:{tid}"
def _processed_set() -> str:       return "processed_tickets"

# ── lifespan ──────────────────────────────────────────────────────────────────
_arq_pool   = None   # set during lifespan
_redis_conn = None   # raw redis.asyncio connection for reads


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _arq_pool, _redis_conn

    # 1. Warm up M1 (always available, no Redis required)
    logger.info("Warming up M1 model …")
    ml_m1._ensure_loaded()
    logger.info("M1 ready.")

    # 2. Connect to Redis (M2 broker) — graceful degradation if Redis is down
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        import redis.asyncio as aioredis

        _arq_pool   = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        _redis_conn = await aioredis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis connected → M2 async mode available.")
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — M2 endpoints will return 503.", exc)
        _arq_pool   = None
        _redis_conn = None

    yield

    if _arq_pool:
        await _arq_pool.aclose()
    if _redis_conn:
        await _redis_conn.aclose()
    logger.info("Shutdown complete.")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart-Support Ticket Routing Engine",
    description=(
        "**Milestone 1** — TF-IDF/LinearSVC + in-memory priority queue (sync, 201)\n\n"
        "**Milestone 2** — DistilBERT + Ridge urgency + Redis/arq async broker (202)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _require_redis():
    if _arq_pool is None or _redis_conn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis broker not available. Run: docker run -p 6379:6379 redis:7-alpine",
        )


# ═════════════════════════════════════════════════════════════════════════════
# SHARED
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Liveness probe — reports M1 model + Redis/M2 status."""
    redis_ok = _redis_conn is not None
    redis_info: dict = {}
    if redis_ok:
        try:
            # arq >= 0.25 uses a sorted set for its queue
            try:
                pending = await _redis_conn.zcard("arq:queue")
            except Exception:
                pending = await _redis_conn.llen("arq:queue")
            processed = await _redis_conn.zcard(_processed_set())
            redis_info = {"pending": pending, "processed": processed}
        except Exception:
            redis_info = {"error": "redis read failed"}

    m1_met = ml_m1.get_metrics()
    return {
        "status":    "ok",
        "m1":        "ready",
        "m2_broker": "ready" if redis_ok else "unavailable (start Redis to enable)",
        "redis":     redis_info,
        "m1_queue":  queue_store.size(),
        "m1_accuracy": m1_met["accuracy"] if m1_met else None,
    }


@app.get("/metrics", response_model=ModelMetrics, tags=["System"])
async def get_metrics() -> ModelMetrics:
    """Return cached training / evaluation metrics for M1 and M2 models."""
    import models_m2 as ml_m2
    return ModelMetrics(
        m1=ml_m1.get_metrics(),
        m2=ml_m2.get_metrics(),
    )


# ═════════════════════════════════════════════════════════════════════════════
# MILESTONE 1 — sync  /v1/...
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/v1/tickets",
    response_model=TicketOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Milestone 1"],
    summary="Ingest ticket (sync 201)",
)
def m1_ingest(payload: TicketIn) -> TicketOut:
    """Classify synchronously with M1 (TF-IDF+SVM) and push to in-memory queue."""
    result = ml_m1.route_ticket(payload.subject, payload.body)
    ticket = TicketOut(
        subject=payload.subject,
        body=payload.body,
        category=result["category"],
        urgency_level=result["urgency_level"],
    )
    queue_store.push(ticket)
    logger.info("M1 ingested %s  cat=%s  urgency=%s",
                ticket.ticket_id, ticket.category, ticket.urgency_level)
    return ticket


@app.get("/v1/tickets/next", tags=["Milestone 1"], summary="Pop highest-priority ticket")
def m1_next() -> dict:
    t = queue_store.pop()
    if t is None:
        raise HTTPException(status_code=404, detail="Queue is empty.")
    return t


@app.get("/v1/tickets/peek", tags=["Milestone 1"], summary="Peek without dequeuing")
def m1_peek() -> dict:
    t = queue_store.peek()
    if t is None:
        raise HTTPException(status_code=404, detail="Queue is empty.")
    return t


@app.get(
    "/v1/tickets/status",
    response_model=QueueStatus,
    tags=["Milestone 1"],
    summary="In-memory queue depth",
)
def m1_status() -> QueueStatus:
    return QueueStatus(pending=queue_store.size(), processed=0)


@app.delete("/v1/tickets", tags=["Milestone 1"], summary="Drain in-memory queue")
def m1_drain() -> dict:
    tickets = queue_store.drain()
    return {"drained": len(tickets), "tickets": tickets}


# ═════════════════════════════════════════════════════════════════════════════
# MILESTONE 2 — async  /v2/...
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/v2/tickets",
    response_model=TicketAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Milestone 2"],
    summary="Ingest ticket (async 202)",
)
async def m2_ingest(payload: TicketIn) -> TicketAccepted:
    """
    Enqueue ticket to Redis/arq and return **202 Accepted** immediately.

    The arq worker (run separately) processes the ticket in the background:
    DistilBERT inference → urgency score S∈[0,1] → Slack webhook if S > 0.8.

    Poll `GET /v2/tickets/{ticket_id}` to retrieve the result.
    """
    _require_redis()

    ticket_id   = str(uuid.uuid4())
    accepted_at = datetime.now(timezone.utc).isoformat()

    # Store initial pending record
    await _redis_conn.hset(
        _ticket_key(ticket_id),
        mapping={
            "status":      "pending",
            "subject":     payload.subject,
            "body":        payload.body,
            "accepted_at": accepted_at,
        },
    )
    await _redis_conn.expire(_ticket_key(ticket_id), 3600)

    # Enqueue arq job → worker.py::process_ticket
    await _arq_pool.enqueue_job(
        "process_ticket",
        ticket_id,
        payload.subject,
        payload.body,
    )

    logger.info("M2 accepted ticket %s → queued.", ticket_id)
    return TicketAccepted(ticket_id=ticket_id)


@app.get(
    "/v2/tickets/next",
    tags=["Milestone 2"],
    summary="Pop processed ticket with highest urgency score",
)
async def m2_next() -> dict:
    """ZPOPMAX from processed sorted set → highest urgency_score first."""
    _require_redis()
    result = await _redis_conn.zpopmax(_processed_set(), count=1)
    if not result:
        raise HTTPException(status_code=404, detail="No processed tickets.")

    ticket_id, score = result[0]
    data = await _redis_conn.hgetall(_ticket_key(ticket_id))
    return {"ticket_id": ticket_id, "urgency_score": float(score), **data}


@app.get(
    "/v2/tickets/status",
    response_model=QueueStatus,
    tags=["Milestone 2"],
    summary="Pending + processed queue depth",
)
async def m2_status() -> QueueStatus:
    _require_redis()
    # arq ≥ 0.25 stores jobs in a sorted set "arq:queue"; fallback for older list
    try:
        pending = await _redis_conn.zcard("arq:queue")
    except Exception:
        try:
            pending = await _redis_conn.llen("arq:queue")
        except Exception:
            pending = 0
    processed = await _redis_conn.zcard(_processed_set())
    return QueueStatus(pending=pending, processed=processed)


@app.get(
    "/v2/tickets/{ticket_id}",
    response_model=TicketResult,
    tags=["Milestone 2"],
    summary="Get async ticket result",
)
async def m2_get_ticket(ticket_id: str) -> TicketResult:
    """Fetch current state of an async ticket (pending → processed)."""
    _require_redis()
    data = await _redis_conn.hgetall(_ticket_key(ticket_id))
    if not data:
        return TicketResult(ticket_id=ticket_id, status="not_found")

    raw_score = data.get("urgency_score")
    raw_conf  = data.get("confidence")
    return TicketResult(
        ticket_id=ticket_id,
        subject=data.get("subject"),
        body=data.get("body"),
        status=data.get("status", "pending"),          # type: ignore[arg-type]
        category=data.get("category"),                  # type: ignore[arg-type]
        urgency_score=float(raw_score) if raw_score else None,
        confidence=float(raw_conf) if raw_conf else None,
        model_used=data.get("model_used"),
        processed_at=data.get("processed_at"),
    )
