"""
api/main.py — Smart-Support FastAPI (Milestone 1)

Endpoints:
    POST   /tickets          Ingest a ticket → route + enqueue → return TicketOut
    GET    /tickets/next     Pop the highest-priority ticket from the queue
    GET    /tickets/peek     Peek at the queue head without dequeuing
    GET    /tickets/status   Queue depth
    DELETE /tickets          Drain (clear) the entire queue
    GET    /health           Liveness check + model status

Run with:
    uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# ── local imports ──
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import models as ml_models
from api import queue_store
from api.schemas import QueueStatus, TicketIn, TicketOut

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Lifespan: warm up the model before accepting traffic
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up ML model …")
    ml_models._ensure_loaded()          # triggers training / cache load
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Smart-Support Ticket Routing Engine",
    description=(
        "Milestone 1 MVP — classifies support tickets into Billing / Legal / Technical "
        "and manages an in-memory priority queue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "model":  "m1-ready",
        "queue_depth": queue_store.size(),
    }


@app.post(
    "/tickets",
    response_model=TicketOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Tickets"],
    summary="Ingest a ticket",
)
def ingest_ticket(payload: TicketIn) -> TicketOut:
    """
    Classify a support ticket and add it to the priority queue.

    - **subject** / **body**: raw ticket text (multilingual OK)
    - Returns the enriched ticket with `category` and `urgency_level`
    """
    result   = ml_models.route_ticket(payload.subject, payload.body)
    ticket   = TicketOut(
        subject=payload.subject,
        body=payload.body,
        category=result["category"],
        urgency_level=result["urgency_level"],
    )
    queue_store.push(ticket)
    logger.info(
        "Ingested ticket %s  category=%s  urgency=%s",
        ticket.ticket_id, ticket.category, ticket.urgency_level,
    )
    return ticket


@app.get(
    "/tickets/next",
    tags=["Queue"],
    summary="Pop next ticket (highest priority)",
)
def next_ticket() -> dict:
    """Remove and return the highest-priority ticket."""
    ticket = queue_store.pop()
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue is empty.",
        )
    logger.info("Processed ticket %s", ticket.get("ticket_id"))
    return ticket


@app.get(
    "/tickets/peek",
    tags=["Queue"],
    summary="Peek at next ticket without dequeuing",
)
def peek_ticket() -> dict:
    """Return the next ticket without removing it."""
    ticket = queue_store.peek()
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Queue is empty.",
        )
    return ticket


@app.get(
    "/tickets/status",
    response_model=QueueStatus,
    tags=["Queue"],
    summary="Queue depth",
)
def queue_status() -> QueueStatus:
    return QueueStatus(queued=queue_store.size())


@app.delete(
    "/tickets",
    tags=["Queue"],
    summary="Drain the entire queue",
)
def drain_queue() -> dict:
    """Clear all pending tickets (useful for testing)."""
    tickets = queue_store.drain()
    return {"drained": len(tickets), "tickets": tickets}
