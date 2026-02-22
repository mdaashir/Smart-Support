"""
api/worker.py — Milestone 2 async broker worker (arq + Redis)

Start with:
    uv run arq api.worker.WorkerSettings

Responsibilities:
  1. Read jobs from the Redis queue (arq-enqueued by main.py)
  2. Acquire an atomic Redis lock per ticket_id (prevents duplicate processing
     even if 10+ simultaneous requests arrive at the same millisecond)
  3. Run M2 inference (DistilBERT + Ridge); fall back to M1 if M2 not ready
  4. Persist the result back into Redis
  5. Trigger Slack/Discord mock webhook when urgency_score > 0.8
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import httpx
from arq.connections import RedisSettings

# ── local imports ─────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import models as ml_m1           # Milestone-1 fallback
import models_m2 as ml_m2        # Milestone-2 transformer model

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
REDIS_URL         = os.getenv("REDIS_URL",   "redis://localhost:6379")
WEBHOOK_URL       = os.getenv("WEBHOOK_URL", "")          # empty = mock-only
WEBHOOK_THRESHOLD = float(os.getenv("WEBHOOK_THRESHOLD", "0.8"))
LOCK_TTL_SECONDS  = 30    # atomic lock TTL
RESULT_TTL_HOURS  = 1     # how long to keep processed results in Redis

# Redis key helpers
def _ticket_key(tid: str) -> str:      return f"ticket:{tid}"
def _lock_key(tid: str) -> str:        return f"lock:ticket:{tid}"
def _processed_set() -> str:           return "processed_tickets"


# ── startup / shutdown hooks (shared ctx) ─────────────────────────────────────
async def startup(ctx: dict) -> None:
    """Called once when the worker process starts."""
    logger.info("Worker starting — warming up M1 model …")
    ml_m1._ensure_loaded()
    logger.info("M1 ready.  M2 will load lazily on first job.")
    # reuse a single httpx client for all jobs
    ctx["http"] = httpx.AsyncClient(timeout=10.0)


async def shutdown(ctx: dict) -> None:
    await ctx["http"].aclose()
    logger.info("Worker shut down cleanly.")


# ── core job ──────────────────────────────────────────────────────────────────
async def process_ticket(
    ctx: dict,
    ticket_id: str,
    subject: str,
    body: str,
) -> dict:
    """
    arq job — invoked by the FastAPI layer.

    Steps:
      1. Acquire atomic Redis lock (SET NX) → skip if duplicate
      2. Run ML inference (M2 or M1 fallback)
      3. Persist result to Redis hash + sorted set
      4. Trigger webhook if urgency_score > WEBHOOK_THRESHOLD
      5. Release lock
    """
    redis       = ctx["redis"]
    http_client: httpx.AsyncClient = ctx["http"]

    # ── 1. Atomic lock (prevents race conditions / duplicate processing) ──────
    lock_key = _lock_key(ticket_id)
    acquired = await redis.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS)
    if not acquired:
        logger.warning("Ticket %s already locked — duplicate skipped.", ticket_id)
        return {"status": "duplicate_skipped", "ticket_id": ticket_id}

    try:
        # ── 2. ML inference ──────────────────────────────────────────────────
        if ml_m2.m2_ready():
            result     = ml_m2.predict_ticket_m2(subject, body)
            model_used = "m2"
        else:
            # Try to load M2 (non-blocking attempt; if still loading, use M1)
            try:
                ml_m2.ensure_loaded()
                result     = ml_m2.predict_ticket_m2(subject, body)
                model_used = "m2"
            except Exception as exc:
                logger.warning("M2 load failed (%s) — using M1 fallback.", exc)
                m1 = ml_m1.route_ticket(subject, body)
                result = {
                    "category":      m1["category"],
                    "urgency_score": 1.0 if m1["urgency_level"] == 1 else 0.3,
                    "confidence":    None,
                }
                model_used = "m1-fallback"

        category      = result["category"]
        urgency_score = result["urgency_score"]
        confidence    = result.get("confidence", None)
        processed_at  = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Ticket %s → category=%s  S=%.3f  conf=%s  model=%s",
            ticket_id, category, urgency_score,
            f"{confidence:.2f}" if confidence is not None else "n/a",
            model_used,
        )

        # ── 3. Persist result ────────────────────────────────────────────────
        ticket_key = _ticket_key(ticket_id)
        mapping = {
            "status":        "processed",
            "category":      category,
            "urgency_score": str(urgency_score),
            "model_used":    model_used,
            "processed_at":  processed_at,
        }
        if confidence is not None:
            mapping["confidence"] = str(round(confidence, 4))
        await redis.hset(ticket_key, mapping=mapping)
        await redis.expire(ticket_key, RESULT_TTL_HOURS * 3600)

        # Add to processed sorted set (score = urgency_score → ZPOPMAX gives
        # highest-urgency ticket first, matching Milestone-1 priority queue)
        await redis.zadd(_processed_set(), {ticket_id: urgency_score})

        # ── 4. Webhook when S > threshold ────────────────────────────────────
        if urgency_score > WEBHOOK_THRESHOLD:
            await _trigger_webhook(
                http_client, ticket_id, subject, category, urgency_score
            )

        return {
            "status":        "processed",
            "ticket_id":     ticket_id,
            "category":      category,
            "urgency_score": urgency_score,
            "model_used":    model_used,
        }

    finally:
        # ── 5. Always release lock ────────────────────────────────────────────
        await redis.delete(lock_key)


# ── webhook helper ─────────────────────────────────────────────────────────────
async def _trigger_webhook(
    client: httpx.AsyncClient,
    ticket_id: str,
    subject: str,
    category: str,
    score: float,
) -> None:
    payload = {
        "text": f":rotating_light: *High-urgency ticket detected!* (S={score:.3f})",
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {"title": "Ticket ID", "value": ticket_id,        "short": True},
                    {"title": "Category",  "value": category,          "short": True},
                    {"title": "Score",     "value": f"{score:.4f}",    "short": True},
                    {"title": "Subject",   "value": subject,           "short": False},
                ],
            }
        ],
    }

    if WEBHOOK_URL:
        try:
            resp = await client.post(WEBHOOK_URL, json=payload)
            logger.info("Webhook → HTTP %s | ticket=%s", resp.status_code, ticket_id)
        except Exception as exc:
            logger.error("Webhook failed for %s: %s", ticket_id, exc)
    else:
        # Mock: just log the would-be payload
        logger.info(
            "MOCK WEBHOOK | ticket=%s | category=%s | S=%.4f | subject=%r",
            ticket_id, category, score, subject,
        )


# ── arq WorkerSettings ────────────────────────────────────────────────────────
class WorkerSettings:
    functions  = [process_ticket]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs   = 50          # handle 10+ simultaneous with headroom
    job_timeout = 120        # seconds before a job is considered failed
    keep_result = 3600       # keep arq job result for 1 h
