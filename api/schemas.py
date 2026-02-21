"""Pydantic schemas for the Smart-Support API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
import uuid

from pydantic import BaseModel, Field


# ── input ─────────────────────────────────────────────────────────────────────

class TicketIn(BaseModel):
    subject: str = Field(..., min_length=1, max_length=500,  examples=["Server down"])
    body:    str = Field(..., min_length=1, max_length=5000, examples=["Production API returning 500 errors ASAP"])


# ── Milestone 1 — sync response ───────────────────────────────────────────────

class TicketOut(BaseModel):
    """Synchronous M1 response (201 Created)."""
    ticket_id:     str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject:       str
    body:          str
    category:      Literal["Billing", "Legal", "Technical"]
    urgency_level: Literal[1, 5]   # 1 = HIGH, 5 = NORMAL
    queued_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Milestone 2 — async 202 response ─────────────────────────────────────────

class TicketAccepted(BaseModel):
    """Async M2 response (202 Accepted) — returned immediately."""
    ticket_id:  str
    status:     Literal["accepted"] = "accepted"
    message:    str = "Ticket queued for processing."
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TicketResult(BaseModel):
    """
    Async ticket result retrieved via GET /tickets/{ticket_id}.
    status is 'pending' until the worker finishes.
    """
    ticket_id:     str
    subject:       Optional[str]   = None
    body:          Optional[str]   = None
    status:        Literal["pending", "processed", "not_found"]
    category:      Optional[Literal["Billing", "Legal", "Technical"]] = None
    urgency_score: Optional[float] = Field(None, ge=0.0, le=1.0,
                                           description="Continuous score S ∈ [0,1]")
    model_used:    Optional[str]   = None
    processed_at:  Optional[str]   = None


# ── shared ────────────────────────────────────────────────────────────────────

class QueueStatus(BaseModel):
    pending:   int
    processed: int
    message:   str = "ok"
