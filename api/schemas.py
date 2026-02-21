"""Pydantic schemas for the Smart-Support API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
import uuid

from pydantic import BaseModel, Field


class TicketIn(BaseModel):
    subject: str = Field(..., min_length=1, max_length=500, examples=["Server down"])
    body: str = Field(..., min_length=1, max_length=5000, examples=["Production API returning 500 errors ASAP"])


class TicketOut(BaseModel):
    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    body: str
    category: Literal["Billing", "Legal", "Technical"]
    urgency_level: Literal[1, 5]     # 1 = HIGH, 5 = NORMAL
    queued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class QueueStatus(BaseModel):
    queued: int
    message: str = "ok"
