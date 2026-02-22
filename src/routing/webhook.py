"""Milestone 2 — Mock Slack / Discord webhook for high-urgency tickets.

When urgency score S > ``URGENCY_WEBHOOK_THRESHOLD`` (0.8), fires a POST
to the configured webhook URL.  Uses ``aiohttp`` for non-blocking HTTP.
In test / demo mode, logs the payload instead of making a real call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from src.config import URGENCY_WEBHOOK_THRESHOLD, WEBHOOK_URL

logger = logging.getLogger(__name__)

# ── In-memory audit log (for demo / testing) ────────────────────────────
_webhook_log: list[dict] = []


async def fire_webhook(
    ticket_id: str,
    category: str,
    urgency_score: float,
    text_preview: str = "",
    *,
    url: str = WEBHOOK_URL,
) -> dict:
    """Send a webhook notification for urgent tickets.

    Returns the payload dict (always — even if the HTTP call is mocked).
    """
    payload = {
        "text": (
            f":rotating_light: **High-Urgency Ticket** :rotating_light:\n"
            f"• **ID**: {ticket_id}\n"
            f"• **Category**: {category}\n"
            f"• **Urgency Score**: {urgency_score:.2f}\n"
            f"• **Preview**: {text_preview[:200]}\n"
            f"• **Timestamp**: {datetime.now(timezone.utc).isoformat()}"
        ),
    }

    _webhook_log.append(payload)

    if "MOCK" in url:
        logger.info(
            "[MOCK WEBHOOK] ticket=%s  score=%.2f  category=%s",
            ticket_id, urgency_score, category,
        )
    else:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    logger.info(
                        "Webhook sent → %s  status=%d", url, resp.status,
                    )
        except Exception:
            logger.exception("Webhook call failed — payload logged locally")

    return payload


def get_webhook_log() -> list[dict]:
    """Return the in-memory webhook audit log (for demo / testing)."""
    return list(_webhook_log)


def clear_webhook_log() -> None:
    _webhook_log.clear()
