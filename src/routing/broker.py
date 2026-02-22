"""Milestone 2 — Async broker with 202 Accepted pattern.

Provides an asyncio-based in-memory broker that:
* Accepts tickets immediately (202 Accepted)
* Queues them for background processing via asyncio.Queue
* Supports atomic ticket ingestion with asyncio.Lock
* Falls back gracefully — no external Redis required at runtime
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class TicketJob:
    """Represents a queued ticket awaiting classification."""
    job_id: str
    subject: str
    body: str
    status: str = "pending"          # pending → processing → done | failed
    result: dict | None = None


class AsyncBroker:
    """In-memory async message broker (Milestone 2).

    * ``submit()`` → returns immediately with ``job_id``
    * Background worker dequeues, classifies, and stores results
    * ``get_result()`` → poll for completion
    * ``_lock`` ensures atomic ingestion under 10+ concurrent requests
    """

    def __init__(self, max_queue: int = 1000) -> None:
        self._queue: asyncio.Queue[TicketJob] = asyncio.Queue(maxsize=max_queue)
        self._results: dict[str, TicketJob] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task] = []
        self._process_fn: Callable[..., Coroutine] | None = None

    # ── Setup ────────────────────────────────────────────────────────

    def register_processor(self, fn: Callable[..., Coroutine]) -> None:
        """Register the async function that processes each ticket."""
        self._process_fn = fn

    async def start_workers(self, n: int = 4) -> None:
        """Launch *n* background worker coroutines."""
        for i in range(n):
            t = asyncio.create_task(self._worker(i))
            self._workers.append(t)
        logger.info("Started %d async workers", n)

    async def stop_workers(self) -> None:
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ── Submit (atomic) ──────────────────────────────────────────────

    async def submit(self, subject: str, body: str) -> str:
        """Atomically create a job and enqueue it. Returns ``job_id``."""
        async with self._lock:
            job = TicketJob(
                job_id=str(uuid.uuid4()),
                subject=subject,
                body=body,
            )
            self._results[job.job_id] = job
            await self._queue.put(job)
        logger.debug("Enqueued job %s", job.job_id)
        return job.job_id

    # ── Result retrieval ─────────────────────────────────────────────

    def get_result(self, job_id: str) -> TicketJob | None:
        return self._results.get(job_id)

    # ── Worker loop ──────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        logger.debug("Worker %d started", worker_id)
        while True:
            job = await self._queue.get()
            try:
                job.status = "processing"
                if self._process_fn:
                    job.result = await self._process_fn(job.subject, job.body)
                    job.status = "done"
                else:
                    job.status = "failed"
                    job.result = {"error": "no processor registered"}
            except Exception as exc:
                logger.exception("Worker %d failed on job %s", worker_id, job.job_id)
                job.status = "failed"
                job.result = {"error": str(exc)}
            finally:
                self._queue.task_done()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def total_jobs(self) -> int:
        return len(self._results)
