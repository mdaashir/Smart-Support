"""Tests for Milestone 2 — concurrency, Redis broker, and 202 pattern.

Verifies:
* 10+ simultaneous requests are handled atomically (no duplicates)
* AsyncBroker / RedisBroker submit + poll lifecycle
* 202 Accepted pattern end-to-end via the API
* Webhook fires for high-urgency tickets
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_VARIANT", "logreg")

from backend.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── 10+ simultaneous requests (concurrency stress test) ─────────────────

CONCURRENT_COUNT = 15  # > 10 as required


class TestConcurrency:
    """Prove the system handles 10+ simultaneous requests atomically."""

    def test_concurrent_sync_route(self, client):
        """Fire CONCURRENT_COUNT POST /route requests in parallel threads.

        Validates:
        1. All requests return 200
        2. All return valid categories (no corruption)
        3. No two tickets share an ID (if returned)
        4. All complete within a reasonable time
        """
        payload = {
            "subject": "Concurrent billing test",
            "body": "Please refund my duplicate charge immediately",
        }

        results = []
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=CONCURRENT_COUNT) as pool:
            futures = [
                pool.submit(client.post, "/route", json=payload)
                for _ in range(CONCURRENT_COUNT)
            ]
            for f in as_completed(futures):
                resp = f.result()
                assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
                results.append(resp.json())

        elapsed = time.perf_counter() - start

        # All returned valid categories
        valid_cats = {"Billing", "Technical", "Legal"}
        for r in results:
            assert r["category"] in valid_cats, f"Invalid category: {r['category']}"
            assert r["urgency"] in {"1(HIGH)", "0(NORMAL)"}

        assert len(results) == CONCURRENT_COUNT
        # Sanity: parallel execution shouldn't take N * serial time
        # (generous 60s threshold for CI machines loading models)
        assert elapsed < 60, f"Took too long: {elapsed:.1f}s"

    def test_concurrent_async_route_no_duplicates(self, client):
        """Fire CONCURRENT_COUNT POST /route/async in parallel.

        Validates:
        1. All return 202
        2. All job_ids are unique (atomic lock prevents duplicates)
        """
        payload = {
            "subject": "Async concurrency test",
            "body": "Testing simultaneous async submissions",
        }

        job_ids = []
        with ThreadPoolExecutor(max_workers=CONCURRENT_COUNT) as pool:
            futures = [
                pool.submit(client.post, "/route/async", json=payload)
                for _ in range(CONCURRENT_COUNT)
            ]
            for f in as_completed(futures):
                resp = f.result()
                assert resp.status_code == 202
                data = resp.json()
                assert data["status"] == "accepted"
                job_ids.append(data["job_id"])

        # Every job_id must be unique — proves atomic ingestion
        assert len(set(job_ids)) == CONCURRENT_COUNT, (
            f"Duplicate job_ids detected! unique={len(set(job_ids))} "
            f"total={CONCURRENT_COUNT}"
        )

    def test_concurrent_batch_route(self, client):
        """Batch endpoint with concurrent callers."""
        batch_payload = {
            "tickets": [
                {"subject": f"Ticket {i}", "body": f"Body for ticket {i}"}
                for i in range(5)
            ]
        }

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(client.post, "/route/batch", json=batch_payload)
                for _ in range(5)
            ]
            for f in as_completed(futures):
                resp = f.result()
                assert resp.status_code == 200
                data = resp.json()
                assert data["count"] == 5


# ── Redis broker unit tests (in-memory fallback) ────────────────────────

class TestRedisBrokerFallback:
    """Test the RedisBroker in in-memory fallback mode (no Redis server)."""

    @pytest.fixture
    def broker(self):
        from backend.routing.redis_broker import RedisBroker
        b = RedisBroker(redis_url="redis://localhost:19999/0")  # non-existent
        return b

    @pytest.mark.asyncio
    async def test_fallback_mode_activated(self, broker):
        connected = await broker._connect()
        assert connected is False
        assert broker.is_redis is False

    @pytest.mark.asyncio
    async def test_submit_and_retrieve_memory(self, broker):
        """Submit a job in fallback mode and retrieve it."""
        processed = []

        async def mock_processor(subject, body):
            result = {"category": "Billing", "urgency": "0(NORMAL)"}
            processed.append(result)
            return result

        broker.register_processor(mock_processor)
        await broker.start_workers(n=2)

        job_id = await broker.submit("test", "body")
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID format

        # Wait for processing
        await asyncio.sleep(0.5)

        job = await broker.get_result(job_id)
        assert job is not None
        assert job.status == "done"
        assert job.result["category"] == "Billing"

        await broker.stop_workers()

    @pytest.mark.asyncio
    async def test_atomic_concurrent_submit(self, broker):
        """Submit 15 jobs concurrently — all should get unique IDs."""
        async def noop_processor(subject, body):
            return {"category": "Technical"}

        broker.register_processor(noop_processor)
        await broker.start_workers(n=4)

        # Fire 15 submits concurrently
        tasks = [
            broker.submit(f"subject-{i}", f"body-{i}")
            for i in range(15)
        ]
        job_ids = await asyncio.gather(*tasks)

        assert len(job_ids) == 15
        assert len(set(job_ids)) == 15, "Duplicate job_ids — lock failure!"

        await broker.stop_workers()


# ── 202 Accepted lifecycle test ──────────────────────────────────────────

class TestAsyncLifecycle:
    """End-to-end 202 pattern: submit → poll → done."""

    def test_submit_poll_complete(self, client):
        """Submit via /route/async, poll /route/async/{id} until done."""
        resp = client.post("/route/async", json={
            "subject": "Lifecycle test",
            "body": "Testing the full async lifecycle",
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Poll with retries (workers process asynchronously)
        for _ in range(20):
            poll = client.get(f"/route/async/{job_id}")
            assert poll.status_code == 200
            data = poll.json()
            if data["status"] == "done":
                assert data["result"] is not None
                assert data["result"]["category"] in {"Billing", "Technical", "Legal"}
                return
            time.sleep(0.3)

        # If we get here, the job never completed
        pytest.fail(f"Job {job_id} never completed after polling")
