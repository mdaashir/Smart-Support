# Smart-Support — Ticket Routing Engine

> **Hackathon Challenge** · System Design & NLP Track
> Intelligent ticket classification, urgency scoring, and priority-based routing

---

## Overview

Smart-Support is a high-throughput support ticket routing engine that:

- **Classifies** tickets into **Billing**, **Legal**, or **Technical** categories
- **Detects urgency** using regex heuristics (M1) and DistilBERT regression (M2)
- **Routes tickets** via in-memory priority queues (M1) and Redis async broker (M2)
- **Triggers webhooks** for high-urgency tickets (urgency score > 0.8)

---

## Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                      FastAPI Application                    │
│                                                            │
│  ┌──────── Milestone 1 ────────┐ ┌──── Milestone 2 ─────┐ │
│  │ POST /v1/tickets  → 201     │ │ POST /v2/tickets → 202│ │
│  │ GET  /v1/tickets/next       │ │ GET  /v2/tickets/{id} │ │
│  │ GET  /v1/tickets/peek       │ │ GET  /v2/tickets/next │ │
│  │ GET  /v1/tickets/status     │ │ GET  /v2/tickets/status│ │
│  │ DELETE /v1/tickets (drain)  │ │                       │ │
│  └──────────┬──────────────────┘ └───────────┬───────────┘ │
│             │                                 │             │
│  ┌──────────▼──────────┐      ┌───────────────▼──────────┐ │
│  │ TF-IDF + LinearSVC  │      │    Redis / arq broker    │ │
│  │ heapq priority queue│      │    DistilBERT + Ridge    │ │
│  └─────────────────────┘      │    Slack/Discord webhook │ │
│                                └──────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Language** | Python 3.13 |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **API Framework** | FastAPI + Uvicorn |
| **ML — M1** | scikit-learn (TfidfVectorizer + LinearSVC) |
| **ML — M2** | PyTorch + HuggingFace Transformers (DistilBERT) |
| **Async Broker** | Redis + arq |
| **HTTP Client** | httpx (webhook calls) |

---

## Project Structure

```text
Smart-Support/
├── api/
│   ├── __init__.py
│   ├── main.py            # FastAPI app (v1 sync + v2 async endpoints)
│   ├── schemas.py          # Pydantic request/response models
│   ├── queue_store.py      # Thread-safe in-memory heapq (M1)
│   └── worker.py           # arq background worker (M2)
├── dataset/
│   └── aa_dataset-tickets-multi-lang-5-2-50-version.csv  (28,587 rows)
├── model_cache/            # Auto-generated on first run
│   ├── m1.joblib           # Cached M1 model
│   └── m2.joblib           # Cached M2 classifiers
├── models.py               # M1 ML module (TF-IDF + SVM)
├── models_m2.py            # M2 ML module (DistilBERT embeddings)
├── models.ipynb            # Experimental notebook
├── pyproject.toml          # Dependencies & project config
├── .env.example            # Environment variables template
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python ≥ 3.13**
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker** (for Redis)

### 1. Install Dependencies

```bash
uv sync
```

### 2. Start Redis

```bash
docker run -d --name smart-support-redis -p 6379:6379 redis:7-alpine
```

### 3. Configure Environment (optional)

```bash
cp .env.example .env
# Edit .env to set WEBHOOK_URL for real Slack/Discord integration
```

### 4. Start the API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On first startup, M1 model trains automatically (~10s) and is cached to `model_cache/m1.joblib`.

### 5. Start the Background Worker (M2)

```bash
uv run arq api.worker.WorkerSettings
```

On first job, M2 model trains (~5 min on CPU) using DistilBERT embeddings and is cached to `model_cache/m2.joblib`.

### 6. Open Swagger Docs

Navigate to **<http://localhost:8000/docs>** for interactive API documentation.

---

## Milestone 1 — Minimum Viable Router

### M1 Model Details

- **Vectorizer**: TF-IDF with character n-grams (3–5), max 20k features
- **Classifier**: LinearSVC with balanced class weights
- **Urgency**: Regex heuristic — keywords like *asap*, *urgent*, *critical*, *broken*, *outage* → urgency=1 (HIGH), else urgency=5 (NORMAL)
- **Accuracy**: **95.7%** on held-out test set

### Category Mapping

| Dataset Queue | Category |
| --- | --- |
| Billing and Payments, Returns and Exchanges | **Billing** |
| Human Resources | **Legal** |
| Technical Support, Product Support, Customer Service, IT Support, Service Outages, Sales, General Inquiry | **Technical** |

### M1 Endpoints

| Method | Path | Status | Description |
| --- | --- | --- | --- |
| `POST` | `/v1/tickets` | 201 | Classify ticket synchronously, push to priority queue |
| `GET` | `/v1/tickets/next` | 200 | Pop highest-priority ticket (urgency=1 first) |
| `GET` | `/v1/tickets/peek` | 200 | Peek without dequeuing |
| `GET` | `/v1/tickets/status` | 200 | Queue depth |
| `DELETE` | `/v1/tickets` | 200 | Drain all tickets |

### M1 Example

```bash
# Submit a ticket
curl -X POST http://localhost:8000/v1/tickets \
  -H "Content-Type: application/json" \
  -d '{"subject": "Server broken", "body": "Production API returning 500 errors ASAP"}'

# Response (201)
{
  "ticket_id": "uuid-...",
  "subject": "Server broken",
  "body": "Production API returning 500 errors ASAP",
  "category": "Technical",
  "urgency_level": 1,
  "queued_at": "2026-02-21T15:00:00Z"
}

# Pop highest-priority ticket
curl http://localhost:8000/v1/tickets/next
```

---

## Milestone 2 — The Intelligent Queue

### M2 Model Details

- **Embeddings**: DistilBERT (`distilbert-base-uncased`) frozen [CLS] token embeddings (768-dim)
- **Classifier**: Logistic Regression on embeddings → Billing / Legal / Technical
- **Urgency Regressor**: Ridge regression → continuous score S ∈ [0, 1]
  - Training labels: `low → 0.0`, `medium → 0.5`, `high → 1.0`
- **Training**: 5,000-sample subset for CPU feasibility
- **Metrics**: Category accuracy ≈ 0.605, Urgency RMSE ≈ 0.384

### Async Architecture

1. **POST /v2/tickets** → returns **202 Accepted** immediately
2. Ticket is enqueued to **Redis** via **arq** job queue
3. **Background worker** picks up the job:
   - Acquires **atomic Redis lock** (SET NX, TTL=30s) to prevent duplicate processing
   - Runs DistilBERT inference (falls back to M1 if M2 unavailable)
   - Persists result to Redis hash + sorted set (ZADD with urgency score)
   - Triggers **Slack/Discord webhook** if urgency score > 0.8
   - Releases lock
4. **GET /v2/tickets/{id}** → poll for result (`pending` → `processed`)
5. **GET /v2/tickets/next** → ZPOPMAX (highest urgency score dequeued first)

### Concurrency & Reliability

- **Atomic locks**: `SET NX` with TTL prevents race conditions even with 10+ simultaneous requests
- **Graceful degradation**: If Redis is down, M2 endpoints return 503; M1 always works
- **M1 fallback**: If DistilBERT fails to load, worker automatically falls back to M1 model
- **Worker config**: `max_jobs=50`, `job_timeout=120s`, results kept for 1 hour

### M2 Endpoints

| Method | Path | Status | Description |
| --- | --- | --- | --- |
| `POST` | `/v2/tickets` | 202 | Accept ticket, enqueue for async processing |
| `GET` | `/v2/tickets/{id}` | 200 | Poll ticket status (pending/processed/not_found) |
| `GET` | `/v2/tickets/next` | 200 | ZPOPMAX — pop highest urgency score |
| `GET` | `/v2/tickets/status` | 200 | Pending + processed counts |

### M2 Example

```bash
# Submit ticket (async)
curl -X POST http://localhost:8000/v2/tickets \
  -H "Content-Type: application/json" \
  -d '{"subject": "Critical system failure", "body": "All servers down ASAP"}'

# Response (202)
{
  "ticket_id": "uuid-...",
  "status": "accepted",
  "message": "Ticket queued for processing.",
  "accepted_at": "2026-02-21T15:00:00Z"
}

# Poll for result
curl http://localhost:8000/v2/tickets/<ticket_id>

# Response (processed)
{
  "ticket_id": "uuid-...",
  "subject": "Critical system failure",
  "body": "All servers down ASAP",
  "status": "processed",
  "category": "Technical",
  "urgency_score": 0.9431,
  "model_used": "m2",
  "processed_at": "2026-02-21T15:00:25Z"
}

# Get highest-urgency processed ticket
curl http://localhost:8000/v2/tickets/next
```

---

## Shared Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe — M1 status, Redis/M2 broker status, queue depths |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

---

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `WEBHOOK_URL` | *(empty — mock mode)* | Slack/Discord incoming webhook URL |
| `WEBHOOK_THRESHOLD` | `0.8` | Urgency score above which webhook fires |

---

## Dataset

The training dataset contains **28,587 multilingual support tickets** with 10 queue categories and 3 priority levels:

| Queue | Count |
| --- | --- |
| Technical Support | 8,362 |
| Product Support | 5,252 |
| Customer Service | 4,268 |
| IT Support | 3,433 |
| Billing and Payments | 2,788 |
| Returns and Exchanges | 1,437 |
| Service Outages | 1,148 |
| Sales and Pre-Sales | 918 |
| Human Resources | 576 |
| General Inquiry | 405 |

---

## Development

### Run Tests

```bash
uv run pytest
```

### Train Models Manually

```bash
# M1 (TF-IDF + SVM)
uv run python models.py

# M2 (DistilBERT + LogisticRegression + Ridge)
uv run python models_m2.py
```

### Clear Model Cache

```bash
rm -rf model_cache/
```

---

## License

See [LICENSE](LICENSE) for details.
