<div align="center">

# Smart-Support

**AI-powered multilingual customer support ticket routing system**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Automatically classify, prioritise, deduplicate and route 61 000+ real multilingual (EN/DE) support tickets using progressively advanced ML models across three milestones.*

</div>

---

## What It Does

| Input | Output |
|-------|--------|
| A customer ticket (subject + body, English or German) | **Category** (Billing / Technical / Legal) + **Urgency Score** S in [0,1] + **Agent Assignment** |

Smart-Support routes incoming support tickets through:

1. **Classification** — predicts which department should handle the ticket (3 categories per PDF spec)
2. **Urgency scoring** — regression model produces continuous S in [0,1], flags critical via webhook when S > 0.8
3. **Semantic deduplication** — sentence embeddings detect duplicate floods, creates Master Incidents
4. **Circuit breaker** — monitors transformer latency, fails over to fast M1 model if > 500ms
5. **Skill-based routing** — assigns tickets to agents via skill-affinity vectors + capacity constraints
6. **Async processing** — 202 Accepted pattern with background workers and atomic locks

---

## Architecture

```
                                    ┌──────────────────┐
                                    │  Semantic Dedup   │
                                    │  (sentence-BERT)  │
                                    └────────┬─────────┘
                                             │
┌──────────────┐    ┌───────────────┐    ┌───▼──────────┐    ┌────────────┐
│  FastAPI API  │───▶│  Async Broker │───▶│  Classifier  │───▶│   Skill    │
│  /route       │    │  (asyncio Q)  │    │  (LogReg /   │    │   Router   │
│  /route/async │    │  202 Accepted │    │   SVC /      │    │  (agents)  │
│  /route/batch │    │  atomic locks │    │   DistilBERT)│    └────────────┘
│  /stats       │    └───────────────┘    └──────┬───────┘
│  /incidents   │                                │
│  /agents      │                         ┌──────▼───────┐
└──────────────┘                          │  Circuit      │
                                          │  Breaker      │
                                          │  (500ms fail) │
                                          └──────┬───────┘
                                          ┌──────▼───────┐
                                          │  Urgency      │
                                          │  Regressor    │
                                          │  S ∈ [0,1]    │
                                          └──────┬───────┘
                                          ┌──────▼───────┐
                                          │  Webhook      │
                                          │  (S > 0.8)    │
                                          └──────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
# Requires Python 3.13+ and uv (https://docs.astral.sh/uv/)
uv sync
```

### 2. Train models

```bash
# Milestone 1 — real data + Logistic Regression (with 5-fold CV)
python -m backend.scripts.train --milestone 1

# Milestone 2 — real data + LinearSVC + urgency regressor
python -m backend.scripts.train --milestone 2

# Milestone 3 — real data + DistilBERT (multilingual)
python -m backend.scripts.train --milestone 3

# Or train everything
python -m backend.scripts.train --milestone all
```

### 3. Launch the API

```bash
# Choose model: logreg | svc | distilbert
MODEL_VARIANT=svc uvicorn backend.api.main:app --reload

# Open the dashboard at http://localhost:8000/ui
```

### 4. Route a ticket (sync)

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"subject": "Dringend: Server ausgefallen", "body": "Seit heute Morgen ist der Produktionsserver nicht erreichbar."}'
```

Response:
```json
{
  "category": "Technical",
  "urgency": "1(HIGH)",
  "urgency_score": 0.87,
  "model_used": "svc",
  "agent": "Agent_A",
  "dedup": {"is_duplicate": false, "master_incident_id": null, "similarity": 0.0}
}
```

### 5. Route a ticket (async — 202 Accepted)

```bash
curl -X POST http://localhost:8000/route/async \
  -H "Content-Type: application/json" \
  -d '{"subject": "billing error", "body": "charged twice"}'
# → 202: {"job_id": "abc-123", "status": "accepted", ...}

curl http://localhost:8000/route/async/abc-123
# → {"job_id": "abc-123", "status": "done", "result": {...}}
```

---

## Dataset

Uses the **open-source** [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset from HuggingFace Hub:

| Stat | Value |
|------|-------|
| Total tickets | 61 765 |
| Languages | English, German |
| Queues (raw) | 10+ department queues |
| Mapped categories | **3 (Billing, Technical, Legal)** — per PDF spec |
| Priority levels | critical, high, medium, low, very_low |
| Tags per ticket | up to 8 |

The dataset is automatically downloaded and cached on first use. **No synthetic data** — all models train exclusively on real-world tickets with SMOTE augmentation available for class imbalance.

---

## Anti-Overfitting Strategy

| Technique | Implementation |
|-----------|---------------|
| **Stratified K-Fold CV** | 5-fold cross-validation logged during training |
| **Regularisation** | C=1.0 for LogReg, SVC; Ridge alpha=1.0 for urgency |
| **Class balancing** | `class_weight="balanced"` on all classifiers |
| **SMOTE augmentation** | Available in `backend/data/augmentation.py` for imbalanced splits |
| **No synthetic templates** | Removed template-based generation — avoids trivial separation |
| **Real data only** | All milestones train on 61K+ real multilingual tickets |

---

## Project Structure

```
Smart-Support/
├── frontend/
│   └── index.html                 # Mission-control dashboard UI
├── backend/
│   ├── config.py                  # Centralised config (3 categories, M2/M3 params)
│   ├── api/
│   │   └── main.py                # FastAPI REST API (all milestones)
│   ├── models/
│   │   ├── tfidf_logreg.py        # M1: TF-IDF + Logistic Regression
│   │   ├── tfidf_svc.py           # M2: Char n-gram TF-IDF + LinearSVC
│   │   └── distilbert_classifier.py  # M3: DistilBERT embeddings + LogReg
│   ├── data/
│   │   ├── dataset_loader.py      # HuggingFace download, caching, feature engineering
│   │   └── augmentation.py        # SMOTE on TF-IDF space for class balance
│   ├── preprocessing/
│   │   └── text_cleaner.py        # Shared text normalisation
│   ├── routing/
│   │   ├── urgency.py             # M1: Regex-based urgency (EN + DE)
│   │   ├── urgency_regressor.py   # M2: Regression model S ∈ [0,1]
│   │   ├── broker.py              # M2: Async broker (202 Accepted, atomic locks)
│   │   ├── webhook.py             # M2: Mock Slack/Discord webhook (S > 0.8)
│   │   ├── queue.py               # M1: Min-heap priority queue
│   │   ├── router.py              # Core ticket router
│   │   ├── deduplicator.py        # M3: Sentence embeddings + cosine similarity
│   │   ├── circuit_breaker.py     # M3: Latency failover (500ms threshold)
│   │   └── skill_router.py        # M3: Agent skill vectors + constraint routing
│   ├── evaluation/
│   │   └── evaluator.py           # Metrics + confusion matrix artifacts
│   └── scripts/
│       └── train.py               # CLI training (real data, CV, all milestones)
├── tests/
│   ├── test_synthetic.py          # M1 tests (real data, CV, urgency, queue)
│   ├── test_svc.py                # M2 tests (SVC, urgency regressor)
│   ├── test_distilbert.py         # M3 smoke tests (embedder, classifier)
│   ├── test_m2_m3.py              # M2/M3 component tests (broker, dedup, CB, webhook)
│   └── test_api.py                # API endpoint tests (sync, async, batch, stats)
├── data/                          # Cached HuggingFace dataset (auto-downloaded)
├── saved_models/                  # Trained model artifacts (.joblib)
├── pyproject.toml
└── README.md
```

---

## Milestone Progression

### Milestone 1 — MVR (Minimum Viable Router)
- **Data**: 61 765 real multilingual tickets → 3 categories
- **Model**: TF-IDF (word bigrams, 15K features) → Logistic Regression (C=1.0, balanced)
- **Features**: Regex urgency, in-memory priority queue, REST API
- **Anti-overfit**: 5-fold stratified CV, class balancing

### Milestone 2 — Intelligent Queue
- **Model**: Char n-gram TF-IDF (3–5, 20K features) → LinearSVC
- **Urgency**: Regression model S ∈ [0,1] (Ridge on TF-IDF + keyword blend)
- **Async**: 202 Accepted pattern, asyncio broker, 4 background workers, atomic locks
- **Webhook**: Mock Slack/Discord notification when S > 0.8

### Milestone 3 — Autonomous Orchestrator
- **Model**: `distilbert-base-multilingual-cased` [CLS] embeddings → Logistic Regression
- **Deduplication**: sentence-transformers (`all-MiniLM-L6-v2`), cosine similarity > 0.9, Master Incident when 10+ in 5 min
- **Circuit Breaker**: Monitors transformer latency, failover to M1 if > 500ms
- **Skill Routing**: Agent skill vectors + capacity-constrained greedy assignment

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model status, categories, circuit breaker state, uptime |
| `POST` | `/route` | Route a single ticket (sync) |
| `POST` | `/route/async` | **202 Accepted** — queue for async processing |
| `GET` | `/route/async/{job_id}` | Poll async job result |
| `POST` | `/route/batch` | Route up to 100 tickets in one request |
| `GET` | `/stats` | Live stats (counts, webhook fires, master incidents) |
| `GET` | `/incidents` | List semantic dedup master incidents |
| `GET` | `/agents` | Agent load and utilisation |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Running Tests

```bash
# Run all tests
uv run pytest -v

# Run specific milestone tests
uv run pytest tests/test_synthetic.py -v   # M1
uv run pytest tests/test_svc.py -v         # M2
uv run pytest tests/test_distilbert.py -v  # M3
uv run pytest tests/test_m2_m3.py -v       # M2/M3 components
uv run pytest tests/test_api.py -v         # API
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Package manager | uv |
| ML (traditional) | scikit-learn (TF-IDF, LogReg, LinearSVC, Ridge) |
| ML (deep learning) | PyTorch + HuggingFace Transformers (DistilBERT) |
| Sentence embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Data augmentation | imbalanced-learn (SMOTE) |
| Data | HuggingFace Datasets, pandas |
| API | FastAPI + Pydantic v2 |
| Async | asyncio (broker, workers, atomic locks) |
| Webhook | aiohttp (mock Slack/Discord) |
| Visualisation | matplotlib + seaborn |
| Testing | pytest |

---

## License

[MIT](LICENSE)
