<div align="center">

# 🧠 Smart-Support

**AI-powered multilingual customer support ticket routing system**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Automatically classify, prioritise and route 61 000+ real multilingual (EN/DE) support tickets using progressively advanced ML models.*

</div>

---

## ✨ What It Does

| Input | Output |
|-------|--------|
| A customer ticket (subject + body, English or German) | **Category** (Billing / Technical / HR / General) + **Urgency** (HIGH / NORMAL) |

Smart-Support routes incoming support tickets through:

1. **Classification** — predicts which department should handle the ticket
2. **Urgency detection** — flags critical keywords in EN and DE
3. **Priority queuing** — processes urgent tickets first via a min-heap

---

## 🏗️ Architecture

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  FastAPI API  │───▶│  Router       │───▶│  Classifier  │
│  /route       │    │  (urgency +   │    │  (LogReg /   │
│  /route/batch │    │   combine)    │    │   SVC /      │
│  /stats       │    └───────────────┘    │   DistilBERT)│
└──────────────┘                          └──────────────┘
                                                │
                                          ┌─────▼──────┐
                                          │  Priority   │
                                          │  Queue      │
                                          │  (min-heap) │
                                          └────────────┘
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Requires Python 3.13+ and uv (https://docs.astral.sh/uv/)
uv sync
```

### 2. Train models

```bash
# Milestone 1 — synthetic data + Logistic Regression
python -m scripts.train --milestone 1

# Milestone 2 — real HuggingFace data + LinearSVC
python -m scripts.train --milestone 2

# Milestone 3 — real data + DistilBERT (multilingual)
python -m scripts.train --milestone 3

# Or train everything
python -m scripts.train --milestone all
```

### 3. Launch the API

```bash
# Choose model: logreg | svc | distilbert
MODEL_VARIANT=svc uvicorn api.main:app --reload
```

### 4. Route a ticket

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
  "model_used": "svc"
}
```

---

## 📊 Dataset

Uses the **open-source** [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset from HuggingFace Hub:

| Stat | Value |
|------|-------|
| Total tickets | 61 765 |
| Languages | English, German |
| Queues (raw) | 10+ department queues |
| Mapped categories | 4 (Billing, Technical, HR, General) |
| Priority levels | critical → very_low |
| Tags per ticket | up to 8 |

The dataset is automatically downloaded and cached on first use.

---

## 📂 Project Structure

```
Smart-Support/
├── api/
│   └── main.py              # FastAPI REST API (route, batch, stats, health)
├── evaluation/
│   └── evaluator.py          # Metrics + confusion matrix artifact generation
├── scripts/
│   └── train.py              # CLI training script for all milestones
├── src/
│   ├── config.py             # Centralised configuration (single source of truth)
│   ├── data/
│   │   ├── dataset_loader.py # HuggingFace download, caching, feature engineering
│   │   └── synthetic_generator.py  # Synthetic data for Milestone 1
│   ├── models/
│   │   ├── tfidf_logreg.py   # TF-IDF + Logistic Regression
│   │   ├── tfidf_svc.py      # Char n-gram TF-IDF + LinearSVC
│   │   └── distilbert_classifier.py  # DistilBERT embeddings + LogReg
│   ├── preprocessing/
│   │   └── text_cleaner.py   # Shared text normalisation
│   └── routing/
│       ├── urgency.py        # Regex-based urgency detection (EN + DE)
│       ├── queue.py          # Min-heap priority queue
│       └── router.py         # Dependency-injectable ticket router
├── tests/
│   ├── test_synthetic.py     # Milestone 1 tests (7 tests)
│   ├── test_svc.py           # Milestone 2 tests (4 tests)
│   ├── test_distilbert.py    # Milestone 3 smoke tests (4 tests)
│   └── test_api.py           # API endpoint tests (5 tests)
├── pyproject.toml            # uv-managed dependencies & pytest config
└── README.md
```

---

## 🔬 Milestone Progression

### Milestone 1 — Synthetic LogReg (Baseline)
- **Data**: 24 000 synthetic tickets (4 categories × 6 000)
- **Model**: TF-IDF (word bigrams, 15K features) → Logistic Regression
- **Purpose**: Prove the pipeline works end-to-end

### Milestone 2 — Real Data + LinearSVC
- **Data**: 61 765 real multilingual tickets from HuggingFace
- **Model**: Char n-gram TF-IDF (3–5, 20K features) → LinearSVC
- **Why**: Character n-grams handle German compound words and mixed-language text

### Milestone 3 — DistilBERT (multilingual)
- **Data**: Same real dataset
- **Model**: `distilbert-base-multilingual-cased` [CLS] embeddings → Logistic Regression
- **Why**: Contextual embeddings capture semantic meaning beyond bag-of-words

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model status, supported categories, uptime |
| `POST` | `/route` | Route a single ticket |
| `POST` | `/route/batch` | Route up to 100 tickets in one request |
| `GET` | `/stats` | Live routing statistics (counts by category, urgency) |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🧪 Running Tests

```bash
# Run all 20 tests
uv run pytest -v

# Run a specific milestone
uv run pytest tests/test_synthetic.py -v
uv run pytest tests/test_api.py -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Package manager | uv |
| ML (traditional) | scikit-learn (TF-IDF, LogReg, LinearSVC) |
| ML (deep learning) | PyTorch + HuggingFace Transformers (DistilBERT) |
| Data | HuggingFace Datasets, pandas |
| API | FastAPI + Pydantic v2 |
| Visualisation | matplotlib + seaborn |
| Testing | pytest |

---

## 📜 License

[MIT](LICENSE)
