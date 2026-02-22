# ══════════════════════════════════════════════════════════════════════════════
# Smart-Support — Backend (FastAPI + ML stack)
# Multi-stage build: deps in builder, lean runtime image.
#
# CPU-only PyTorch is installed to save ~1.5 GB vs the default CUDA wheel.
#
# Build context: project root (needs pyproject.toml + uv.lock)
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS builder

WORKDIR /build

# Install uv from the official distroless image (no curl / apt needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Copy manifests first — layer is cached unless deps change
COPY pyproject.toml uv.lock ./

# Export exact versions from uv.lock to requirements.txt
RUN uv export --frozen --no-dev --no-hashes -o requirements.txt

# Install CPU-only torch FIRST so pip won't re-pull the giant CUDA wheel
# when processing requirements.txt (already-satisfied constraint is skipped).
RUN pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies from PyPI (torch already satisfied)
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: slim runtime ─────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Copy installed packages from builder (no build tools land in this layer)
COPY --from=builder /usr/local/lib/python3.13/site-packages \
                    /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (tests and scripts excluded via .dockerignore)
COPY backend/ backend/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default model variant — override with -e MODEL_VARIANT=distilbert
    MODEL_VARIANT=svc

EXPOSE 8000

# Lightweight HTTP health-check (no curl dependency needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
