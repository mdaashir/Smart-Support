# ══════════════════════════════════════════════════════════════════════════════
# Smart-Support — Frontend (Dash dashboard)
# Single-stage, ultra-light: only Dash + Plotly + requests.
# No ML libraries, no PyTorch — entire image is ~250 MB.
#
# Build context: ./frontend
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.13-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Points to the backend service name in docker-compose
    API_BASE=http://backend:8000

# Install only the packages the dashboard actually needs
RUN pip install --no-cache-dir \
        dash==4.0.0 \
        dash-bootstrap-components==2.0.4 \
        requests==2.32.5

COPY dash_app.py .

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8050')"

CMD ["python", "dash_app.py"]
