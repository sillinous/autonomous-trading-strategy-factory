FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ATSF_REGISTRY_PATH=/data/atsf.sqlite3

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY ui ./ui

RUN pip install --no-cache-dir . "uvicorn[standard]>=0.30" \
    && useradd --create-home --uid 10001 atsf \
    && mkdir -p /data \
    && chown -R atsf:atsf /app /data

USER atsf

EXPOSE 8000

# Keep container liveness independent from authenticated readiness.
# /health is intentionally a public process/liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" 

CMD ["uvicorn", "atsf.api:app", "--host", "0.0.0.0", "--port", "8000"]
