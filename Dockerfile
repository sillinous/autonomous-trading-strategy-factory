FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ATSF_REGISTRY_PATH=/data/atsf.sqlite3

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . "uvicorn[standard]>=0.30" \
    && useradd --create-home --uid 10001 atsf \
    && mkdir -p /data \
    && chown -R atsf:atsf /app /data

USER atsf

EXPOSE 8000

CMD ["uvicorn", "atsf.api:app", "--host", "0.0.0.0", "--port", "8000"]
