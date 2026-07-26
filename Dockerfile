ARG BASE_IMAGE=python:3.12-slim

FROM node:22-alpine AS web-builder

WORKDIR /build/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8001

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
COPY --from=web-builder /build/web/dist /app/web/dist

RUN useradd -m -u 10001 appuser && \
    mkdir -p /app/reports /app/logs /app/config && \
    chown -R appuser:appuser /app && \
    chmod +x /app/docker-entrypoint.sh

EXPOSE 8001

# Stays root at container start so the entrypoint can reconcile ownership of the
# bind-mounted reports/logs/config volumes (which may have been written to by a
# different UID) before dropping to appuser to actually run the app.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
