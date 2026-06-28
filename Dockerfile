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
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    python -c "import nltk; d='/usr/local/share/nltk_data'; \
nltk.download('punkt_tab', quiet=True, download_dir=d); \
nltk.download('wordnet', quiet=True, download_dir=d); \
nltk.download('punkt', quiet=True, download_dir=d); \
nltk.download('perluniprops', quiet=True, download_dir=d); \
nltk.download('omw-1.4', quiet=True, download_dir=d)"

COPY . .
COPY --from=web-builder /build/web/dist /app/web/dist

RUN useradd -m -u 10001 appuser && \
    mkdir -p /app/reports /app/logs /app/config && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
