# Local Testing Guide

Run the LLM Eval Pipeline locally: a FastAPI backend (port **8001**) and a
React/Vite frontend (port **5173**). The frontend dev server proxies `/api`
and `/ws` to the backend, so start both.

## Docker ile Tek Komut (Önerilen)

Tüm stack (backend + frontend) tek container'da ayağa kalkar:

```bash
# İlk build (NLTK + npm + pip dahil, ~3-5 dk)
docker compose up --build

# Sonraki başlatmalar (build olmadan)
docker compose up

# Arka planda çalıştır
docker compose up -d

# Logları takip et
docker logs llm-eval-dashboard -f

# Pipeline loglarını filtrele
docker logs llm-eval-dashboard -f 2>&1 | grep "│"

# Durdur
docker compose down

# Durdur + volume/cache temizle
docker compose down --volumes

# Image cache dahil her şeyi temizle (sıfırdan build için)
docker compose down --volumes --rmi all

# Sadece build cache temizle
docker builder prune -f

# Tam temizlik (image + cache + volume)
docker compose down --volumes --rmi all && docker builder prune -f
```

> **Not:** `config/` klasörüne yazma izni gereklidir (model ekleme/silme için):
> ```bash
> chmod -R 777 config/
> ```

Uygulama: **http://localhost:8001**

---

## 1. Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- (Optional) API keys for real model calls: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## 2. Backend (FastAPI)

From the repo root:

```bash
# Create & activate a virtualenv (first time only)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) provide model keys for real evaluations
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Run the API on port 8001 (the port the frontend proxies to)
uvicorn api.main:app --reload --port 8001
```

Verify it is up:

```bash
curl http://localhost:8001/api/health   # or open http://localhost:8001/docs
```

## 3. Frontend (React + Vite)

In a second terminal:

```bash
cd web
npm install            # first time only
npm run dev            # serves http://localhost:5173
```

Open **http://localhost:5173**. API/WebSocket requests are proxied to
`localhost:8001` automatically (see `web/vite.config.ts`).

## 4. Smoke test the UI

- **Dashboard** — loads run reports (empty state + "Run Evaluation" CTA if none).
- **⌘K / Ctrl+K** — command palette jumps to any page/action.
- **Playground** — run a prompt A/B experiment.
- **RAG Eval / Custom Metrics / Red-Team** — each has a "?" help popover.
- **Failures** — pick a saved report from the dropdown (or paste JSON) and cluster.
- Errors and successes surface as toasts (bottom-right).

## 5. Production build check (optional)

```bash
cd web
npm run build          # type-check + bundle into web/dist
npm run preview        # serve the built bundle
```

## 6. Backend tests (optional)

```bash
# from repo root, venv active
pytest -q
```

## Troubleshooting

- **Frontend loads but data is empty / network errors** — the backend isn't
  running on `8001`. Start it (step 2) or update the proxy target in
  `web/vite.config.ts`.
- **`uvicorn: command not found`** — the virtualenv isn't active, or
  `pip install -r requirements.txt` didn't run.
- **Port already in use** — change the port (`--port 8002`) and update the
  proxy target in `web/vite.config.ts` to match.
