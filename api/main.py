"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env before any module that reads os.environ (httpx proxy, openai clients, etc.)
# override=True ensures .env values always win over stale shell env vars
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.config import get_settings
from api.routers import (
    custom_datasets_router,
    evaluations_router,
    experiments_router,
    hitl_router,
    models_router,
    redteam_router,
    results_router,
    traces_router,
    ws_router,
)
from api.routers.websocket import set_eval_service
from api.routers.evaluations import _eval_service


FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Share the same EvalService singleton between evaluations router and WS
    set_eval_service(_eval_service)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(models_router, prefix="/api")
    app.include_router(evaluations_router, prefix="/api")
    app.include_router(results_router, prefix="/api")
    app.include_router(hitl_router, prefix="/api")
    app.include_router(custom_datasets_router, prefix="/api")
    app.include_router(traces_router, prefix="/api")
    app.include_router(experiments_router, prefix="/api")
    app.include_router(redteam_router, prefix="/api")
    app.include_router(ws_router)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    if FRONTEND_DIST_DIR.exists():
        index_file = FRONTEND_DIST_DIR / "index.html"

        @app.get("/", include_in_schema=False)
        async def serve_frontend_root():
            return FileResponse(index_file)

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend(full_path: str):
            candidate = FRONTEND_DIST_DIR / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index_file)

    return app


app = create_app()
