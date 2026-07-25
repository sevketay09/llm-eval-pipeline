"""FastAPI application entry point."""
import asyncio
import contextlib
import logging
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
    custom_metrics_router,
    evaluations_router,
    experiments_router,
    failure_clustering_router,
    hitl_router,
    models_router,
    rag_eval_router,
    redteam_router,
    results_router,
    skill_eval_router,
    traces_router,
    ws_router,
)
from api.routers.websocket import set_eval_service
from api.routers.evaluations import _eval_service
from api.routers.experiments import get_service as get_experiment_service
from api.routers.redteam import get_service as get_redteam_service
from api.routers.custom_metrics import get_service as get_custom_metric_service
from api.routers.traces import get_store as get_trace_store


FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"

_AUTOSAVE_INTERVAL_SECONDS = 60
logger = logging.getLogger(__name__)


async def _autosave_loop(experiment_service, redteam_service, custom_metric_service, trace_store, state_dir: Path):
    """Periodic snapshot so a hard crash (kill -9) loses at most one interval's work."""
    while True:
        await asyncio.sleep(_AUTOSAVE_INTERVAL_SECONDS)
        try:
            experiment_service.save_state(state_dir / "experiments.json")
            redteam_service.save_state(state_dir / "redteam_sessions.json")
            custom_metric_service.save(state_dir / "custom_metrics.json")
            await trace_store.save(state_dir / "traces.json")
        except Exception:
            logger.exception("Periodic state autosave failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Share the same EvalService singleton between evaluations router and WS
    set_eval_service(_eval_service)

    # Restore in-memory stores from their last snapshot, if any, so a process
    # restart doesn't silently wipe out experiments/red-team sessions/custom
    # metrics/traces. This does not make these stores shared across multiple
    # worker processes — see README "Persistence Model" for that limitation.
    settings = get_settings()
    state_dir = Path(settings.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    experiment_service = get_experiment_service()
    redteam_service = get_redteam_service()
    custom_metric_service = get_custom_metric_service()
    trace_store = get_trace_store()

    experiment_service.load_state(state_dir / "experiments.json")
    redteam_service.load_state(state_dir / "redteam_sessions.json")
    custom_metric_service.load_from(state_dir / "custom_metrics.json")
    await trace_store.load_from(state_dir / "traces.json")

    autosave_task = asyncio.create_task(
        _autosave_loop(experiment_service, redteam_service, custom_metric_service, trace_store, state_dir)
    )

    yield

    autosave_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await autosave_task

    experiment_service.save_state(state_dir / "experiments.json")
    redteam_service.save_state(state_dir / "redteam_sessions.json")
    custom_metric_service.save(state_dir / "custom_metrics.json")
    await trace_store.save(state_dir / "traces.json")


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
    app.include_router(custom_metrics_router, prefix="/api")
    app.include_router(rag_eval_router, prefix="/api")
    app.include_router(failure_clustering_router, prefix="/api")
    app.include_router(skill_eval_router, prefix="/api")
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
