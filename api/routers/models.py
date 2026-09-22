"""Models CRUD API."""
import time

from fastapi import APIRouter, HTTPException, Depends

from api.rate_limit import RateLimiter
from api.schemas.models import ModelConfig, ModelConfigUpdate, ModelListResponse
from api.services.config_service import ConfigService
from decisions.config import is_decision_provider

router = APIRouter(prefix="/models", tags=["models"])

_test_rate_limit = RateLimiter("model_test", limit=10, window_seconds=60)


def get_config_service() -> ConfigService:
    return ConfigService()


def _mask_api_key(model: dict) -> dict:
    """Replace a stored api_key with a redacted placeholder before it leaves the API.

    Never return the raw value: it may be a literal secret typed into the
    create-model form, not just an ${ENV_VAR} reference.
    """
    masked = dict(model)
    if masked.get("api_key"):
        masked["api_key"] = "***"
    return masked


@router.get("", response_model=ModelListResponse)
def list_models(svc: ConfigService = Depends(get_config_service)):
    models = svc.get_models()
    masked = {model_id: _mask_api_key(cfg) for model_id, cfg in models.items()}
    return ModelListResponse(models=masked, total=len(masked))


@router.get("/embeddings")
def list_embedding_models(svc: ConfigService = Depends(get_config_service)):
    """Distinct from list_models: embedding_models.yaml entries have their own
    shape (embedding_dim, max_sequence_length, ...) that doesn't fit
    ModelConfig, and callers (e.g. the RAG Eval model picker) need to tell the
    two families apart. Registered before /{model_id} so "embeddings" isn't
    swallowed by that path parameter.
    """
    models = svc.get_embedding_models()
    masked = {model_id: _mask_api_key(cfg) for model_id, cfg in models.items()}
    return {"models": masked, "total": len(masked)}


@router.get("/capabilities")
def get_capabilities(svc: ConfigService = Depends(get_config_service)):
    """Split configured models into decision (Jev) vs. generative (LLM) ids.

    Feeds the frontend's DecisionModelSelect and the "test model to evaluate"
    picker. Registered before /{model_id} so "capabilities" isn't swallowed
    by that path parameter (same reasoning as /embeddings above).
    """
    models = svc.get_models()
    decision_ids = [mid for mid, cfg in models.items() if is_decision_provider(cfg.get("provider"))]
    llm_ids = [mid for mid in models if mid not in decision_ids]
    return {
        "decision_available": bool(decision_ids),
        "decision_models": decision_ids,
        "llm_models": llm_ids,
    }


@router.get("/{model_id}")
def get_model(model_id: str, svc: ConfigService = Depends(get_config_service)):
    model = svc.get_model(model_id)
    if model is None:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return {"id": model_id, **_mask_api_key(model)}


@router.post("/{model_id}", status_code=201)
def create_model(
    model_id: str,
    config: ModelConfig,
    svc: ConfigService = Depends(get_config_service),
):
    existing = svc.get_model(model_id)
    if existing is not None:
        raise HTTPException(409, f"Model '{model_id}' already exists")
    svc.upsert_model(model_id, config.model_dump(exclude_none=True))
    return {"id": model_id, "status": "created"}


@router.put("/{model_id}")
def update_model(
    model_id: str,
    config: ModelConfigUpdate,
    svc: ConfigService = Depends(get_config_service),
):
    existing = svc.get_model(model_id)
    if existing is None:
        raise HTTPException(404, f"Model '{model_id}' not found")

    updated = {**existing, **config.model_dump(exclude_none=True)}
    svc.upsert_model(model_id, updated)
    return {"id": model_id, "status": "updated"}


@router.delete("/{model_id}")
def delete_model(model_id: str, svc: ConfigService = Depends(get_config_service)):
    if not svc.delete_model(model_id):
        raise HTTPException(404, f"Model '{model_id}' not found")
    return {"id": model_id, "status": "deleted"}


@router.post("/{model_id}/test", dependencies=[Depends(_test_rate_limit)])
def test_model(model_id: str, svc: ConfigService = Depends(get_config_service)):
    """Fire a single cheap call against `model_id` to confirm the key/endpoint works.

    Never returns the raw error verbatim if it might contain the API key —
    the key is stripped from the message before it leaves this endpoint.
    """
    from api.config import get_settings
    from decisions.config import get_model_config

    stored = svc.get_model(model_id)
    if stored is None:
        raise HTTPException(404, f"Model '{model_id}' not found")

    config_path = get_settings().models_config_path
    resolved = get_model_config(model_id, config_path)
    provider = resolved.get("provider")
    api_key = resolved.get("api_key") or ""

    start = time.perf_counter()
    error = None
    ok = False
    try:
        if is_decision_provider(provider):
            from decisions.config import build_decision_client
            from decisions.types import NoulQ

            client = build_decision_client(model_id, config_path)
            client.decide("ping", {"ok": NoulQ("The text is a greeting or ping")})
            ok = True
        else:
            from adapters.unified_adapter import UnifiedLLMAdapter

            adapter = UnifiedLLMAdapter(resolved, model_key=model_id)
            result = adapter.generate([{"role": "user", "content": "ping"}], max_tokens=5)
            if result.get("error"):
                error = result["error"]
            else:
                ok = True
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller as a masked message
        error = str(exc)
    latency_ms = (time.perf_counter() - start) * 1000

    if error and api_key:
        error = error.replace(api_key, "***")

    return {"ok": ok, "latency_ms": round(latency_ms, 2), "error": error}


@router.post("/import")
def import_models(
    models: dict[str, ModelConfig],
    overwrite: bool = False,
    svc: ConfigService = Depends(get_config_service),
):
    raw = {k: v.model_dump(exclude_none=True) for k, v in models.items()}
    added, skipped = svc.import_models(raw, overwrite=overwrite)
    return {"added": added, "skipped": skipped}


@router.get("/export/yaml")
def export_models_yaml(svc: ConfigService = Depends(get_config_service)):
    import yaml
    models = svc.get_models()
    masked = {model_id: _mask_api_key(cfg) for model_id, cfg in models.items()}
    content = yaml.dump({"models": masked}, default_flow_style=False, allow_unicode=True, sort_keys=False)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content, media_type="text/yaml")
