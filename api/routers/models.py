"""Models CRUD API."""
from fastapi import APIRouter, HTTPException, Depends

from api.schemas.models import ModelConfig, ModelConfigUpdate, ModelListResponse
from api.services.config_service import ConfigService

router = APIRouter(prefix="/models", tags=["models"])


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
