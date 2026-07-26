"""Security regression tests for the /api/models router."""
from fastapi.testclient import TestClient

from api.main import app
from api.routers.models import get_config_service
from api.services.config_service import ConfigService


def _make_client(tmp_path, monkeypatch):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  secret-model:\n"
        "    provider: openai\n"
        "    model_name: gpt-4o\n"
        "    api_key: sk-super-secret-literal-key\n"
        "  env-ref-model:\n"
        "    provider: openai\n"
        "    model_name: gpt-4o-mini\n"
        "    api_key: ${OPENAI_API_KEY}\n"
        "  no-key-model:\n"
        "    provider: ollama\n"
        "    model_name: llama3\n",
        encoding="utf-8",
    )

    def _override():
        return ConfigService(config_path=str(config_path))

    app.dependency_overrides[get_config_service] = _override
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_config_service, None)


def test_list_models_masks_literal_api_key(tmp_path, monkeypatch):
    gen = _make_client(tmp_path, monkeypatch)
    client = next(gen)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"]["secret-model"]["api_key"] == "***"
    assert "sk-super-secret-literal-key" not in resp.text
    next(gen, None)


def test_get_model_masks_api_key(tmp_path, monkeypatch):
    gen = _make_client(tmp_path, monkeypatch)
    client = next(gen)
    resp = client.get("/api/models/env-ref-model")
    assert resp.status_code == 200
    assert resp.json()["api_key"] == "***"
    next(gen, None)


def test_no_key_model_stays_unset(tmp_path, monkeypatch):
    gen = _make_client(tmp_path, monkeypatch)
    client = next(gen)
    resp = client.get("/api/models/no-key-model")
    assert resp.status_code == 200
    assert not resp.json().get("api_key")
    next(gen, None)


def test_export_yaml_masks_api_key(tmp_path, monkeypatch):
    gen = _make_client(tmp_path, monkeypatch)
    client = next(gen)
    resp = client.get("/api/models/export/yaml")
    assert resp.status_code == 200
    assert "sk-super-secret-literal-key" not in resp.text
    assert "***" in resp.text
    next(gen, None)
