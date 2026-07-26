"""Regression tests for the opt-in bearer-token guard (api/main.py)."""
import importlib
from typing import Optional

from fastapi.testclient import TestClient


def _reload_app_with_token(monkeypatch, token: Optional[str]):
    if token is None:
        monkeypatch.delenv("EVAL_API_AUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EVAL_API_AUTH_TOKEN", token)

    from api import config as config_module
    config_module.get_settings.cache_clear()

    from api import main as main_module
    importlib.reload(main_module)
    return main_module.app


def test_default_no_token_leaves_endpoints_open(monkeypatch):
    app = _reload_app_with_token(monkeypatch, None)
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200


def test_token_set_blocks_unauthenticated_requests(monkeypatch):
    app = _reload_app_with_token(monkeypatch, "secret-token")
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 401


def test_token_set_allows_correct_bearer_token(monkeypatch):
    app = _reload_app_with_token(monkeypatch, "secret-token")
    client = TestClient(app)
    resp = client.get("/api/models", headers={"Authorization": "Bearer secret-token"})
    assert resp.status_code == 200


def test_health_endpoint_always_open(monkeypatch):
    app = _reload_app_with_token(monkeypatch, "secret-token")
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
