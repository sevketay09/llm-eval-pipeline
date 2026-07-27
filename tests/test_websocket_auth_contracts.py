"""Regression tests for the opt-in WebSocket token guard (api/routers/websocket.py)."""
import importlib

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _reload_app_with_token(monkeypatch, token):
    if token is None:
        monkeypatch.delenv("EVAL_API_AUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EVAL_API_AUTH_TOKEN", token)

    from api import config as config_module
    config_module.get_settings.cache_clear()

    from api import main as main_module
    importlib.reload(main_module)
    return main_module.app


def test_no_token_configured_allows_connection(monkeypatch):
    app = _reload_app_with_token(monkeypatch, None)
    client = TestClient(app)
    with client.websocket_connect("/ws/runs") as ws:
        data = ws.receive_json()
        assert "active_runs" in data


def test_token_configured_rejects_missing_token(monkeypatch):
    app = _reload_app_with_token(monkeypatch, "ws-secret")
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws/runs"):
            pass
        assert False, "expected the connection to be rejected"
    except WebSocketDisconnect as exc:
        assert exc.code == 4401


def test_token_configured_allows_correct_token(monkeypatch):
    app = _reload_app_with_token(monkeypatch, "ws-secret")
    client = TestClient(app)
    with client.websocket_connect("/ws/runs?token=ws-secret") as ws:
        data = ws.receive_json()
        assert "active_runs" in data
