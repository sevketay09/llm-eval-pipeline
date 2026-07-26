"""Regression tests for the in-process rate limiter (api/rate_limit.py)."""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.rate_limit import RateLimiter


def _build_app(limit: int, window_seconds: int):
    app = FastAPI()
    limiter = RateLimiter(f"test-bucket-{limit}-{window_seconds}", limit=limit, window_seconds=window_seconds)

    @app.get("/limited", dependencies=[Depends(limiter)])
    def limited():
        return {"ok": True}

    return app


def test_allows_requests_under_the_limit():
    client = TestClient(_build_app(limit=3, window_seconds=60))
    for _ in range(3):
        resp = client.get("/limited")
        assert resp.status_code == 200


def test_blocks_requests_over_the_limit():
    client = TestClient(_build_app(limit=2, window_seconds=60))
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200
    resp = client.get("/limited")
    assert resp.status_code == 429


def test_window_resets_after_expiry():
    client = TestClient(_build_app(limit=1, window_seconds=0))
    assert client.get("/limited").status_code == 200
    import time
    time.sleep(0.05)
    assert client.get("/limited").status_code == 200


def test_different_buckets_do_not_interfere():
    app = FastAPI()
    bucket_a = RateLimiter("bucket-a", limit=1, window_seconds=60)
    bucket_b = RateLimiter("bucket-b", limit=1, window_seconds=60)

    @app.get("/a", dependencies=[Depends(bucket_a)])
    def a():
        return {"ok": True}

    @app.get("/b", dependencies=[Depends(bucket_b)])
    def b():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/a").status_code == 200
    assert client.get("/b").status_code == 200
    assert client.get("/a").status_code == 429
    assert client.get("/b").status_code == 429
