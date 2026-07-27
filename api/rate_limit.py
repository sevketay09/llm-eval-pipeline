"""Lightweight in-process rate limiting for cost-sensitive endpoints.

No external dependency: a single-process sliding-window counter keyed by
(bucket, client IP). Good enough to blunt accidental or malicious request
floods against endpoints that trigger paid LLM calls; not a substitute for
a real gateway-level limiter in a multi-worker/multi-instance deployment.
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    _hits: dict[str, deque] = defaultdict(deque)
    _lock = threading.Lock()

    def __init__(self, bucket: str, limit: int, window_seconds: int):
        self.bucket = bucket
        self.limit = limit
        self.window_seconds = window_seconds

    def __call__(self, request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        key = f"{self.bucket}:{client_ip}"
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for '{self.bucket}': max {self.limit} requests per {self.window_seconds}s",
                )
            hits.append(now)
