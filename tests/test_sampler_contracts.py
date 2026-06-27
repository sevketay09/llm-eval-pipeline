"""Contract tests for tracing/sampler.py — no LLM, no network."""
from __future__ import annotations

import uuid
import pytest

from tracing.sdk import EvalTrace
from tracing.sampler import OnlineSampler


class TestOnlineSampler:
    def test_rate_zero_rejects_all(self):
        s = OnlineSampler(rate=0.0)
        for _ in range(50):
            assert s.sample(uuid.uuid4().hex) is False

    def test_rate_one_accepts_all(self):
        s = OnlineSampler(rate=1.0)
        for _ in range(50):
            assert s.sample(uuid.uuid4().hex) is True

    def test_deterministic_same_id(self):
        s = OnlineSampler(rate=0.5)
        tid = "fixed-trace-id-abc"
        assert s.sample(tid) == s.sample(tid)

    def test_different_seeds_give_different_results(self):
        s1 = OnlineSampler(rate=0.5, seed="alpha")
        s2 = OnlineSampler(rate=0.5, seed="beta")
        diffs = [s1.sample(uuid.uuid4().hex) != s2.sample(uuid.uuid4().hex) for _ in range(100)]
        assert any(diffs)

    def test_approximate_rate_10pct(self):
        s = OnlineSampler(rate=0.1)
        n = 10_000
        accepted = sum(s.sample(uuid.uuid4().hex) for _ in range(n))
        assert 0.07 <= accepted / n <= 0.13

    def test_should_eval_true_at_rate_one(self):
        s = OnlineSampler(rate=1.0)
        t = EvalTrace(trace_id=uuid.uuid4().hex, name="fn")
        assert s.should_eval(t) is True

    def test_should_eval_false_at_rate_zero(self):
        s = OnlineSampler(rate=0.0)
        t = EvalTrace(trace_id=uuid.uuid4().hex, name="fn")
        assert s.should_eval(t) is False

    def test_invalid_rate_above_one_raises(self):
        with pytest.raises(ValueError):
            OnlineSampler(rate=1.5)

    def test_invalid_rate_negative_raises(self):
        with pytest.raises(ValueError):
            OnlineSampler(rate=-0.01)

    def test_default_seed(self):
        s = OnlineSampler(rate=0.5)
        assert s.seed == "eval"
