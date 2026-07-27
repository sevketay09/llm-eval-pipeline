"""
tracing/sampler.py — Deterministic hash-based online sampler.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from tracing.sdk import EvalTrace


@dataclass
class OnlineSampler:
    """
    rate=0.1 → 10% of traces forwarded to eval.
    Deterministic: same trace_id + seed always yields same decision.
    """
    rate: float = 0.1
    seed: str = "eval"

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")

    def sample(self, trace_id: str) -> bool:
        key = f"{self.seed}:{trace_id}".encode()
        h = int(hashlib.md5(key).hexdigest(), 16)
        bucket = h % 10_000
        return bucket < int(self.rate * 10_000)

    def should_eval(self, trace: EvalTrace) -> bool:
        return self.sample(trace.trace_id)


if __name__ == "__main__":
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="OnlineSampler stats")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", default="eval")
    args = parser.parse_args()

    sampler = OnlineSampler(rate=args.rate, seed=args.seed)
    accepted = sum(sampler.sample(uuid.uuid4().hex) for _ in range(args.n))
    print(f"rate={args.rate}  n={args.n}  accepted={accepted}  actual={accepted/args.n:.3f}")
